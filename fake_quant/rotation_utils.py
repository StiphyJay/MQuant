import tqdm
import torch
import typing
from fake_quant import module_util
from fake_quant import utils
from fake_quant.hadamard_utils import (
    random_hadamard_matrix,
    apply_exact_had_to_linear,
    is_pow2,
)
from fast_hadamard_transform import hadamard_transform


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64).to(W_)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, "bias"):
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)


def fuse_merger_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        w_o, w_i = W_.shape
        size = layernorm.weight.shape[0]
        linear.weight.data = (
            (W_.view(w_o, -1, size) * layernorm.weight.double())
            .to(linear_dtype)
            .view(w_o, w_i)
        )

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64).to(W_)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_.view(w_o, -1, size), layernorm.bias.double()
            ).sum(dim=-1)
            linear.bias.data = linear.bias.data.to(linear_dtype)

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, "bias"):
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)


def bake_mean_into_conv(conv: torch.nn.Conv2d) -> None:
    """
    This function takes a convolutional layer and subtracts the means from the
    weights and biases. This will result in the convolutional layer performing
    the mean substitution which is usually done inside layernorm.
    """
    conv_dtype = conv.weight.dtype
    W_ = conv.weight.data.double()
    conv.weight.data = W_ - W_.mean(dim=0, keepdim=True)
    conv.weight.data = conv.weight.data.to(conv_dtype)
    if conv.bias is not None:
        b_ = conv.bias.data.double()
        conv.bias.data = b_ - b_.mean()
        conv.bias.data = conv.bias.data.to(conv_dtype)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def fuse_qwenvl_layer_norms(model, args):
    print("fuse qwenvl layer norms")
    if not args.no_fuse_visual_clip:
        # fuse qwenvl visual transformer layer norms

        for layer in model.transformer.visual.transformer.resblocks:
            fuse_ln_linear(
                layer.ln_1, [layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj]
            )
            fuse_ln_linear(layer.ln_2, [layer.mlp.c_fc])

            bake_mean_into_linear(layer.attn.out_proj)
            bake_mean_into_linear(layer.mlp.c_proj)

        module_util.replace_modules(
            model.transformer.visual.transformer.resblocks,
            torch.nn.LayerNorm,
            lambda _: module_util.RMSN(model.config.visual["width"], eps=1e-6),
            replace_layers=False,
        )

    if not args.no_fuse_visual_cross_attn:
        # fuse qwenvl visual cross attention
        dtype = model.transformer.visual.attn_pool.pos_embed.data.dtype
        model.transformer.visual.attn_pool.pos_embed_kv.data = (
            model.transformer.visual.attn_pool.pos_embed_kv.data.double()
            / model.transformer.visual.attn_pool.ln_kv.weight.data.double()
        ).to(dtype)
        fuse_ln_linear(
            model.transformer.visual.attn_pool.ln_kv,
            [
                model.transformer.visual.attn_pool.attn.k_proj,
                model.transformer.visual.attn_pool.attn.v_proj,
            ],
        )

        model.transformer.visual.attn_pool.pos_embed.data = (
            model.transformer.visual.attn_pool.pos_embed.data.double()
            / model.transformer.visual.attn_pool.ln_q.weight.data.double()
        ).to(dtype)
        fuse_ln_linear(
            model.transformer.visual.attn_pool.ln_q,
            [model.transformer.visual.attn_pool.attn.q_proj],
        )

        model.transformer.visual.attn_pool.query.data = (
            model.transformer.visual.attn_pool.query.data
            - model.transformer.visual.attn_pool.query.data.double().mean(
                dim=-1, keepdim=True
            )
        ).to(dtype)
        bake_mean_into_linear(model.transformer.visual.attn_pool.kv_proj)
        module_util.replace_modules(
            model.transformer.visual.attn_pool,
            torch.nn.LayerNorm,
            lambda _: module_util.RMSN(model.config.visual["output_dim"], eps=1e-6),
            replace_layers=False,
        )

        # 修改了模型结构，删除了 proj,新增了一个proj_fc
        # # fuse qwenvl visual ln_post  # model.transformer.visual.proj是一个embedding的参数，不是一个 linear，这样的话，ln_post的bias不能融合进去
        # # 所以这不转为rmsnorm，因为rmsnorm的bias是0，这里的bias是有值的，这边让ln的weight为 1,ln的输入均值为 0，这样的话out_proj和proj可以做 rotate
        # linear_dtype = model.transformer.visual.proj.data.dtype
        # W_ = model.transformer.visual.proj.data.double()
        # model.transformer.visual.proj.data = (
        #     model.transformer.visual.ln_post.weight.data.double().unsqueeze(1) * W_
        # ).to(linear_dtype)
        # if hasattr(model.transformer.visual.ln_post, "bias"):
        #     model.transformer.visual.ln_post.bias.data = (
        #         model.transformer.visual.ln_post.bias.data.double()
        #         / model.transformer.visual.ln_post.weight.data.double()
        #     ).to(linear_dtype)
        # model.transformer.visual.ln_post.weight.data = torch.ones_like(
        #     model.transformer.visual.ln_post.weight.data
        # )
        fuse_ln_linear(
            model.transformer.visual.ln_post, [model.transformer.visual.proj_fc]
        )
        # 输出均值设为 0
        bake_mean_into_linear(model.transformer.visual.attn_pool.attn.out_proj)
        model.transformer.visual.ln_post = module_util.RMSN(
            model.config.visual["output_dim"], eps=1e-6
        )

    if not args.no_fuse_llm:
        # fuse qwen 7b
        for layer in model.transformer.h:
            fuse_ln_linear(layer.ln_2, [layer.mlp.w1, layer.mlp.w2])
            fuse_ln_linear(
                layer.ln_1, [layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj]
            )

        # 最后一个rmsnorm需要和ln_head合并
        fuse_ln_linear(model.transformer.ln_f, [model.lm_head])


def rotate_conv(layer, Q_v, embed_dims):
    dtype = layer.weight.dtype
    weight_shape = layer.weight.data.shape
    layer.weight.data = (
        torch.matmul(Q_v.T, layer.weight.data.double().view(embed_dims, -1))
        .to(dtype)
        .view(weight_shape)
    )
    if layer.bias is not None:
        layer.bias.data = torch.matmul(layer.bias.data.double(), Q_v).to(dtype)

def rotate_embeddings(model, Q, is_minicpmv=False) -> None:
    if not is_minicpmv:
        dtype = model.transformer.wte.weight.data.dtype
        W_ = model.transformer.wte.weight.data.to(dtype=torch.float64)
        model.transformer.wte.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

        W_ = model.transformer.visual.proj_fc.weight.data.to(dtype=torch.float64)
        model.transformer.visual.proj_fc.weight.data = torch.matmul(Q.T, W_).to(
            dtype=dtype
        )
        if model.transformer.visual.proj_fc.bias is not None:
            b = model.transformer.visual.proj_fc.bias.data.to(dtype=torch.float64)
            model.transformer.visual.proj_fc.bias.data = torch.matmul(b, Q).to(
                dtype=dtype
            )
    else:
        dtype = model.llm.model.embed_tokens.weight.data.dtype
        W_ = model.llm.model.embed_tokens.weight.data.to(dtype=torch.float64)
        model.llm.model.embed_tokens.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

        W_ = model.resampler.proj_fc.weight.data.to(dtype=torch.float64)
        model.resampler.proj_fc.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
        if model.resampler.proj_fc.bias is not None:
            b = model.resampler.proj_fc.bias.data.to(dtype=torch.float64)
            model.resampler.proj_fc.bias.data = torch.matmul(b, Q).to(dtype=dtype)


def rotate_head(model, Q: torch.Tensor, is_minicpmv=False) -> None:
    # Rotate the head.
    if not is_minicpmv:
        dtype = model.lm_head.weight.data.dtype
        W_ = model.lm_head.weight.data.to(dtype=torch.float64)
        model.lm_head.weight.data = torch.matmul(W_, Q).to(dtype=dtype)
    else:
        dtype = model.llm.lm_head.weight.data.dtype
        W_ = model.llm.lm_head.weight.data.to(dtype=torch.float64)
        model.llm.lm_head.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_kv_proj(model, Q: torch.Tensor, is_minicpmv=False) -> None:
    # Rotate the head.
    kv_proj = (
        model.transformer.visual.attn_pool.kv_proj
        if not is_minicpmv
        else model.resampler.kv_proj
    )
    dtype = kv_proj.weight.dtype
    W_ = kv_proj.weight.to(dtype=torch.float64)
    kv_proj.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_attention_inputs(layer, Q, is_minicpmv=False) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.

    layer_list = (
        [layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj]
        if not is_minicpmv
        else [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
    )
    for W in layer_list:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_cross_attention_inputs(layer, Q_q, Q_kv) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.attn.q_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q_q).to(dtype=dtype)

    for W in [layer.attn.k_proj, layer.attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q_kv).to(dtype=dtype)


def rotate_cross_embeddings(model, Q_q, Q_kv, is_minicpmv=False):
    if not is_minicpmv:
        # q rotate
        dtype = model.transformer.visual.attn_pool.query.data.dtype
        W_ = model.transformer.visual.attn_pool.query.data.to(dtype=torch.float64)
        model.transformer.visual.attn_pool.query.data = torch.matmul(W_, Q_q).to(dtype)
        model.transformer.visual.attn_pool.pos_embed.data = torch.matmul(
            model.transformer.visual.attn_pool.pos_embed.data.to(dtype=torch.float64),
            Q_q,
        ).to(dtype)

        # kv rotate
        W_ = model.transformer.visual.attn_pool.kv_proj.weight.data.to(
            dtype=torch.float64
        )
        model.transformer.visual.attn_pool.kv_proj.weight.data = torch.matmul(
            Q_kv.T, W_
        ).to(dtype)
        if model.transformer.visual.attn_pool.kv_proj.bias is not None:
            b = model.transformer.visual.attn_pool.kv_proj.bias.data.to(
                dtype=torch.float64
            )
            model.transformer.visual.attn_pool.kv_proj.bias.data = torch.matmul(
                b, Q_kv
            ).to(dtype)

        model.transformer.visual.attn_pool.pos_embed_kv.data = torch.matmul(
            model.transformer.visual.attn_pool.pos_embed_kv.data.to(
                dtype=torch.float64
            ),
            Q_kv,
        ).to(dtype)
    else:
        # q rotate
        dtype = model.resampler.query.data.dtype
        W_ = model.resampler.query.data.to(dtype=torch.float64)
        model.resampler.query.data = torch.matmul(W_, Q_q).to(dtype)

        # kv rotate
        W_ = model.resampler.kv_proj.weight.data.to(dtype=torch.float64)
        model.resampler.kv_proj.weight.data = torch.matmul(Q_kv.T, W_).to(dtype)
        if model.resampler.kv_proj.bias is not None:
            b = model.resampler.kv_proj.bias.data.to(dtype=torch.float64)
            model.resampler.kv_proj.bias.data = torch.matmul(b, Q_kv).to(dtype)

        model.resampler.pos_embed.data = torch.matmul(
            model.resampler.pos_embed.data.to(dtype=torch.float64),
            Q_kv,
        ).to(dtype)


def rotate_attention_output(layer, Q, is_visual=False) -> None:
    # Rotate output matrix of the self-attention layer.
    if is_visual:
        if hasattr(layer, "attn"):
            W = layer.attn.out_proj
        else:
            W = layer.self_attn.out_proj
    else:
        if hasattr(layer, "attn"):
            W = layer.attn.c_proj
        else:
            W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def rotate_mlp_input(layer, Q, is_visual=False, is_minicpmv=False) -> None:
    # Rotate the MLP input weights.
    if is_visual:
        mlp_inputs = [layer.mlp.c_fc if hasattr(layer.mlp, "c_fc") else layer.mlp.fc1]
    else:
        mlp_inputs = (
            [layer.mlp.w1, layer.mlp.w2]
            if not is_minicpmv
            else [layer.mlp.up_proj, layer.mlp.gate_proj]
        )
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_mlp_output(layer, Q, online_hadamard=False):
    out_layer = (
        layer.mlp.c_proj
        if hasattr(layer.mlp, "c_proj")
        else (layer.mlp.down_proj if hasattr(layer.mlp, "down_proj") else layer.mlp.fc2)
    )
    # out_layer = layer.mlp.c_proj if hasattr(layer.mlp, "c_proj") else layer.mlp.fc2
    # Rotate the MLP output weights and bias.
    dtype = out_layer.weight.data.dtype
    W_ = out_layer.weight.data.to(dtype=torch.float64)
    out_layer.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)

    if online_hadamard:
        # input做hadamard变换, 它的输入做在线hadamard变换
        apply_exact_had_to_linear(
            out_layer, had_dim=-1, output=False
        )  # apply exact (inverse) hadamard on the weights of mlp output

    if out_layer.bias is not None:
        b = out_layer.bias.data.to(dtype=torch.float64)
        out_layer.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, is_visual=False, is_minicpmv=False):
    if is_visual:
        if hasattr(layer, "attn"):
            v_proj = layer.attn.v_proj
            o_proj = layer.attn.out_proj
        else:
            v_proj = layer.self_attn.v_proj
            o_proj = layer.self_attn.out_proj
    else:
        if not is_minicpmv:
            v_proj = layer.attn.v_proj
            o_proj = layer.attn.c_proj
        else:
            v_proj = layer.self_attn.v_proj
            o_proj = layer.self_attn.o_proj
    if is_visual:
        Q = get_orthogonal_matrix(head_dim, mode="hadamard")
        dtype = v_proj.weight.dtype
        W_ = v_proj.weight.data.to(dtype=torch.float64).T.reshape(
            -1, head_num, head_dim
        )

        v_proj.weight.data = (
            torch.matmul(W_, Q).reshape(-1, head_num * head_dim).T.to(dtype=dtype)
        )
        if v_proj.bias is not None:
            b = v_proj.bias.data.to(dtype=torch.float64).reshape(head_num, head_dim)
            v_proj.bias.data = torch.matmul(b, Q).to(dtype=dtype).reshape(-1)

        W_ = o_proj.weight.data.to(dtype=torch.float64).reshape(-1, head_num, head_dim)
        o_proj.weight.data = (
            torch.matmul(W_, Q).reshape(-1, head_num * head_dim).to(dtype=dtype)
        )

    else:
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)


def rotate_o_ln_proj(layer, Q_o):
    # rotate out_proj, ln bias, proj
    layer_dtype = layer.attn_pool.attn.out_proj.weight.data
    layer.attn_pool.attn.out_proj.weight.data = torch.matmul(
        Q_o.T, layer.attn_pool.attn.out_proj.weight.data.double()
    ).to(layer_dtype)

    layer.ln_post.bias.data = torch.matmul(layer.ln_post.bias.data.double(), Q_o).to(
        layer_dtype
    )

    layer.proj.data = torch.matmul(
        Q_o.T,
        layer.proj.data.double(),
    ).to(layer_dtype)


def rotate_o_ln_proj_fc(layer, Q_o, is_minicpmv=False):
    if not is_minicpmv:
        # rotate out_proj, ln bias, proj
        layer_dtype = layer.attn_pool.attn.out_proj.weight.data
        layer.attn_pool.attn.out_proj.weight.data = torch.matmul(
            Q_o.T, layer.attn_pool.attn.out_proj.weight.data.double()
        ).to(layer_dtype)
        if layer.attn_pool.attn.out_proj.bias is not None:
            layer.attn_pool.attn.out_proj.bias.data = torch.matmul(
                layer.attn_pool.attn.out_proj.bias.data.double(), Q_o
            ).to(layer_dtype)

        layer.proj_fc.weight.data = torch.matmul(
            layer.proj_fc.weight.data.double(), Q_o
        ).to(layer_dtype)
    else:
        # rotate out_proj, ln bias, proj
        layer_dtype = layer.attn.out_proj.weight.data
        layer.attn.out_proj.weight.data = torch.matmul(
            Q_o.T, layer.attn.out_proj.weight.data.double()
        ).to(layer_dtype)
        if layer.attn.out_proj.bias is not None:
            layer.attn.out_proj.bias.data = torch.matmul(
                layer.attn.out_proj.bias.data.double(), Q_o
            ).to(layer_dtype)

        layer.proj_fc.weight.data = torch.matmul(
            layer.proj_fc.weight.data.double(), Q_o
        ).to(layer_dtype)


@torch.inference_mode()
def rotate_model(model, args):
    print("rotate model")
    if args.rotate_visual_clip:
        # rotate visual transformer
        num_heads = model.config.visual["heads"]
        head_dim = model.config.visual["width"] // num_heads
        Q_v = get_orthogonal_matrix(model.config.visual["width"], args.rotate_mode)
        for idx, layer in enumerate(
            tqdm.tqdm(
                model.transformer.visual.transformer.resblocks,
                unit="layer",
                desc="Rotating Visual CLIP",
            )
        ):
            rotate_attention_inputs(
                model.transformer.visual.transformer.resblocks[idx], Q_v
            )
            rotate_attention_output(
                model.transformer.visual.transformer.resblocks[idx], Q_v, is_visual=True
            )
            rotate_mlp_input(
                model.transformer.visual.transformer.resblocks[idx], Q_v, is_visual=True
            )
            rotate_mlp_output(
                model.transformer.visual.transformer.resblocks[idx],
                Q_v,
                args.online_visual_hadamard,
            )

            rotate_ov_proj(
                model.transformer.visual.transformer.resblocks[idx],
                num_heads,
                head_dim,
                is_visual=True,
            )
        rotate_kv_proj(model, Q_v)

        from fake_quant.quant_utils import ActRotateWrapper

        model.transformer.visual.fc_sub_mean.weight.data = (
            Q_v.T @ model.transformer.visual.fc_sub_mean.weight.data.double()
        ).to(model.transformer.visual.fc_sub_mean.weight.data.dtype)

        utils.cleanup_memory()

    if args.rotate_visual_cross_attn:
        print("\n Rotating Visual Cross Attention \n")
        embed_dim = model.transformer.visual.attn_pool.embed_dim
        num_heads = model.transformer.visual.attn_pool.num_heads
        # rotate visual cross attention
        Q_q = get_orthogonal_matrix(model.config.visual["output_dim"], args.rotate_mode)
        Q_kv = get_orthogonal_matrix(
            model.config.visual["output_dim"], args.rotate_mode
        )
        rotate_cross_embeddings(model, Q_q, Q_kv)
        rotate_cross_attention_inputs(model.transformer.visual.attn_pool, Q_q, Q_kv)

        rotate_ov_proj(
            model.transformer.visual.attn_pool,
            num_heads,
            embed_dim // num_heads,
            is_visual=True,
        )

        Q_o = get_orthogonal_matrix(model.config.visual["output_dim"], args.rotate_mode)
        rotate_o_ln_proj_fc(model.transformer.visual, Q_o)
        utils.cleanup_memory()

    if args.rotate_llm:
        if args.online_llm_hadamard:
            model.config.need_pad = False
            from fake_quant.hadamard_utils import auto_pad_size

            new_intermediate_size = auto_pad_size(model.config.intermediate_size)
            if new_intermediate_size != model.config.intermediate_size:
                for name, module in model.named_modules():
                    if (
                        "mlp.c_proj" in name
                        and "transformer.h" in name
                        and isinstance(module, torch.nn.Linear)
                    ):
                        new_module = torch.nn.Linear(
                            new_intermediate_size,
                            module.out_features,
                            dtype=module.weight.dtype,
                        ).to(module.weight.device)
                        with torch.no_grad():
                            new_module.weight[:, : module.in_features] = (
                                module.weight.data
                            )
                            if module.bias is not None:
                                new_module.bias[: module.out_features].copy_(
                                    module.bias
                                )
                        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                        if parent_name:  # 如果模块不是顶层模块
                            parent = dict(model.named_modules())[parent_name]
                            setattr(parent, name.split(".")[-1], new_module)
                        else:  # 如果模块是顶层模块
                            setattr(model, name, new_module)
                model.config.intermediate_size = new_intermediate_size
                model.config.need_pad = True
        # rotate qwen 7b
        Q = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)

        config = model.config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads

        rotate_embeddings(model, Q)
        rotate_head(model, Q)
        utils.cleanup_memory()
        for idx, layer in enumerate(
            tqdm.tqdm(model.transformer.h, unit="layer", desc="Rotating")
        ):
            rotate_attention_inputs(model.transformer.h[idx], Q)
            rotate_attention_output(model.transformer.h[idx], Q)
            rotate_mlp_input(model.transformer.h[idx], Q)
            rotate_mlp_output(model.transformer.h[idx], Q, args.online_llm_hadamard)
            rotate_ov_proj(model.transformer.h[idx], num_heads, head_dim)
        utils.cleanup_memory()

