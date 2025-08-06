import torch
import tqdm
import typing
from fake_quant.rotation_utils import (
    fuse_ln_linear,
    bake_mean_into_conv,
    bake_mean_into_linear,
    rotate_conv,
)
from fake_quant.rotation_utils import get_orthogonal_matrix
from fake_quant import module_util
from fake_quant import utils
from fake_quant.hadamard_utils import apply_exact_had_to_linear


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


def fuse_qwen2vl_layer_norms(model, args):
    print("fuse qwen2vl layer norms")
    if not args.no_fuse_visual_clip:
        # fuse internvl visual transformer layer norms
        bake_mean_into_conv(model.model.visual.patch_embed.proj)
        for layer in model.model.visual.blocks:
            fuse_ln_linear(layer.norm1, [layer.attn.qkv])
            fuse_ln_linear(layer.norm2, [layer.mlp.fc1])

            bake_mean_into_linear(layer.attn.proj)
            bake_mean_into_linear(layer.mlp.fc2)

        module_util.replace_modules(
            model.model.visual.blocks,
            torch.nn.LayerNorm,
            lambda _: module_util.RMSN(
                model.model.visual.patch_embed.embed_dim, eps=1e-6
            ),
            replace_layers=False,
        )

    if not args.no_fuse_visual_cross_attn:
        fuse_merger_linear(
            model.model.visual.merger.ln_q, [model.model.visual.merger.mlp[0]]
        )
        module_util.replace_modules(
            model.model.visual.merger,
            torch.nn.LayerNorm,
            lambda _: module_util.RMSN(
                model.model.visual.patch_embed.embed_dim,
                eps=1e-6,
            ),
            replace_layers=False,
        )

    if not args.no_fuse_llm:
        # fuse internvl2-8b
        for layer in model.model.model.layers:
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj]
            )

        # 最后一个rmsnorm需要和ln_head合并
        fuse_ln_linear(model.model.model.norm, [model.model.lm_head])


def rotate_qwen2vl_attention_inputs(layer, Q, is_visual=False) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.

    layer_list = (
        [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
        if not is_visual
        else [layer.attn.qkv]
    )
    for W in layer_list:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

def rotate_qwen2vl_attention_output(layer, Q, is_visual=False) -> None:
    # Rotate output matrix of the self-attention layer.
    if is_visual:
        W = layer.attn.proj
    else:
        W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def rotate_qwen2vl_mlp_input(layer, Q, is_visual=False) -> None:
    # Rotate the MLP input weights.
    if is_visual:
        mlp_inputs = [layer.mlp.fc1]
    else:
        mlp_inputs = [layer.mlp.gate_proj, layer.mlp.up_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_qwen2vl_mlp_output(layer, Q, is_visual=False, online_hadamard=False):
    out_layer = layer.mlp.fc2 if is_visual else layer.mlp.down_proj
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


def rotate_qwen2vl_ov_proj(layer, head_num, head_dim, is_visual=False):
    if is_visual:
        qkv = layer.attn.qkv
        o_proj = layer.attn.proj

        q_weight, k_weight, v_weight = qkv.weight.data.chunk(3)

        Q = get_orthogonal_matrix(head_dim, mode="hadamard")
        dtype = v_weight.dtype
        W_ = v_weight.to(dtype=torch.float64).T.reshape(-1, head_num, head_dim)

        v_weight = (
            torch.matmul(W_, Q).reshape(-1, head_num * head_dim).T.to(dtype=dtype)
        )

        qkv.weight.data = torch.cat([q_weight, k_weight, v_weight], 0).contiguous()
        if qkv.bias is not None:
            q_bias, k_bias, v_bias = qkv.bias.data.chunk(3)
            v_bias = v_bias.to(dtype=torch.float64).reshape(head_num, head_dim)
            v_bias = torch.matmul(v_bias, Q).to(dtype=dtype).reshape(-1)
            qkv.bias.data = torch.cat([q_bias, k_bias, v_bias], -1).contiguous()

        W_ = o_proj.weight.data.to(dtype=torch.float64).reshape(-1, head_num, head_dim)
        o_proj.weight.data = (
            torch.matmul(W_, Q).reshape(-1, head_num * head_dim).to(dtype=dtype)
        )
    else:
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)

def rotate_visual_merger(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    dtype = model.visual.merger.mlp[0].weight.dtype

    q_shape = Q.shape[0]
    o_shape, i_shape = model.visual.merger.mlp[0].weight.shape

    W_ = (
        model.visual.merger.mlp[0]
        .weight.to(dtype=torch.float64)
        .reshape(o_shape, -1, q_shape)
    )
    model.visual.merger.mlp[0].weight.data = (
        torch.matmul(W_, Q).to(dtype=dtype).reshape(o_shape, i_shape).contiguous()
    )


def rotate_qwen2vl_embeddings(model, Q) -> None:
    Q = Q.to(model.model.embed_tokens.weight.device)
    dtype = model.model.embed_tokens.weight.data.dtype
    W_ = model.model.embed_tokens.weight.data.to(dtype=torch.float64)
    model.model.embed_tokens.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

    Q = Q.to(model.visual.merger.mlp[2].weight.device)
    W_ = model.visual.merger.mlp[2].weight.data.to(dtype=torch.float64)
    model.visual.merger.mlp[2].weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if model.visual.merger.mlp[2].bias is not None:
        b = model.visual.merger.mlp[2].bias.data.to(dtype=torch.float64)
        model.visual.merger.mlp[2].bias.data = torch.matmul(b, Q).to(dtype=dtype)


def rotate_qwen2vl_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    dtype = model.lm_head.weight.data.dtype
    W_ = model.lm_head.weight.data.to(dtype=torch.float64)
    model.lm_head.weight.data = torch.matmul(W_, Q.to(W_.device)).to(dtype=dtype)

@torch.no_grad()
def rotate_qwen2vl_model(model, args):
    print("rotate model")
    if args.rotate_visual_clip:
        # rotate visual transformer
        num_heads = model.visual.blocks[0].attn.num_heads
        head_dim = model.visual.blocks[0].attn.qkv.in_features // num_heads
        Q_v = get_orthogonal_matrix(
            model.visual.blocks[0].attn.qkv.in_features, args.rotate_mode
        )

        rotate_conv(
            model.visual.patch_embed.proj,
            Q_v,
            model.visual.blocks[0].attn.qkv.in_features,
        )

        for idx, layer in enumerate(
            tqdm.tqdm(
                model.visual.blocks,
                unit="layer",
                desc="Rotating Visual CLIP",
            )
        ):
            rotate_qwen2vl_attention_inputs(layer, Q_v, is_visual=True)
            rotate_qwen2vl_attention_output(layer, Q_v, is_visual=True)
            rotate_qwen2vl_mlp_input(layer, Q_v, is_visual=True)
            rotate_qwen2vl_mlp_output(
                layer,
                Q_v,
                True,
                args.online_visual_hadamard,
            )

            rotate_qwen2vl_ov_proj(
                layer,
                num_heads,
                head_dim,
                is_visual=True,
            )

        rotate_visual_merger(model, Q_v)
        utils.cleanup_memory()

    if args.rotate_visual_cross_attn:
        print("\n Rotating Visual Cross Attention \n")
        pass

    if args.rotate_llm:
        if args.online_llm_hadamard:
            model.config.need_pad = False
            from fake_quant.hadamard_utils import auto_pad_size

            new_intermediate_size = auto_pad_size(model.config.intermediate_size)
            if new_intermediate_size != model.config.intermediate_size:
                for name, module in model.named_modules():
                    if "down_proj" in name and isinstance(module, torch.nn.Linear):
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
        Q = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)

        num_attention_heads = model.config.num_attention_heads
        num_key_value_head = model.config.num_key_value_heads
        model_dim = model.config.hidden_size
        head_dim = model_dim // num_attention_heads

        rotate_qwen2vl_embeddings(model, Q)
        rotate_qwen2vl_head(model, Q)
        utils.cleanup_memory()
        for idx, layer in enumerate(
            tqdm.tqdm(model.model.layers, unit="layer", desc="LLM Rotating")
        ):
            layer_device = next(layer.parameters()).device
            Q = Q.to(layer_device)
            rotate_qwen2vl_attention_inputs(layer, Q)
            rotate_qwen2vl_attention_output(layer, Q)
            rotate_qwen2vl_mlp_input(layer, Q)
            rotate_qwen2vl_mlp_output(layer, Q, False, args.online_llm_hadamard)
            rotate_qwen2vl_ov_proj(
                layer, num_attention_heads, head_dim, is_visual=False
            )
        utils.cleanup_memory()
