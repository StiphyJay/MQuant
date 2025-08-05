import torch
import tqdm
from fake_quant.rotation_utils import (
    fuse_ln_linear,
    bake_mean_into_conv,
    bake_mean_into_linear,
    rotate_conv,
)
from fake_quant.rotation_utils import get_orthogonal_matrix
from fake_quant import module_util
from fake_quant.hadamard_utils import apply_exact_had_to_linear


def rotate_internvl_attention_inputs(layer, Q, is_visual=False) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.

    layer_list = [layer.attention.wqkv] if not is_visual else [layer.attn.qkv]
    for W in layer_list:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_internvl_attention_output(layer, Q, is_visual=False) -> None:
    # Rotate output matrix of the self-attention layer.
    if is_visual:
        W = layer.attn.proj
    else:
        W = layer.attention.wo

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def rotate_internvl_mlp_input(layer, Q, is_visual=False) -> None:
    # Rotate the MLP input weights.
    if is_visual:
        mlp_inputs = [layer.mlp.fc1]
    else:
        mlp_inputs = [layer.feed_forward.w1, layer.feed_forward.w3]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_internvl_mlp_output(layer, Q, is_visual=False, online_hadamard=False):
    out_layer = layer.mlp.fc2 if is_visual else layer.feed_forward.w2
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


def rotate_internvl_ov_proj(layer, head_num, head_dim, is_visual=False):
    if is_visual:
        qkv = layer.attn.qkv
        o_proj = layer.attn.proj
    else:
        qkv = layer.attention.wqkv
        o_proj = layer.attention.wo

    q_weight, k_weight, v_weight = qkv.weight.data.chunk(3)

    Q = get_orthogonal_matrix(head_dim, mode="hadamard")
    dtype = v_weight.dtype
    W_ = v_weight.to(dtype=torch.float64).T.reshape(-1, head_num, head_dim)

    v_weight = torch.matmul(W_, Q).reshape(-1, head_num * head_dim).T.to(dtype=dtype)

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


def rotate_internvl_ov_proj_v2(layer, q_head_num, kv_head_num, head_dim):
    qkv = layer.attention.wqkv
    o_proj = layer.attention.wo

    num_key_value_groups = q_head_num // kv_head_num

    qkv_weight = qkv.weight.data
    qkv_o, qkv_i = qkv_weight.shape

    qkv_weight = qkv_weight.T.contiguous()
    qkv_weight = qkv_weight.reshape(qkv_i, -1, 2 + num_key_value_groups, head_dim)

    v_weight = qkv_weight[..., -1, :]
    Q = get_orthogonal_matrix(head_dim, mode="hadamard")
    dtype = v_weight.dtype
    v_weight = torch.matmul(v_weight.double(), Q).to(dtype=dtype)
    qkv_weight[..., -1, :] = v_weight
    qkv_weight = qkv_weight.reshape(qkv_i, qkv_o).T.contiguous()

    qkv.weight.data = qkv_weight

    W_ = o_proj.weight.data.to(dtype=torch.float64).reshape(-1, q_head_num, head_dim)
    o_proj.weight.data = (
        torch.matmul(W_, Q).reshape(-1, q_head_num * head_dim).to(dtype=dtype)
    )


def rotate_mlp1(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    dtype = model.mlp1[1].weight.dtype

    q_shape = Q.shape[0]
    o_shape, i_shape = model.mlp1[1].weight.shape

    W_ = model.mlp1[1].weight.to(dtype=torch.float64).reshape(o_shape, -1, q_shape)
    model.mlp1[1].weight.data = (
        torch.matmul(W_, Q).to(dtype=dtype).reshape(o_shape, i_shape).contiguous()
    )


def rotate_internvl_embeddings(model, Q) -> None:
    dtype = model.language_model.model.tok_embeddings.weight.data.dtype
    W_ = model.language_model.model.tok_embeddings.weight.data.to(dtype=torch.float64)
    model.language_model.model.tok_embeddings.weight.data = torch.matmul(W_, Q).to(
        dtype=dtype
    )

    W_ = model.mlp1[3].weight.data.to(dtype=torch.float64)
    model.mlp1[3].weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if model.mlp1[3].bias is not None:
        b = model.mlp1[3].bias.data.to(dtype=torch.float64)
        model.mlp1[3].bias.data = torch.matmul(b, Q).to(dtype=dtype)


def rotate_internvl_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    dtype = model.language_model.output.weight.data.dtype
    W_ = model.language_model.output.weight.data.to(dtype=torch.float64)
    model.language_model.output.weight.data = torch.matmul(W_, Q).to(dtype=dtype)



def fuse_internvl_layer_norms(model, args):
    print("fuse internvl layer norms")
    if not args.no_fuse_visual_clip:
        # fuse internvl visual transformer layer norms
        bake_mean_into_conv(model.model.vision_model.embeddings.patch_embedding)
        dtype = model.model.vision_model.embeddings.class_embedding.data.dtype
        model.model.vision_model.embeddings.class_embedding.data = (
            model.model.vision_model.embeddings.class_embedding.data
            - model.model.vision_model.embeddings.class_embedding.data.double().mean(
                dim=-1, keepdim=True
            )
        ).to(dtype)
        model.model.vision_model.embeddings.position_embedding.data = (
            model.model.vision_model.embeddings.position_embedding.data
            - model.model.vision_model.embeddings.position_embedding.data.double().mean(
                dim=-1, keepdim=True
            )
        ).to(dtype)

        for layer in model.model.vision_model.encoder.layers:
            fuse_ln_linear(layer.norm1, [layer.attn.qkv])
            fuse_ln_linear(layer.norm2, [layer.mlp.fc1])

            bake_mean_into_linear(layer.attn.proj)
            bake_mean_into_linear(layer.mlp.fc2)

        module_util.replace_modules(
            model.model.vision_model.encoder.layers,
            torch.nn.LayerNorm,
            lambda _: module_util.RMSN(
                model.model.vision_model.encoder.config.hidden_size, eps=1e-6
            ),
            replace_layers=False,
        )

    if not args.no_fuse_visual_cross_attn:
        fuse_ln_linear(model.model.mlp1[0], [model.model.mlp1[1]])
        module_util.replace_modules(
            model.model.mlp1,
            torch.nn.LayerNorm,
            lambda _: module_util.RMSN(
                model.model.vision_model.encoder.config.hidden_size
                * int(1 / model.model.config.downsample_ratio) ** 2,
                eps=1e-6,
            ),
            replace_layers=False,
        )

    if not args.no_fuse_llm:
        # fuse internvl2-8b
        for layer in model.model.language_model.model.layers:
            fuse_ln_linear(layer.attention_norm, [layer.attention.wqkv])
            fuse_ln_linear(
                layer.ffn_norm, [layer.feed_forward.w1, layer.feed_forward.w3]
            )

        # 最后一个rmsnorm需要和ln_head合并
        fuse_ln_linear(
            model.model.language_model.model.norm, [model.model.language_model.output]
        )


@torch.inference_mode()
def rotate_internvl2_model(model, args):
    print("rotate model")
    if args.rotate_visual_clip:
        # rotate visual transformer
        num_heads = model.config.vision_config.num_attention_heads
        head_dim = model.config.vision_config.hidden_size // num_heads
        Q_v = get_orthogonal_matrix(
            model.config.vision_config.hidden_size, args.rotate_mode
        )

        rotate_conv(
            model.vision_model.embeddings.patch_embedding,
            Q_v,
            model.config.vision_config.hidden_size,
        )
        dtype = model.vision_model.embeddings.position_embedding.data.dtype
        model.vision_model.embeddings.position_embedding.data = torch.matmul(
            model.vision_model.embeddings.position_embedding.data.double(), Q_v
        ).to(dtype)

        model.vision_model.embeddings.class_embedding.data = torch.matmul(
            model.vision_model.embeddings.class_embedding.data.double(), Q_v
        ).to(dtype)

        for idx, layer in enumerate(
            tqdm.tqdm(
                model.vision_model.encoder.layers,
                unit="layer",
                desc="Rotating Visual CLIP",
            )
        ):
            rotate_internvl_attention_inputs(layer, Q_v, is_visual=True)
            rotate_internvl_attention_output(layer, Q_v, is_visual=True)
            rotate_internvl_mlp_input(layer, Q_v, is_visual=True)
            rotate_internvl_mlp_output(
                layer,
                Q_v,
                True,
                args.online_visual_hadamard,
            )

            rotate_internvl_ov_proj(
                layer,
                num_heads,
                head_dim,
                is_visual=True,
            )

        rotate_mlp1(model, Q_v)
        utils.cleanup_memory()

    if args.rotate_visual_cross_attn:
        print("\n Rotating Visual Cross Attention \n")
        pass

    if args.rotate_llm:
        Q = get_orthogonal_matrix(model.config.llm_config.hidden_size, args.rotate_mode)

        config = model.config.llm_config
        num_attention_heads = config.num_attention_heads
        num_key_value_head = config.num_key_value_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_attention_heads

        rotate_internvl_embeddings(model, Q)
        rotate_internvl_head(model, Q)
        utils.cleanup_memory()
        for idx, layer in enumerate(
            tqdm.tqdm(
                model.language_model.model.layers, unit="layer", desc="LLM Rotating"
            )
        ):
            rotate_internvl_attention_inputs(layer, Q)
            rotate_internvl_attention_output(layer, Q)
            rotate_internvl_mlp_input(layer, Q)
            rotate_internvl_mlp_output(layer, Q, False, args.online_llm_hadamard)
            rotate_internvl_ov_proj_v2(
                layer, num_attention_heads, num_key_value_head, head_dim
            )
        utils.cleanup_memory()