import torch
from fake_quant.rotation_utils import (
    fuse_ln_linear,
    bake_mean_into_linear,
    bake_mean_into_conv,
    rotate_conv,
)
from fake_quant import module_util
from fake_quant import utils
import tqdm
from fake_quant.rotation_utils import (
    get_orthogonal_matrix,
    rotate_attention_inputs,
    rotate_attention_output,
    rotate_mlp_input,
    rotate_mlp_output,
    rotate_ov_proj,
    rotate_kv_proj,
    rotate_cross_embeddings,
    rotate_cross_attention_inputs,
    rotate_o_ln_proj_fc,
    rotate_embeddings,
    rotate_head,
)

def fuse_minicpmv_layer_norms(model, args):
    print("fuse minicpmv layer norms")
    if not args.no_fuse_visual_clip:
        bake_mean_into_conv(model.vpm.embeddings.patch_embedding)
        dtype = model.vpm.embeddings.position_embedding.weight.data
        model.vpm.embeddings.position_embedding.weight.data = (
            model.vpm.embeddings.position_embedding.weight.data
            - model.vpm.embeddings.position_embedding.weight.data.double().mean(
                dim=-1, keepdim=True
            )
        ).to(dtype)
        for layer in model.vpm.encoder.layers:
            fuse_ln_linear(
                layer.layer_norm1,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])

            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.mlp.fc2)

        fuse_ln_linear(model.vpm.post_layernorm, [model.resampler.kv_proj])
        module_util.replace_modules(
            model.vpm.encoder.layers,
            torch.nn.LayerNorm,
            lambda _: module_util.RMSN(model.vpm.embed_dim, eps=1e-6),
            replace_layers=False,
        )

    if not args.no_fuse_visual_cross_attn:
        # fuse ln_kv
        dtype = model.resampler.pos_embed.data.dtype
        model.resampler.pos_embed.data = (
            model.resampler.pos_embed.data.double()
            / model.resampler.ln_kv.weight.data.double()
        ).to(dtype)
        fuse_ln_linear(
            model.resampler.ln_kv,
            [
                model.resampler.attn.k_proj,
                model.resampler.attn.v_proj,
            ],
        )

        # fuse ln_q
        fuse_ln_linear(
            model.resampler.ln_q,
            [model.resampler.attn.q_proj],
        )

        # fuse ln_post
        fuse_ln_linear(model.resampler.ln_post, [model.resampler.proj_fc])

        # convert ln_q to rmsnorm, ln_kv to rmsnorm, ln_post to rmsnorm
        model.resampler.query.data = (
            model.resampler.query.data
            - model.resampler.query.data.double().mean(dim=-1, keepdim=True)
        ).to(dtype)
        bake_mean_into_linear(model.resampler.kv_proj)
        bake_mean_into_linear(model.resampler.attn.out_proj)
        module_util.replace_modules(
            model.resampler,
            torch.nn.LayerNorm,
            lambda _: module_util.RMSN(model.resampler.embed_dim, eps=1e-6),
            replace_layers=False,
        )

    if not args.no_fuse_llm:
        # fuse qwen2 7b
        for layer in model.llm.model.layers:
            fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
            )
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )

        # 最后一个rmsnorm需要和ln_head合并
        fuse_ln_linear(model.llm.model.norm, [model.llm.lm_head])


@torch.no_grad()
def rotate_minicpmv_model(model, args):
    print("rotate model")
    if args.rotate_visual_clip:
        # rotate visual transformer
        embed_dim = model.vpm.embeddings.embed_dim
        Q_v = get_orthogonal_matrix(embed_dim, args.rotate_mode)

        rotate_conv(model.vpm.embeddings.patch_embedding, Q_v, embed_dim)
        dtype = model.vpm.embeddings.position_embedding.weight.data.dtype
        model.vpm.embeddings.position_embedding.weight.data = torch.matmul(
            model.vpm.embeddings.position_embedding.weight.data.double(), Q_v
        ).to(dtype)

        if args.online_visual_hadamard:
            model.config.vision_config.need_pad = False
            from fake_quant.hadamard_utils import auto_pad_size

            new_intermediate_size = auto_pad_size(
                model.config.vision_config.intermediate_size
            )
            if new_intermediate_size != model.config.vision_config.intermediate_size:
                for name, module in model.named_modules():
                    if "mlp.fc2" in name and isinstance(module, torch.nn.Linear):
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
                model.config.vision_config.intermediate_size = new_intermediate_size
                model.config.vision_config.need_pad = True

        for idx, layer in enumerate(
            tqdm.tqdm(
                model.vpm.encoder.layers,
                unit="layer",
                desc="Rotating Visual CLIP",
            )
        ):
            rotate_attention_inputs(layer, Q_v, is_minicpmv=True)
            rotate_attention_output(layer, Q_v, is_visual=True)
            rotate_mlp_input(layer, Q_v, is_visual=True)
            rotate_mlp_output(layer, Q_v, args.online_visual_hadamard)

            rotate_ov_proj(
                layer,
                layer.self_attn.num_heads,
                layer.self_attn.head_dim,
                is_visual=True,
            )
        rotate_kv_proj(model, Q_v, is_minicpmv=True)
        utils.cleanup_memory()

    if args.rotate_visual_cross_attn:
        print("Rotating Visual Cross Attention")
        embed_dim = model.resampler.embed_dim
        num_heads = model.resampler.num_heads
        # rotate visual cross attention
        Q_q = get_orthogonal_matrix(embed_dim, args.rotate_mode)
        Q_kv = get_orthogonal_matrix(embed_dim, args.rotate_mode)
        rotate_cross_embeddings(model, Q_q, Q_kv, is_minicpmv=True)
        rotate_cross_attention_inputs(model.resampler, Q_q, Q_kv)

        rotate_ov_proj(
            model.resampler,
            num_heads,
            embed_dim // num_heads,
            is_visual=True,
        )

        Q_o = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
        rotate_o_ln_proj_fc(model.resampler, Q_o, is_minicpmv=True)
        utils.cleanup_memory()

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

        # rotate qwen2 7b
        Q = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)

        config = model.config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads

        rotate_embeddings(model, Q, is_minicpmv=True)
        for idx, layer in enumerate(
            tqdm.tqdm(model.llm.model.layers, unit="layer", desc="Rotating")
        ):
            rotate_attention_inputs(layer, Q, is_minicpmv=True)
            rotate_attention_output(layer, Q)
            rotate_mlp_input(layer, Q, is_minicpmv=True)
            rotate_mlp_output(layer, Q, args.online_llm_hadamard)
            rotate_ov_proj(layer, num_heads, head_dim, is_minicpmv=True)

        rotate_head(model, Q, is_minicpmv=True)
        utils.cleanup_memory()