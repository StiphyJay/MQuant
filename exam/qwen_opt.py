import torch, torch.nn as nn, torch.nn.functional as F, argparse, datetime, os
from model.modeling_qwen import QWenLMHeadModel
from model.tokenization_qwen import QWenTokenizer
from loguru import logger

torch.set_grad_enabled(False)


def get_qwen_and_opt_model(model_id="weights/Qwen-VL-Chat", fp16=True):
    tokenizer = QWenTokenizer.from_pretrained(
        model_id,
    )
    model = QWenLMHeadModel.from_pretrained(
        model_id, config=f"{model_id}/config.json", device_map="cuda", fp16=fp16
    ).eval()

    from model.visual import get_abs_pos

    tgz_size = (
        model.transformer.visual.grid_size[0] * model.transformer.visual.grid_size[1]
    )
    pos_embed_kv_data = get_abs_pos(
        model.transformer.visual.attn_pool.pos_embed.data.clone(), tgz_size
    )

    model.transformer.visual.attn_pool.pos_embed_kv = nn.Parameter(pos_embed_kv_data)

    model.transformer.visual.positional_embedding.data = get_abs_pos(
        model.transformer.visual.positional_embedding.data.clone(), tgz_size
    )

    embed_dim = model.transformer.visual.attn_pool.attn.embed_dim
    model.transformer.visual.attn_pool.attn.q_proj = nn.Linear(embed_dim, embed_dim)
    model.transformer.visual.attn_pool.attn.k_proj = nn.Linear(embed_dim, embed_dim)
    model.transformer.visual.attn_pool.attn.v_proj = nn.Linear(embed_dim, embed_dim)

    w_q, w_k, w_v = model.transformer.visual.attn_pool.attn.in_proj_weight.data.chunk(3)
    b_q, b_k, b_v = model.transformer.visual.attn_pool.attn.in_proj_bias.data.chunk(3)

    model.transformer.visual.attn_pool.attn.q_proj.weight.data = w_q.clone()
    model.transformer.visual.attn_pool.attn.k_proj.weight.data = w_k.clone()
    model.transformer.visual.attn_pool.attn.v_proj.weight.data = w_v.clone()

    model.transformer.visual.attn_pool.attn.q_proj.bias.data = b_q.clone()
    model.transformer.visual.attn_pool.attn.k_proj.bias.data = b_k.clone()
    model.transformer.visual.attn_pool.attn.v_proj.bias.data = b_v.clone()

    del model.transformer.visual.attn_pool.attn.in_proj_weight
    del model.transformer.visual.attn_pool.attn.in_proj_bias

    proj_fc_weight = model.transformer.visual.proj.data.clone().T.contiguous()
    model.transformer.visual.proj_fc = nn.Linear(
        proj_fc_weight.size(1), proj_fc_weight.size(0), bias=True
    )
    model.transformer.visual.proj_fc.weight.data = proj_fc_weight
    model.transformer.visual.proj_fc.bias.data = torch.zeros(proj_fc_weight.size(0)).to(
        proj_fc_weight
    )
    del model.transformer.visual.proj

    model.transformer.visual.fc_sub_mean = nn.Linear(
        model.transformer.visual.conv1.out_channels,
        model.transformer.visual.conv1.out_channels,
        bias=False,
    )
    sub_mean_w = (
        torch.diag(torch.ones(model.transformer.visual.conv1.out_channels)).to(
            proj_fc_weight
        )
        - torch.ones(
            model.transformer.visual.conv1.out_channels,
            model.transformer.visual.conv1.out_channels,
        ).to(proj_fc_weight)
        / model.transformer.visual.conv1.out_channels
    )
    model.transformer.visual.fc_sub_mean.weight.data = sub_mean_w

    for block in model.transformer.h:
        block.attn.q_proj = nn.Linear(
            block.attn.hidden_size, block.attn.projection_size
        )
        block.attn.k_proj = nn.Linear(
            block.attn.hidden_size, block.attn.projection_size
        )
        block.attn.v_proj = nn.Linear(
            block.attn.hidden_size, block.attn.projection_size
        )

        block.attn.q_proj.weight.data = block.attn.c_attn.weight.data[
            : block.attn.projection_size, :
        ].clone()
        block.attn.q_proj.bias.data = block.attn.c_attn.bias.data[
            : block.attn.projection_size
        ].clone()
        block.attn.k_proj.weight.data = block.attn.c_attn.weight.data[
            block.attn.projection_size : 2 * block.attn.projection_size, :
        ].clone()
        block.attn.k_proj.bias.data = block.attn.c_attn.bias.data[
            block.attn.projection_size : 2 * block.attn.projection_size
        ].clone()
        block.attn.v_proj.weight.data = block.attn.c_attn.weight.data[
            2 * block.attn.projection_size :, :
        ].clone()
        block.attn.v_proj.bias.data = block.attn.c_attn.bias.data[
            2 * block.attn.projection_size :
        ].clone()

        del block.attn.c_attn

    for resblock in model.transformer.visual.transformer.resblocks:
        weight = resblock.attn.in_proj.weight.data.clone().T
        bias = resblock.attn.in_proj.bias.data.clone()

        weight = weight.view(resblock.attn.embed_dim, resblock.attn.num_heads, 3, -1)
        bias = bias.view(resblock.attn.num_heads, 3, -1)

        resblock.attn.q_proj = nn.Linear(
            resblock.attn.embed_dim, resblock.attn.embed_dim
        )
        resblock.attn.k_proj = nn.Linear(
            resblock.attn.embed_dim, resblock.attn.embed_dim
        )
        resblock.attn.v_proj = nn.Linear(
            resblock.attn.embed_dim, resblock.attn.embed_dim
        )

        resblock.attn.q_proj.weight.data = (
            weight[:, :, 0]
            .reshape(resblock.attn.embed_dim, resblock.attn.embed_dim)
            .T.clone()
            .contiguous()
        )
        resblock.attn.q_proj.bias.data = bias[:, 0].reshape(-1).clone()

        resblock.attn.k_proj.weight.data = (
            weight[:, :, 1]
            .reshape(resblock.attn.embed_dim, resblock.attn.embed_dim)
            .T.clone()
            .contiguous()
        )
        resblock.attn.k_proj.bias.data = bias[:, 1].reshape(-1).clone()

        resblock.attn.v_proj.weight.data = (
            weight[:, :, 2]
            .reshape(resblock.attn.embed_dim, resblock.attn.embed_dim)
            .T.clone()
            .contiguous()
        )
        resblock.attn.v_proj.bias.data = bias[:, 2].reshape(-1).clone()

        del resblock.attn.in_proj

    model.save_pretrained("weights/Qwen-VL-Chat-opt")


def main(args):
    get_qwen_and_opt_model(args.model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="weights/Qwen-VL-Chat")

    args = parser.parse_args()

    main(args)
