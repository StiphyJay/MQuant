import torch
from PIL import Image
import argparse
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def main(args):
    torch.manual_seed(0)

    model = AutoModel.from_pretrained(
        "weights/MiniCPM-V-2_6",
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )  # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()

    # convert proj to proj_fc
    proj_fc_weight = model.resampler.proj.data.clone().T.contiguous()
    model.resampler.proj_fc = nn.Linear(
        proj_fc_weight.size(1), proj_fc_weight.size(0), bias=True
    )
    model.resampler.proj_fc.weight.data = proj_fc_weight
    model.resampler.proj_fc.bias.data = torch.zeros(proj_fc_weight.size(0)).to(
        proj_fc_weight
    )
    del model.resampler.proj

    # convert nn.MultiheadAttention to MultiheadAttention
    embed_dim = model.resampler.attn.embed_dim
    model.resampler.attn.q_proj = nn.Linear(embed_dim, embed_dim)
    model.resampler.attn.k_proj = nn.Linear(embed_dim, embed_dim)
    model.resampler.attn.v_proj = nn.Linear(embed_dim, embed_dim)

    w_q, w_k, w_v = model.resampler.attn.in_proj_weight.data.chunk(3)
    b_q, b_k, b_v = model.resampler.attn.in_proj_bias.data.chunk(3)

    model.resampler.attn.q_proj.weight.data = w_q.clone()
    model.resampler.attn.k_proj.weight.data = w_k.clone()
    model.resampler.attn.v_proj.weight.data = w_v.clone()

    model.resampler.attn.q_proj.bias.data = b_q.clone()
    model.resampler.attn.k_proj.bias.data = b_k.clone()
    model.resampler.attn.v_proj.bias.data = b_v.clone()

    del model.resampler.attn.in_proj_weight
    del model.resampler.attn.in_proj_bias

    model.save_pretrained("weights/MiniCPM-V-2_6-opt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--act_per_tensor",
    #     action="store_true",
    #     default=False,
    #     help="Quantize the activations per tensor",
    # )
    # parser.add_argument(
    #     "--draw_save_path",
    #     type=str,
    #     default="output/minicpm_base",
    #     help="""analysis act save path. """,
    # )
    args = parser.parse_args()
    main(args)
