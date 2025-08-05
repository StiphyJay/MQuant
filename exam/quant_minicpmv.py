import torch
from PIL import Image
import argparse
import datetime
import functools
import numpy as np
from exam.minicpmv import MiniCPM_V
from fake_quant import utils
from fake_quant import quant_utils, gptq
from fake_quant import hadamard_utils
from fake_quant.minicpmv_rotation import fuse_minicpmv_layer_norms, rotate_minicpmv_model
from vlmeval.vlm.minicpm_v import QMiniCPM_V_2_6
from evaluation.eval import eval_dataset


def main(args):
    utils.seed_everything(args.seed)
    model = MiniCPM_V(model_path="weights/MiniCPM-V-2_6-opt", ckpt=None, device="cuda")

    if not args.not_fuse_layer_norms:
        fuse_minicpmv_layer_norms(model.model, args)

    if args.rotate:
        rotate_minicpmv_model(model.model, args)

    model = QMiniCPM_V_2_6(model.model, model.tokenizer)
    if not args.quant and args.online_llm_hadamard:
        if args.rotate_llm:
            args.quant_llm = True
        quant_utils.minicpmv_add_act_qaunt(model.model, args)
        qlayers = quant_utils.find_qlayers(
            model.model, layers=[quant_utils.ActQuantWrapper]
        )
        for name in qlayers:
            if "mlp.down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
                if model.model.config.need_pad:
                    hook = functools.partial(
                        utils.revise_down_input,
                        new_size=model.model.config.intermediate_size,
                    )
                    qlayers[name].register_forward_pre_hook(hook)

    if not args.quant and args.online_visual_hadamard:
        if args.rotate_visual_clip:
            args.quant_visual_clip = True
        quant_utils.minicpmv_add_act_qaunt(model.model, args)
        qlayers = quant_utils.find_qlayers(
            model.model.transformer.visual, layers=[quant_utils.ActQuantWrapper]
        )
        for name in qlayers:
            if "mlp.fc2" in name:
                had_K, K = hadamard_utils.get_hadK(
                    int(model.model.config.vision_config.intermediate_size)
                )
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
                if model.model.config.vision_config.need_pad:
                    hook = functools.partial(
                        utils.revise_down_input,
                        new_size=model.model.config.vision_config.intermediate_size,
                    )
                    qlayers[name].register_forward_pre_hook(hook)

    if args.quant:
        if args.online_llm_hadamard:
            if args.rotate_llm:
                args.quant_llm = True
        if args.online_visual_hadamard:
            if args.rotate_visual_clip:
                args.quant_visual_clip = True
        quant_utils.minicpmv_add_act_qaunt(model.model, args)

        if args.online_llm_hadamard and args.rotate_llm:
            print("adding online hadamard rotation")
            qlayers = quant_utils.find_qlayers(
                model.model, layers=[quant_utils.ActQuantWrapper]
            )
            for name in qlayers:
                if "mlp.down_proj" in name:
                    had_K, K = hadamard_utils.get_hadK(
                        model.model.config.intermediate_size
                    )
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                    qlayers[name].split = args.llm_split
                    if args.llm_split:
                        qlayers[name].split_weights()
                    if model.model.config.need_pad:
                        hook = functools.partial(
                            utils.revise_down_input,
                            new_size=model.model.config.intermediate_size,
                        )
                        qlayers[name].register_forward_pre_hook(hook)

        if args.online_visual_hadamard and args.rotate_visual_clip:
            print("adding online hadamard rotation")
            qlayers = quant_utils.find_qlayers(
                model.model, layers=[quant_utils.ActQuantWrapper]
            )
            for name in qlayers:
                if "mlp.fc2" in name:
                    had_K, K = hadamard_utils.get_hadK(
                        int(model.model.config.vision_config.intermediate_size)
                    )
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                    qlayers[name].split = args.visual_split
                    if args.visual_split:
                        qlayers[name].split_weights()
                    if model.model.config.vision_config.need_pad:
                        hook = functools.partial(
                            utils.revise_down_input,
                            new_size=model.model.config.vision_config.intermediate_size,
                        )
                        qlayers[name].register_forward_pre_hook(hook)

        if args.load_gptq:
            print("Loading GPTQ model from: ", args.load_gptq)
            model.model = torch.load(args.load_gptq)
        else:
            # from torch.utils.data import ConcatDataset
            from vlmeval.dataset import build_dataset

            dataset = build_dataset(args.dataset_name)
            model.set_dump_image(dataset.dump_image)

            gptq.minicpmv_rtn_gptq_fwrd_plus(
                model, dataset, utils.DEV, args.dataset_name, args
            )
            if args.dump_gptq:
                torch.save(model.model, args.dump_gptq)
                print("Dumped the GPTQ model to: ", args.dump_gptq)

        if args.visual_a_bits < 16 or args.visual_static:
            if args.visual_static and args.visual_a_bits >= 16:
                print("if you want to run act with fp16, please set --static False")
            qlayers = quant_utils.find_qlayers(
                model.model.vpm, layers=[quant_utils.ActQuantWrapper]
            )
            qlayers.update(
                quant_utils.find_qlayers(
                    model.model.resampler, layers=[quant_utils.ActQuantWrapper]
                )
            )
            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue
                layer_input_bits = args.visual_a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not (args.a_asym)
                layer_a_clip = args.a_clip_ratio

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.visual_static,
                    observer_type="minmax",
                )

        if args.llm_a_bits < 16 or args.llm_static:
            if args.llm_static and args.llm_a_bits >= 16:
                print("if you want to run act with fp16, please set --static False")
            qlayers = quant_utils.find_qlayers(
                model.model, layers=[quant_utils.ActQuantWrapper]
            )
            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue
                layer_input_bits = args.llm_a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not (args.a_asym)
                layer_a_clip = args.a_clip_ratio

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.llm_static,
                    observer_type="minmax",
                )
    model.model.to(utils.DEV)

    from vlmeval.dataset import build_dataset

    dataset = build_dataset(args.dataset_name)
    model.set_dump_image(dataset.dump_image)
    if args.llm_static or args.visual_static:
        quant_utils.calib_vqa_plus(model, args, dataset, args.calib_num)

    # Evaluate the model on textVQA and docVQA datasets
    eval_dataset(
        model,
        dataset,
        args.dataset_name,
        model_name="minicpm-v2.6",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_fuse_visual_clip",
        action="store_true",
        default=False,
        help="Quantize the QWen7B llm model",
    )

    parser.add_argument(
        "--no_fuse_visual_cross_attn",
        action="store_true",
        default=False,
        help="Quantize the visual features model",
    )

    parser.add_argument(
        "--no_fuse_llm",
        action="store_true",
        default=False,
        help="Quantize the cross attention model",
    )
    parser.add_argument(
        "--not_fuse_layer_norms",
        action="store_true",
        default=False,
        help="Quantize the cross attention model",
    )
    # Rotation Arguments
    parser.add_argument(
        "--rotate", action="store_true", default=False, help="""Rotate the moodel. """
    )
    parser.add_argument(
        "--rotate_visual_clip",
        action="store_true",
        default=False,
        help="""Rotate the moodel. """,
    )
    parser.add_argument(
        "--rotate_visual_cross_attn",
        action="store_true",
        default=False,
        help="""Rotate the moodel. """,
    )
    parser.add_argument(
        "--rotate_llm",
        action="store_true",
        default=False,
        help="""Rotate the moodel. """,
    )
    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument(
        "--act_per_tensor",
        action="store_true",
        default=False,
        help="Quantize the activations per tensor",
    )
    parser.add_argument(
        "--draw_save_path",
        type=str,
        default="output/minicpm_base",
        help="""analysis act save path. """,
    )

    parser.add_argument(
        "--batchsize", type=int, default=1, help="Batch size for processing."
    )

    # eval
    parser.add_argument(
        "--eval_textVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on textVQA.",
    )
    parser.add_argument(
        "--eval_docVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on docVQA.",
    )

    # textVQA
    parser.add_argument("--textVQA_image_dir", type=str, default="")
    parser.add_argument("--textVQA_ann_path", type=str, default="")

    # docVQA
    parser.add_argument("--docVQA_image_dir", type=str, default="")
    parser.add_argument("--docVQA_ann_path", type=str, default="")

    parser.add_argument("--max_sample_num", type=int, default=None)

    parser.add_argument(
        "--generate_method",
        type=str,
        default="",
        help="generate with interleave or not.",
    )
    # result path
    parser.add_argument("--answer_path", type=str, default="./output/answers-new")
    parser.add_argument("--model_name", type=str, default="minicpm")
    parser.add_argument("--quant", action="store_true")
    parser.add_argument(
        "--quant_llm",
        action="store_true",
        default=False,
        help="Quantize the QWen7B llm model",
    )

    parser.add_argument(
        "--quant_visual_clip",
        action="store_true",
        default=False,
        help="Quantize the visual features model",
    )

    parser.add_argument(
        "--quant_cross_attention",
        action="store_true",
        default=False,
        help="Quantize the cross attention model",
    )
    parser.add_argument(
        "--visual_a_bits",
        type=int,
        default=16,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--llm_a_bits",
        type=int,
        default=16,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action="store_true",
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--visual_w_bits",
        type=int,
        default=16,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--llm_w_bits",
        type=int,
        default=16,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--w_asym",
        action="store_true",
        default=False,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--visual_w_rtn",
        action="store_true",
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--llm_w_rtn",
        action="store_true",
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--visual_w_clip",
        action="store_true",
        default=False,
        help="""Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--llm_w_clip",
        action="store_true",
        default=False,
        help="""Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--visual_static",
        action="store_true",
        default=False,
        help="quant act with static scale and zero point",
    )
    parser.add_argument(
        "--llm_static",
        action="store_true",
        default=False,
        help="quant act with static scale and zero point",
    )
    parser.add_argument(
        "--act_order", action="store_true", default=False, help="act-order in GPTQ"
    )
    parser.add_argument(
        "--calib_num",
        type=int,
        default=32,
        help="calibration number",
    )

    parser.add_argument(
        "--eval_num",
        type=int,
        default=32,
        help="evaluation number",
    )

    parser.add_argument(
        "--calib_mode",
        type=str,
        default="v2",
        help="calibration mode, v1 or v2",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration data samples for GPTQ.",
    )
    parser.add_argument(
        "--skip_names",
        nargs="+",
        default=[],
        help="Skip the quantization of the layers with these names",
    )
    parser.add_argument(
        "--dump_gptq",
        type=str,
        default=None,
        help="Dump the GPTQ model to this path",
    )
    parser.add_argument(
        "--load_gptq",
        type=str,
        default=None,
        help="Load the GPTQ model from this path",
    )

    parser.add_argument(
        "--online_visual_hadamard",
        action="store_true",
        default=False,
        help="Online Hadamard rotation",
    )

    parser.add_argument(
        "--online_llm_hadamard",
        action="store_true",
        default=False,
        help="Online Hadamard rotation",
    )
    parser.add_argument(
        "--fp32_had",
        action="store_true",
        default=False,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="OCRBench",
        help="dataset name",
    )
    parser.add_argument(
        "--visual_split",
        action="store_true",
        default=False,
        help="visual split",
    )
    parser.add_argument(
        "--llm_split",
        action="store_true",
        default=False,
        help="Online Hadamard rotation",
    )
    args = parser.parse_args()
    main(args)
