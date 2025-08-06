import math
import time
import tqdm
import torch
import torch.nn as nn
from fake_quant import utils
from fake_quant import quant_utils
from fake_quant.gptq.gptq_utils import GPTQ, GPTQConv
import logging

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def qwen2vl_visual_clip_rtn(model, dev, args, quantizers):
    # visiual conv1
    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(
        args.visual_w_bits,
        perchannel=True,
        sym=not (args.w_asym),
        mse=args.visual_w_clip,
    )
    W = model.visual.patch_embed.proj.module.weight.data
    quantizer.find_params(W)
    model.visual.patch_embed.proj.module.weight.data = quantizer.quantize(W).to(
        model.visual.patch_embed.proj.module.weight.dtype
    )
    quantizers["model.visual.patch_embed.proj.module"] = quantizer.cpu()

    # visual transformer resblocks
    layers = model.visual.blocks
    for i in tqdm.tqdm(
        range(len(layers)), desc="(RtN Quant.) visual transformer Layers"
    ):
        layer = layers[i]

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            if any(p_name in name for p_name in args.skip_names) or "L1" in name:
                continue
            layer_weight_bits = args.visual_w_bits
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.visual_w_clip,
            )
            W = subset[name].weight.data
            dtype = W.dtype
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(dtype)
            quantizers["model.visual.blocks.%d.%s" % (i, name)] = quantizer.cpu()
        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_qwen2vl_fwrd_visual_clip_conv1(
    model, dataset, dev, dataset_name, args, quantizers
):
    """
    From GPTQ repo
    """
    # conv1 gptq
    inps = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            raise ValueError

    model.model.visual.patch_embed = Catcher(model.model.visual.patch_embed)

    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break

        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass

    model.model.visual.patch_embed = model.model.visual.patch_embed.module
    layer_weight_bits = args.visual_w_bits
    layer_weight_sym = not (args.w_asym)
    conv1_gptq = GPTQConv(model.model.visual.patch_embed.proj.module)
    conv1_gptq.quantizer = quant_utils.WeightQuantizer()
    conv1_gptq.quantizer.configure(
        layer_weight_bits,
        perchannel=True,
        sym=layer_weight_sym,
        mse=args.visual_w_clip,
    )

    def add_batch():
        def tmp(_, inp, out):
            conv1_gptq.add_batch(inp[0].data, out.data)

        return tmp

    handles = []
    handles.append(
        model.model.visual.patch_embed.proj.module.register_forward_hook(add_batch())
    )
    for j in range(args.nsamples):
        model.model.visual.patch_embed(
            inps[j],
        )
    for h in handles:
        h.remove()
    layer_w_groupsize = args.w_groupsize
    conv1_gptq.fasterquant(
        percdamp=args.percdamp,
        groupsize=layer_w_groupsize,
        actorder=args.act_order,
        static_groups=False,
    )
    quantizers["model.visual.patch_embed.proj.module"] = conv1_gptq.quantizer.cpu()
    conv1_gptq.free()
    del conv1_gptq
    utils.cleanup_memory(verbos=True)
    print("-----GPTQ Quantization visual clip conv1 Done-----")


@torch.no_grad()
def gptq_qwen2vl_fwrd_visual_clip_resblocks(
    model, dataset, dev, dataset_name, args, quantizers
):
    """
    gptq for visual transformer resblocks
    """
    layers = model.model.visual.blocks

    layers[0] = layers[0]
    inps = [None] * args.nsamples
    cu_seqlens = [None] * args.nsamples
    rotary_pos_emb = [None] * args.nsamples

    cache = {"i": 0}

    # 实现猴子补丁
    def monkey_patched_forward(self, *args, **kwargs):
        inps[cache["i"]] = args[0]  # 存储输入
        cu_seqlens[cache["i"]] = kwargs.get("cu_seqlens", None)
        rotary_pos_emb[cache["i"]] = kwargs.get("rotary_pos_emb", None)
        cache["i"] += 1
        raise ValueError("Catcher triggered")

    # 给 SimpleModule 的 forward 方法打上猴子补丁】
    forward = layers[0].forward
    layers[0].forward = monkey_patched_forward.__get__(layers[0], layers[0].__class__)

    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass

    layers[0].forward = forward

    outs = [None] * args.nsamples

    sequential = [
        [
            "attn.qkv.module",
        ],
        ["attn.proj.module"],
        ["mlp.fc1.module"],
    ]
    if args.visual_split:
        sequential.append(["mlp.fc2.L2"])
    else:
        sequential.append(["mlp.fc2.module"])

    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i]
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            if any(p_name in name for p_name in args.skip_names):
                continue
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.visual_w_bits
                layer_weight_sym = not (args.w_asym)
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.visual_w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j], cu_seqlens=cu_seqlens[j], rotary_pos_emb=rotary_pos_emb[j]
                )
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
                quantizers["model.visual.blocks.%d.%s" % (i, name)] = gptq[
                    name
                ].quantizer.cpu()
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j], cu_seqlens=cu_seqlens[j], rotary_pos_emb=rotary_pos_emb[j]
            )

        layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization visual clip resblocks Done----")
    return quantizers


def qwen2vl_visual_cross_attention_rtn(model, dev, args, quantizers):
    print("-----Rtn Quantization visual clip cross attention-----")
    # visiual cross attention
    subset = quant_utils.find_qlayers(model.visual.merger, layers=[torch.nn.Linear])
    for name in subset:
        layer_weight_bits = args.visual_w_bits
        quantizer = quant_utils.WeightQuantizer()
        quantizer.configure(
            layer_weight_bits,
            perchannel=True,
            sym=not (args.w_asym),
            mse=args.visual_w_clip,
        )
        W = subset[name].weight.data
        quantizer.find_params(W)
        subset[name].weight.data = quantizer.quantize(W).to(subset[name].weight.dtype)
        quantizers["model.visual.merger.%s" % name] = quantizer.cpu()


def gptq_qwen2vl_fwrd_visual_clip_cross_attention(
    model, dataset, dev, dataset_name, args, quantizers
):
    """
    From GPTQ repo
    """
    print("-----GPTQ Quantization visual clip cross attention-----")
    layer = model.model.visual.merger
    inps = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            inps[cache["i"]] = args[0]
            cache["i"] += 1
            raise ValueError

    model.model.visual.merger = Catcher(model.model.visual.merger)
    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass
    model.model.visual.merger = model.model.visual.merger.module

    sequential = [
        ["mlp.0.module"],
        ["mlp.2.module"],
    ]

    full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
    for names in sequential:
        subset = {n: full[n] for n in names}

        gptq = {}
        for name in subset:
            print(f"{name}", end="  ", flush=True)
            layer_weight_bits = args.visual_w_bits
            layer_weight_sym = not (args.w_asym)
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = quant_utils.WeightQuantizer()
            gptq[name].quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=layer_weight_sym,
                mse=args.visual_w_clip,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            layer(inps[j])
        for h in handles:
            h.remove()

        for name in subset:
            layer_w_groupsize = args.w_groupsize
            gptq[name].fasterquant(
                percdamp=args.percdamp,
                groupsize=layer_w_groupsize,
                actorder=args.act_order,
                static_groups=False,
            )
            quantizers["model.visual.merger.%s" % name] = gptq[name].quantizer
            gptq[name].free()

    model.model.resampler = layer
    del gptq
    torch.cuda.empty_cache()
    utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization visual clip cross attention Done-----")


def qwen2vl_llm_rtn(model, dev, args, quantizers):
    print("-----Rtn Quantization llm---")
    layers = model.model.layers
    torch.cuda.empty_cache()

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) LLM Layers"):
        layer = layers[i]

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            if any(p_name in name for p_name in args.skip_names) or "L1" in name:
                continue
            layer_weight_bits = args.llm_w_bits
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.llm_w_clip,
            )
            W = subset[name].weight.data
            dtype = W.dtype
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(dtype)
            quantizers["model.model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_qwen2vl_fwrd_llm(model, dataset, dev, dataset_name, args, quantizers):
    """
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    """
    print("-----GPTQ Quantization LLM-----")
    use_cache = model.model.config.use_cache
    model.model.config.use_cache = False
    layers = model.model.model.layers

    layers[0] = layers[0]

    inps = [None] * args.nsamples
    attention_masks = [None] * args.nsamples
    position_ids = [None] * args.nsamples
    attention_masks = [None] * args.nsamples
    cache_position = [None] * args.nsamples

    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp

            attention_masks[cache["i"]] = kwargs["attention_mask"]
            position_ids[cache["i"]] = kwargs["position_ids"]
            cache_position[cache["i"]] = kwargs["cache_position"]
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = [None] * args.nsamples

    sequential = [
        [
            "self_attn.q_proj.module",
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
    ]
    if args.llm_split:
        sequential.append(["mlp.down_proj.L2"])
    else:
        sequential.append(["mlp.down_proj.module"])
    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i]
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            if any(p_name in name for p_name in args.skip_names):
                continue
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.llm_w_bits
                layer_weight_sym = not (args.w_asym)
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.llm_w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j],
                    attention_mask=attention_masks[j],
                    position_ids=position_ids[j],
                    cache_position=cache_position[j],
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
                quantizers["model.model.layers.%d.%s" % (i, name)] = gptq[
                    name
                ].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_masks[j],
                position_ids=position_ids[j],
                cache_position=cache_position[j],
            )[0]

        layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization LLM Done-----")
    return quantizers


@torch.no_grad()
def qwen2vl_rtn_gptq_fwrd_plus(model, dataset, dev, dataset_name, args):
    """
    From GPTQ repo
    """
    logging.info("-----RTN Or GPTQ Quantization-----")

    quantizers = dict()

    if args.quant_visual_clip:
        if args.visual_w_rtn:
            qwen2vl_visual_clip_rtn(model.model, dev, args, quantizers)
        else:
            gptq_qwen2vl_fwrd_visual_clip_conv1(
                model, dataset, dev, dataset_name, args, quantizers
            )
            gptq_qwen2vl_fwrd_visual_clip_resblocks(
                model, dataset, dev, dataset_name, args, quantizers
            )

    if args.quant_cross_attention:
        if args.visual_w_rtn:
            qwen2vl_visual_cross_attention_rtn(model.model, dev, args, quantizers)
        else:
            gptq_qwen2vl_fwrd_visual_clip_cross_attention(
                model, dataset, dev, dataset_name, args, quantizers
            )

    if args.quant_llm:
        if args.llm_w_rtn:
            qwen2vl_llm_rtn(model.model, dev, args, quantizers)
        else:
            gptq_qwen2vl_fwrd_llm(model, dataset, dev, dataset_name, args, quantizers)
    return quantizers
