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


def minicpmv_visual_clip_rtn(model, dev, args):
    quantizers = dict()
    # visiual conv1
    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(
        args.visual_w_bits,
        perchannel=True,
        sym=not (args.w_asym),
        mse=args.visual_w_clip,
    )
    W = model.vpm.embeddings.patch_embedding.module.weight.data
    quantizer.find_params(W)
    model.vpm.embeddings.patch_embedding.module.weight.data = quantizer.quantize(W).to(
        model.vpm.embeddings.patch_embedding.module.weight.dtype
    )

    # visual transformer resblocks
    layers = model.vpm.encoder.layers
    for i in tqdm.tqdm(
        range(len(layers)), desc="(RtN Quant.) visual transformer Layers"
    ):
        layer = layers[i].to(dev)

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
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype
            )
            quantizers["model.vpm.encoder.layers.%d.%s" % (i, name)] = quantizer.cpu()
        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_minicpmv_fwrd_visual_clip_conv1(
    model, dataset, dev, dataset_name, args, quantizers
):
    """
    From GPTQ repo
    """
    use_cache = model.model.config.use_cache
    model.model.config.use_cache = False
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

    model.model.vpm.embeddings.patch_embedding = Catcher(
        model.model.vpm.embeddings.patch_embedding
    )

    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass

    model.model.vpm.embeddings.patch_embedding = (
        model.model.vpm.embeddings.patch_embedding.module
    )
    layer_weight_bits = args.visual_w_bits
    layer_weight_sym = not (args.w_asym)
    conv1_gptq = GPTQConv(model.model.vpm.embeddings.patch_embedding.module)
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
        model.model.vpm.embeddings.patch_embedding.module.register_forward_hook(
            add_batch()
        )
    )
    for j in range(args.nsamples):
        model.model.vpm.embeddings.patch_embedding(
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
    quantizers["model.vpm.embeddings.patch_embedding"] = conv1_gptq.quantizer
    conv1_gptq.free()
    del conv1_gptq
    model.model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    print("-----GPTQ Quantization visual clip conv1 Done-----")


@torch.no_grad()
def gptq_minicpmv_fwrd_visual_clip_resblocks(
    model, dataset, dev, dataset_name, args, quantizers
):
    """
    gptq for visual transformer resblocks
    """
    use_cache = model.model.config.use_cache
    model.model.config.use_cache = False
    layers = model.model.vpm.encoder.layers

    layers[0] = layers[0].to(dev)
    inps = [None] * args.nsamples
    attention_masks = [None] * args.nsamples

    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            inps[cache["i"]] = args[0]
            attention_masks[cache["i"]] = args[1]
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = [None] * args.nsamples

    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.out_proj.module"],
        ["mlp.fc1.module"],
    ]

    if args.visual_split:
        sequential.append(["mlp.fc2.L2"])
    else:
        sequential.append(["mlp.fc2.module"])

    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
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
                outs[j] = layer(inps[j], attention_masks[j])
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
                quantizers["model.vpm.encoder.layers.%d.%s" % (i, name)] = gptq[
                    name
                ].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j], attention_masks[j])[0]

        layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization visual clip resblocks Done----")
    return quantizers


def minicpmv_visual_cross_attention_rtn(model, dev, args):
    print("-----Rtn Quantization visual clip cross attention-----")
    # visiual cross attention
    subset = quant_utils.find_qlayers(model.resampler, layers=[torch.nn.Linear])
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


def gptq_minicpmv_fwrd_visual_clip_cross_attention(
    model, dataset, dev, dataset_name, args, quantizers
):
    """
    From GPTQ repo
    """
    print("-----GPTQ Quantization visual clip cross attention-----")
    layer = model.model.resampler
    inps = [None] * args.nsamples
    tgt_sizes = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            inps[cache["i"]] = args[0]
            tgt_sizes[cache["i"]] = args[1]
            cache["i"] += 1
            raise ValueError

    model.model.resampler = Catcher(model.model.resampler)
    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass
    model.model.resampler = model.model.resampler.module

    sequential = [
        ["kv_proj.module"],
        [
            "attn.k_proj.module",
            "attn.v_proj.module",
            "attn.q_proj.module",
        ],
        ["attn.out_proj.module"],
        ["proj_fc.module"],
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
            layer(inps[j], tgt_sizes[j])
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
            quantizers["model.resampler.%s" % name] = gptq[name].quantizer
            gptq[name].free()

    model.model.resampler = layer
    del gptq
    torch.cuda.empty_cache()
    utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization visual clip cross attention Done-----")


def minicpmv_llm_rtn(model, dev, args, quantizers):
    print("-----Rtn Quantization llm---")
    layers = model.llm.model.layers
    torch.cuda.empty_cache()

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) LLM Layers"):
        layer = layers[i].to(dev)

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
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype
            )
            quantizers["model.llm.model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_minicpmv_fwrd_llm(model, dataset, dev, dataset_name, args, quantizers):
    """
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    """
    print("-----GPTQ Quantization LLM-----")

    use_cache = model.model.config.use_cache
    model.model.config.use_cache = False
    layers = model.model.llm.model.layers

    layers[0] = layers[0].to(dev)

    inps = [None] * args.nsamples
    attention_masks = [None] * args.nsamples
    position_ids = [None] * args.nsamples
    attention_masks = [None] * args.nsamples

    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp

            attention_masks[cache["i"]] = kwargs["attention_mask"]
            position_ids[cache["i"]] = kwargs["position_ids"]
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = [None] * args.nsamples

    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
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
        layer = layers[i].to(dev)
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
                quantizers["model.llm.model.layers.%d.%s" % (i, name)] = gptq[
                    name
                ].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_masks[j],
                position_ids=position_ids[j],
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
def minicpmv_rtn_gptq_fwrd_plus(model, dataset, dev, dataset_name, args):
    """
    From GPTQ repo
    """
    logging.info("-----RTN Or GPTQ Quantization-----")

    quantizers = dict()

    if args.quant_visual_clip:
        if args.visual_w_rtn:
            minicpmv_visual_clip_rtn(model.model, dev, args)
        else:
            gptq_minicpmv_fwrd_visual_clip_conv1(
                model, dataset, dev, dataset_name, args, quantizers
            )
            gptq_minicpmv_fwrd_visual_clip_resblocks(
                model, dataset, dev, dataset_name, args, quantizers
            )

    if args.quant_cross_attention:
        if args.visual_w_rtn:
            minicpmv_visual_cross_attention_rtn(model.model, dev, args)
        else:
            gptq_minicpmv_fwrd_visual_clip_cross_attention(
                model, dataset, dev, dataset_name, args, quantizers
            )

    if args.quant_llm:
        if args.llm_w_rtn:
            minicpmv_llm_rtn(model.model, dev, args, quantizers)
        else:
            gptq_minicpmv_fwrd_llm(model, dataset, dev, dataset_name, args, quantizers)
    return quantizers
