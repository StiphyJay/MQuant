import math
import time
import tqdm
import torch
import torch.nn as nn
from fake_quant import utils
from fake_quant import quant_utils
import logging
from fake_quant.gptq.gptq_utils import GPTQ, GPTQConv

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def qwenvl_visual_clip_rtn(model, dev, args):
    # visiual conv1
    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(
        args.visual_w_bits,
        perchannel=True,
        sym=not (args.w_asym),
        mse=args.visual_w_clip,
    )
    W = model.transformer.visual.conv1.module.weight.data
    quantizer.find_params(W)
    model.transformer.visual.conv1.module.weight.data = quantizer.quantize(W).to(
        model.transformer.visual.conv1.module.weight.dtype
    )

    # visual transformer resblocks
    layers = model.transformer.visual.transformer.resblocks
    for i in tqdm.tqdm(
        range(len(layers)), desc="(RtN Quant.) visual transformer Layers"
    ):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            if any(p_name in name for p_name in args.skip_names) or "L1" in name:
                continue
            layer_weight_bits = args.visual_w_bits
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8

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
        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_fwrd_visual_clip_conv1(model, dataset, dev, args, quantizers):
    """
    From GPTQ repo
    """
    print("-----GPTQ Quantization visual clip conv1-----")
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

    model.model.transformer.visual.conv1 = Catcher(model.model.transformer.visual.conv1)
    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass

    model.model.transformer.visual.conv1 = model.model.transformer.visual.conv1.module
    layer_weight_bits = args.visual_w_bits
    layer_weight_sym = not (args.w_asym)
    conv1_gptq = GPTQConv(model.model.transformer.visual.conv1.module)
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
        model.model.transformer.visual.conv1.module.register_forward_hook(add_batch())
    )
    for j in range(args.nsamples):
        model.model.transformer.visual.conv1(
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
    quantizers["model.transformer.visual.conv1"] = conv1_gptq.quantizer
    conv1_gptq.free()
    del conv1_gptq
    model.model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization visual clip conv1 Done-----")


@torch.no_grad()
def gptq_fwrd_visual_clip_resblocks(model, dataset, dev, args, quantizers):
    print("-----GPTQ Quantization visual clip resblocks-----")
    use_cache = model.model.config.use_cache
    model.model.config.use_cache = False
    layers = model.model.transformer.visual.transformer.resblocks

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.model.parameters())).dtype

    inps = [None] * args.nsamples
    attention_masks = [None] * args.nsamples
    rotary_pos_embs = [None] * args.nsamples

    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
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
            "attn.k_proj.module",
            "attn.v_proj.module",
            "attn.q_proj.module",
        ],
        ["attn.out_proj.module"],
        ["mlp.c_fc.module"],
    ]
    if args.online_visual_hadamard and args.visual_split:
        sequential.append(["mlp.c_proj.L2"])
    else:
        sequential.append(["mlp.c_proj.module"])
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
                outs[j] = layer(
                    inps[j],
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
                quantizers[
                    "model.transformer.visual.transformer.resblocks.%d.%s" % (i, name)
                ] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j],
            )

        layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    print("-----GPTQ Quantization visual clip resblocks Done----")
    return quantizers


def qwenvl_visual_cross_attention_rtn(model, dev, args):
    subset = quant_utils.find_qlayers(
        model.transformer.visual.attn_pool, layers=[torch.nn.Linear]
    )
    for name in subset:
        layer_weight_bits = args.visual_w_bits
        if "lm_head" in name:
            layer_weight_bits = 16
            continue
        if args.int8_down_proj and "down_proj" in name:
            layer_weight_bits = 8

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

    #  quant proj_fc weight
    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(
        layer_weight_bits,
        perchannel=True,
        sym=not (args.w_asym),
        mse=args.visual_w_clip,
    )
    W = model.transformer.visual.proj_fc.weight.data
    quantizer.find_params(W)
    model.transformer.visual.proj_fc.weight.data = quantizer.quantize(W).to(
        model.transformer.visual.proj_fc.weight.dtype
    )


@torch.no_grad()
def gptq_fwrd_visual_clip_cross_attention(model, dataset, dev, args, quantizers):
    """
    From GPTQ repo
    """
    print("-----GPTQ Quantization visual clip cross attention-----")
    use_cache = model.model.config.use_cache
    model.model.config.use_cache = False
    layer = model.model.transformer.visual.attn_pool

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

    model.model.transformer.visual.attn_pool = Catcher(
        model.model.transformer.visual.attn_pool
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
    model.model.transformer.visual.attn_pool = (
        model.model.transformer.visual.attn_pool.module
    )

    sequential = [
        ["kv_proj.module"],
        [
            "attn.k_proj.module",
            "attn.v_proj.module",
            "attn.q_proj.module",
        ],
        ["attn.out_proj.module"],
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
            layer(
                inps[j],
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
            quantizers["model.model.transformer.visual.attn_pool.%s" % name] = gptq[
                name
            ].quantizer
            gptq[name].free()

    model.model.transformer.visual.attn_pool = layer
    del gptq

    # proj_fc use gptq
    proj_fc_inps = [None] * args.nsamples
    for j in range(args.nsamples):
        proj_fc_inps[j] = model.model.transformer.visual.ln_post(layer(inps[j]))

    layer = model.model.transformer.visual.proj_fc
    gptq_layer = GPTQ(layer)
    gptq_layer.quantizer = quant_utils.WeightQuantizer()
    gptq_layer.quantizer.configure(
        layer_weight_bits,
        perchannel=True,
        sym=layer_weight_sym,
        mse=args.visual_w_clip,
    )

    def add_batch(name):
        def tmp(_, inp, out):
            gptq_layer.add_batch(inp[0].data, out.data)

        return tmp

    handles = []
    handles.append(layer.register_forward_hook(add_batch(name)))
    for j in range(args.nsamples):
        layer(
            proj_fc_inps[j],
        )
    for h in handles:
        h.remove()

    layer_w_groupsize = args.w_groupsize
    gptq_layer.fasterquant(
        percdamp=args.percdamp,
        groupsize=layer_w_groupsize,
        actorder=args.act_order,
        static_groups=False,
    )
    quantizers["model.model.transformer.visual.proj_fc"] = gptq_layer.quantizer
    gptq_layer.free()

    torch.cuda.empty_cache()

    utils.cleanup_memory(verbos=True)
    print("-----GPTQ Quantization visual clip cross attention Done-----")


def qwenvl_llm_rtn(model, dev, args):
    """
    From GPTQ repo
    """
    quantizers = dict()
    layers = model.transformer.h
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
            quantizers["model.transformer.h.%d.%s" % (i, name)] = quantizer.cpu()
        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_fwrd_llm(model, dataset, dev, args, quantizers):
    """
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    """
    print("-----GPTQ Quantization LLM-----\n")

    use_cache = model.model.config.use_cache
    model.model.config.use_cache = False
    layers = model.model.transformer.h

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.model.parameters())).dtype

    inps = [None] * args.nsamples
    attention_masks = [None] * args.nsamples
    rotary_pos_embs = [None] * args.nsamples

    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp

            attention_masks[cache["i"]] = kwargs["attention_mask"]
            rotary_pos_embs[cache["i"]] = kwargs["rotary_pos_emb"]
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
            "attn.k_proj.module",
            "attn.v_proj.module",
            "attn.q_proj.module",
        ],
        ["attn.c_proj.module"],
        ["mlp.w1.module", "mlp.w2.module"],
    ]
    if args.online_llm_hadamard and args.llm_split:
        sequential.append(["mlp.c_proj.L2"])
    else:
        sequential.append(["mlp.c_proj.module"])
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
                    rotary_pos_emb=rotary_pos_embs[j],
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
                quantizers["model.transformer.h.%d.%s" % (i, name)] = gptq[
                    name
                ].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_masks[j],
                rotary_pos_emb=rotary_pos_embs[j],
            )[0]

        layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization LLM Done-----\n")
    return quantizers


@torch.no_grad()
def qwenvl_rtn_gptq_fwrd_plus(model, dataset, dev, args):
    """
    From GPTQ repo
    """
    logging.info("-----RTN Or GPTQ Quantization-----")
    quantizers = dict()
    if args.quant_visual_clip:
        if args.visual_w_rtn:
            qwenvl_visual_clip_rtn(model.model, dev, args)
        else:
            gptq_fwrd_visual_clip_conv1(model, dataset, dev, args, quantizers)
            gptq_fwrd_visual_clip_resblocks(model, dataset, dev, args, quantizers)

    if args.quant_cross_attention:
        if args.visual_w_rtn:
            qwenvl_visual_cross_attention_rtn(model.model, dev, args)
        else:
            gptq_fwrd_visual_clip_cross_attention(model, dataset, dev, args, quantizers)

    if args.quant_llm:
        if args.llm_w_rtn:
            qwenvl_llm_rtn(model.model, dev, args)
        else:
            gptq_fwrd_llm(model, dataset, dev, args, quantizers)
    return quantizers
