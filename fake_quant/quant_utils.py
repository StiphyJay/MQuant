import math
import transformers
import torch
import os
from fake_quant import utils
from fake_quant import hadamard_utils
import fast_hadamard_transform
from collections import OrderedDict
from fake_quant.observer import build_observer
from fake_quant.quantizer import build_quantizer
from fake_quant.bit_type import BIT_TYPE_DICT
from functools import partial
from datasets import load_dataset


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def two_compl(x, bits: int):
    return torch.where(x < 0, 2**bits + x, x)


# Pack the int tensor. Each uint8 stores two int4 value.
def pack_i4(q):
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


# Unpack the quantized int4 tensor (stored in uint8) into int32 tensor.
def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, "The tensor to be unpacked should be stored in uint8"

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0F).to(torch.int8)
    x0[x0 >= 8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xF0) >> 4).to(torch.int8)
    x1[x1 >= 8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)


class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self, act_per_tensor=False):
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = 16
        self.act_per_tensor = act_per_tensor
        self.static = False

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        if self.static:
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
                return x
            elif self.quant:
                return self.quantizer(x)
            else:
                return x
        else:
            x_dtype = x.dtype
            if self.bits == 16:
                return x
            elif self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self,
        bits,
        groupsize=-1,
        sym=False,
        clip_ratio=1.0,
        act_per_tensor=False,
        static=False,
        observer_type="minmax",
        calibration_mode="layer_wise",
    ):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.act_per_tensor = act_per_tensor
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"
        self.static = static
        if self.static:
            module_a_type = "activation"
            bit_type_a = BIT_TYPE_DICT[f"int{bits}"]
            if observer_type == "percentile":
                print("Using percentile observer for activations")
            self.observer = build_observer(
                observer_type,
                module_a_type,
                bit_type_a,
                calibration_mode,
            )
            self.quantizer = build_quantizer(
                "uniform", bit_type_a, self.observer, module_a_type
            )
            self.calibrate = False
            self.last_calibrate = False
            self.quant = False

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
        )

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.act_per_tensor:
            tmp = torch.tensor(0).to(x)
            xmin = torch.minimum(x.min(), tmp) * self.clip_ratio
            xmax = torch.maximum(x.max(), tmp) * self.clip_ratio
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                if xmax == 0:
                    self.scale = 1
                else:
                    self.scale = xmax / self.maxq
                self.zero = torch.zeros_like(self.scale)
            else:
                if xmin == 0:
                    xmin = -1
                if xmax == 0:
                    xmax = 1
                self.scale = (xmax - xmin) / self.maxq
                self.zero = torch.round(-xmin / self.scale)
        else:
            if self.groupsize > 0:
                # group-wise per-token quantization
                self.find_params_per_token_groupwise(x)
                utils.cleanup_memory(verbos=False)
                return
            reshaped_x = x.reshape((-1, x.shape[-1]))

            tmp = torch.zeros(reshaped_x.shape[0], device=dev)
            xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
            xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmax == 0
                self.scale = (
                    (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
                )
                self.scale[tmp] = 1
                self.scale = self.scale.reshape(init_shape)
                self.zero = torch.zeros_like(self.scale)
            else:
                tmp = (xmin == 0) & (xmax == 0)
                xmin[tmp] = -1
                xmax[tmp] = +1
                self.scale = (xmax - xmin) / self.maxq
                self.zero = torch.round(-xmin / self.scale)

                self.scale = (
                    self.scale.unsqueeze(1)
                    .repeat(1, reshaped_x.shape[-1])
                    .reshape(init_shape)
                )
                self.zero = (
                    self.zero.unsqueeze(1)
                    .repeat(1, reshaped_x.shape[-1])
                    .reshape(init_shape)
                )


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registerd to rotate the activation before quantization.
    """

    def __init__(self, module: torch.nn.Linear, act_per_tensor=False):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d))
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer(act_per_tensor)
        self.out_quantizer = ActQuantizer(act_per_tensor)
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.split = False

    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def split_weights(self):
        self.L1 = torch.nn.Linear(1, self.module.out_features, bias=False).to(
            self.module.weight.device
        )
        self.L2 = torch.nn.Linear(
            self.module.in_features - 1,
            self.module.out_features,
            bias=True if self.module.bias is not None else False,
        ).to(self.module.weight.device)
        self.L1.weight.data = self.module.weight.data[:, 0:1]
        self.L2.weight.data = self.module.weight.data[:, 1:]
        if self.module.bias is not None:
            self.L2.bias.data = self.module.bias.data

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:

            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(
                    x.reshape(
                        -1, init_shape[-1] // self.had_dim, self.had_dim
                    ).transpose(1, 2),
                    scale=1 / math.sqrt(init_shape[-1] // self.had_dim),
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.split:
            if self.quantizer.static:
                x[..., 1:] = self.quantizer(x[..., 1:])
            elif self.quantizer.bits < 16:
                self.quantizer.find_params(x[..., 1:])
                x[..., 1:] = self.quantizer(x[..., 1:]).to(x_dtype)
                self.quantizer.free()
            x1 = self.L1.float()(x[..., 0:1].float())
            x2 = self.L2.float()(x[..., 1:].float())
            x = (x1 + x2).to(x_dtype)
        else:
            if self.quantizer.static:
                x = self.quantizer(x)
            elif self.quantizer.bits < 16:
                self.quantizer.find_params(x)
                x = self.quantizer(x).to(x_dtype)
                self.quantizer.free()
            x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class ActRotateWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, QMatrix):
        super(ActRotateWrapper, self).__init__()
        self.module = module
        self.register_buffer("q_matrix", QMatrix)
        self.fp32_had = False

    def forward(self, x, y):
        x_dtype = x.dtype

        if self.fp32_had:  # Full Hadamard in FP32
            x = (x.float() @ self.q_matrix).to(x_dtype)
            y.copy_((y.float() @ self.q_matrix).to(y.dtype))
        else:  # Full Hadamard in FP16
            x = x @ self.q_matrix
            y.copy_(y @ self.q_matrix.to(y.dtype))

        x = self.module(x, y).to(x_dtype)
        return x


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


@torch.no_grad()
def fuse_internvl(model):
    print("fuse internvl vision model...")
    for layer in model.model.vision_model.encoder.layers:
        # layer.ls1  # (out_c)
        # layer.attn.proj.weight  # shape is out_c, in_c
        layer.attn.proj.weight.data *= layer.ls1.data.view(-1, 1)
        layer.mlp.fc2.weight.data *= layer.ls2.data.view(-1, 1)
        if hasattr(layer.attn.proj, "bias"):
            layer.attn.proj.bias.data *= layer.ls1.data
        if hasattr(layer.mlp.fc2, "bias"):
            layer.mlp.fc2.bias.data *= layer.ls2.data
        layer.ls1[:] = 1  # = torch.ones_like(layer.ls1)
        layer.ls2[:] = 1  #  = torch.ones_like(layer.ls2)


def internvl_add_act_qaunt(model, args):
    if args.quant_llm:
        add_actquant(
            model.model.language_model.model,
            args.act_per_tensor,
        )

    if args.quant_visual_clip:
        model.model.vision_model.embeddings.patch_embedding = ActQuantWrapper(
            model.model.vision_model.embeddings.patch_embedding, args.act_per_tensor
        )
        add_actquant(model.model.vision_model.encoder, args.act_per_tensor)

    if args.quant_cross_attention:
        add_actquant_for_mlp1(model.model, args.act_per_tensor)

def qwen2vl_add_act_qaunt(model, args):
    if args.quant_llm:
        add_actquant(
            model.model.model,
            args.act_per_tensor,
        )

    if args.quant_visual_clip:
        model.model.visual.patch_embed.proj = ActQuantWrapper(
            model.model.visual.patch_embed.proj, args.act_per_tensor
        )
        add_actquant(model.model.visual.blocks, args.act_per_tensor)

    if args.quant_cross_attention:
        add_actquant(model.model.visual.merger, args.act_per_tensor)


def qwenvl_add_act_qaunt(model, args):
    if args.quant_llm:
        add_actquant(
            model.transformer.h,
            args.act_per_tensor,
        )

    if args.quant_visual_clip:
        model.transformer.visual.conv1 = ActQuantWrapper(
            model.transformer.visual.conv1, args.act_per_tensor
        )
        add_actquant(model.transformer.visual.transformer, args.act_per_tensor)

    if args.quant_cross_attention:
        add_actquant(model.transformer.visual.attn_pool, args.act_per_tensor)
        model.transformer.visual.proj_fc = ActQuantWrapper(
            model.transformer.visual.proj_fc, args.act_per_tensor
        )  # 目前代码是直接用@实现的 linear，暂时先不量化
        # model.transformer.visual.proj = ActQuantWrapper(model.transformer.visual.proj) # 目前代码是直接用@实现的 linear，暂时先不量化


def minicpmv_add_act_qaunt(model, args):
    if args.quant_llm:
        add_actquant(
            model.llm.model.layers,
            args.act_per_tensor,
        )

    if args.quant_visual_clip:
        model.vpm.embeddings.patch_embedding = ActQuantWrapper(
            model.vpm.embeddings.patch_embedding, args.act_per_tensor
        )
        add_actquant(model.vpm.encoder, args.act_per_tensor)

    if args.quant_cross_attention:
        add_actquant(model.resampler, args.act_per_tensor)


def add_actquant_for_mlp1(
    module,
    act_per_tensor=False,
    name="",
    layers=[
        torch.nn.Linear,
    ],
):
    module.mlp1[1] = ActQuantWrapper(module.mlp1[1], act_per_tensor)
    module.mlp1[3] = ActQuantWrapper(module.mlp1[3], act_per_tensor)


def add_actquant(
    module,
    act_per_tensor=False,
    name="",
    layers=[
        torch.nn.Linear,
    ],
):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp, act_per_tensor))
        if type(tmp) == torch.nn.Sequential:
            replaced = OrderedDict()
            for name, child in tmp.named_children():
                if type(child) in layers:
                    replaced[name] = ActQuantWrapper(child, act_per_tensor)
                else:
                    replaced[name] = child
            setattr(module, attr, torch.nn.Sequential(replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child, act_per_tensor))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(
            child,
            act_per_tensor,
            name + "." + name1 if name != "" else name1,
            [torch.nn.Linear],
        )


def find_qlayers(module, layers=[torch.nn.Linear, ActQuantWrapper], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def model_open_calibrate(model, args):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.calibrate = True
    return model


def model_open_last_calibrate(model, args):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.last_calibrate = True
    return model


def model_close_calibrate(model, args):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.calibrate = False
    return model


def model_quant(model, args):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.quant = True
    return model


def model_no_quant(model, args):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.quant = False
    return model


ds_collections = {
    "vqav2_val": {
        "train": "data/vqav2/vqav2_train.jsonl",
        "test": "data/vqav2/vqav2_val.jsonl",
        "question": "data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json",
        "annotation": "data/vqav2/v2_mscoco_val2014_annotations.json",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    "vqav2_testdev": {
        "train": "data/vqav2/vqav2_train.jsonl",
        "test": "data/vqav2/vqav2_testdev.jsonl",
        "metric": None,
        "max_new_tokens": 10,
    },
    "okvqa_val": {
        "train": "data/okvqa/okvqa_train.jsonl",
        "test": "data/okvqa/okvqa_val.jsonl",
        "question": "data/okvqa/OpenEnded_mscoco_val2014_questions.json",
        "annotation": "data/okvqa/mscoco_val2014_annotations.json",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    "textvqa_val": {
        "train": "data/textvqa/textvqa_train.jsonl",
        "test": "data/textvqa/textvqa_val.jsonl",
        "question": "data/textvqa/textvqa_val_questions.json",
        "annotation": "data/textvqa/textvqa_val_annotations.json",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    "vizwiz_val": {
        "train": "data/vizwiz/vizwiz_train.jsonl",
        "test": "data/vizwiz/vizwiz_val.jsonl",
        "question": "data/vizwiz/vizwiz_val_questions.json",
        "annotation": "data/vizwiz/vizwiz_val_annotations.json",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    "vizwiz_test": {
        "train": "data/vizwiz/vizwiz_train.jsonl",
        "test": "data/vizwiz/vizwiz_test.jsonl",
        "metric": None,
        "max_new_tokens": 10,
    },
    "docvqa_val": {
        "train": "data/docvqa/train.jsonl",
        "test": "data/docvqa/val.jsonl",
        "annotation": "data/docvqa/val/val_v1.0.json",
        "metric": "anls",
        "max_new_tokens": 100,
    },
    "docvqa_test": {
        "train": "data/docvqa/train.jsonl",
        "test": "data/docvqa/test.jsonl",
        "metric": None,
        "max_new_tokens": 100,
    },
    "chartqa_test_human": {
        "train": "data/chartqa/train_human.jsonl",
        "test": "data/chartqa/test_human.jsonl",
        "metric": "relaxed_accuracy",
        "max_new_tokens": 100,
    },
    "chartqa_test_augmented": {
        "train": "data/chartqa/train_augmented.jsonl",
        "test": "data/chartqa/test_augmented.jsonl",
        "metric": "relaxed_accuracy",
        "max_new_tokens": 100,
    },
    "gqa_testdev": {
        "train": "data/gqa/train.jsonl",
        "test": "data/gqa/testdev_balanced.jsonl",
        "metric": "accuracy",
        "max_new_tokens": 10,
    },
    "ocrvqa_val": {
        "train": "data/ocrvqa/ocrvqa_train.jsonl",
        "test": "data/ocrvqa/ocrvqa_val.jsonl",
        "metric": "accuracy",
        "max_new_tokens": 100,
    },
    "ocrvqa_test": {
        "train": "data/ocrvqa/ocrvqa_train.jsonl",
        "test": "data/ocrvqa/ocrvqa_test.jsonl",
        "metric": "accuracy",
        "max_new_tokens": 100,
    },
    "ai2diagram_test": {
        "train": "data/ai2diagram/train.jsonl",
        "test": "data/ai2diagram/test.jsonl",
        "metric": "accuracy",
        "max_new_tokens": 10,
    },
}


import json
import random
from tqdm import tqdm


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, use_train=False):
        if use_train:
            self.test = open(train).readlines()
        else:
            self.test = open(test).readlines()
        self.prompt = prompt

        self.few_shot = few_shot
        if few_shot > 0:
            self.train = open(train).readlines()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = (
            data["image"],
            data["question"],
            data["question_id"],
            data.get("answer", None),
        )

        few_shot_prompt = ""
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += (
                    self.prompt.format(sample["image"], sample["question"])
                    + f" {sample['answer']}"
                )

        return {
            "question": few_shot_prompt + self.prompt.format(image, question),
            "question_id": question_id,
            "annotation": annotation,
        }


def collate_fn(batches, tokenizer):

    questions = [_["question"] for _ in batches]
    question_ids = [_["question_id"] for _ in batches]
    annotations = [_["annotation"] for _ in batches]
    input_ids = tokenizer(questions, return_tensors="pt", padding="longest")

    return question_ids, input_ids.input_ids, input_ids.attention_mask, annotations


def calib_vqa(
    model, tokenizers, args, dataset_name, batch_size, num_workers, seed=0, few_shot=0
):
    from copy import deepcopy

    tokenizer = deepcopy(tokenizers)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eod_id
    prompt = "<img>{}</img>{} Answer:"
    dataset = VQADataset(
        train=ds_collections[dataset_name]["train"],
        test=ds_collections[dataset_name]["test"],
        prompt=prompt,
        few_shot=few_shot,
        use_train=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    idx = 0
    step = math.ceil(len(dataset) / batch_size) // args.calib_num

    print("Calibrating...")
    model_open_calibrate(model, args)

    for _, (question_ids, input_ids, attention_mask, annotations) in tqdm(
        enumerate(dataloader)
    ):
        if args.calib_mode == "v1":
            idx += 1
            max_new_tokens = ds_collections[dataset_name]["max_new_tokens"]
            if idx > args.calib_num:
                break
            if idx == args.calib_num:
                model_open_last_calibrate(model, args)
                max_new_tokens = 1
            model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=tokenizer.eod_id,
                eos_token_id=tokenizer.eod_id,
            )
        elif args.calib_mode == "v2":
            if idx % step == 0:
                max_new_tokens = ds_collections[dataset_name]["max_new_tokens"]
                if idx + step > math.ceil(len(dataset) / batch_size):
                    model_open_last_calibrate(model, args)
                    max_new_tokens = 1

                model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eod_id,
                    eos_token_id=tokenizer.eod_id,
                )
            idx += 1
        else:
            raise ValueError("Invalid calibration mode")

    model_close_calibrate(model, args)
    print("Calibrate End...")
    model_quant(model, args)


def analysis_text(model, tokenizer, analysis_num, seqlen, split="test", mode="v1"):
    tokenizer_name = tokenizer.__class__.__name__
    cached_loader = f"./cache/wikitext-2-raw-v1/{split}_{tokenizer_name}_loader.pt"
    if os.path.exists(cached_loader):
        loader = torch.load(cached_loader)
    else:
        wiki_testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
            cache_dir="./cache",
            keep_in_memory=True,
        )
        loader = tokenizer("\n\n".join(wiki_testdata["text"]), return_tensors="pt")
        os.makedirs("./cache/wikitext-2-raw-v1", exist_ok=True)
        torch.save(loader, cached_loader)
    test_loader = loader.input_ids
    nsamples = test_loader.numel() // seqlen

    batches = []
    for i in tqdm(range(nsamples)):
        if i >= analysis_num:
            break
        batch = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
        batches.append(batch)
    model(torch.cat(batches, dim=0))


def analysis(model, tokenizers, dataset_name, analysis_num, mode="v1"):
    from copy import deepcopy

    tokenizer = deepcopy(tokenizers)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eod_id
    prompt = "<img>{}</img>"
    print(dataset_name)
    dataset = VQADataset(
        train=ds_collections[dataset_name]["train"],
        test=ds_collections[dataset_name]["test"],
        prompt=prompt,
        few_shot=0,
    )

    num_data = len(dataset)
    batchs = []
    if mode == "v1":
        for i in range(num_data):
            if i >= analysis_num:
                break
            batchs.append(dataset[i])
    else:
        step = len(dataset) // analysis_num

        for i in range(analysis_num):
            batchs.append(dataset[i * step])

    _, input_ids, attention_mask, _ = collate_fn(batches=batchs, tokenizer=tokenizer)

    model.generate(
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        do_sample=False,
        num_beams=1,
        max_new_tokens=1,
        min_new_tokens=1,
        length_penalty=1,
        num_return_sequences=1,
        output_hidden_states=True,
        use_cache=True,
        pad_token_id=tokenizer.eod_id,
        eos_token_id=tokenizer.eod_id,
    )


def calib_minicpm_vqa(model, dataset, dev, dataset_name, args):
    sampler = None
    from evaluation.minicpmv.eval_utils.vqa_evaluate import collate_fn_vqa

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn_vqa,
    )

    print("Calibrating...")
    model_open_calibrate(model.model, args)

    total_batches = len(dataloader)  # 获取dataloader中总共的批次数量

    for i, batch in enumerate(tqdm(dataloader, desc="per tensor static calibrate"), 1):
        (
            image_paths,
            questions,
            gt_answers,
            ocr_tokens_list,
            question_ids,
            question_type,
        ) = batch

        if i == total_batches - 1:  # 检查是否为最后一个批次
            model_open_last_calibrate(model.model, args)
            model.generate_with_interleaved_calib(
                images=image_paths, questions=questions, datasetname=dataset_name
            )
        else:
            model.generate_with_interleaved(
                images=image_paths, questions=questions, datasetname=dataset_name
            )

    model_close_calibrate(model.model, args)
    print("Calibrate End...")
    model_quant(model.model, args)


def calib_vqa_plus(model, args, dataset, calib_num):
    lt = len(dataset.data)
    step = math.ceil(lt / calib_num)
    print("Calibrating...")
    model_open_calibrate(model.model, args)
    model.kwargs["max_new_tokens"] = 20
    for i in tqdm(range(0, lt, step)):
        if i + step >= lt:
            print("last calibrate")
            model_open_last_calibrate(model.model, args)
            model.kwargs["max_new_tokens"] = 1
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            args.dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=args.dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])
        model.generate(message=struct, dataset=args.dataset_name)

    model.kwargs = {}

    model_close_calibrate(model.model, args)
    print("Calibrate End...")
    model_quant(model.model, args)


def calib_qwen2vl_plus(model, args, dataset, calib_num):
    lt = len(dataset.data)
    step = math.ceil(lt / calib_num)
    print("Calibrating...")
    model_open_calibrate(model.model, args)
    max_new_tokens = model.generate_kwargs["max_new_tokens"]
    model.generate_kwargs["max_new_tokens"] = 20
    for i in tqdm(range(0, lt, step)):
        if i + step >= lt:
            print("last calibrate")
            model_open_last_calibrate(model.model, args)
            model.generate_kwargs["max_new_tokens"] = 1
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            args.dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=args.dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])
        model.generate(message=struct, dataset=args.dataset_name)

    model.generate_kwargs["max_new_tokens"] = max_new_tokens

    model_close_calibrate(model.model, args)
    print("Calibrate End...")
    model_quant(model.model, args)
