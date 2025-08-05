# minicpmv

## Model Change

### 1.1 Download the model weights

```shell
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openbmb/MiniCPM-V-2_6
cd MiniCPM-V-2_6
git lfs pull
ln -s path weights
```

### 1.2 generate the opt version model weights

#### 1.2.3.1 run script

copy weights folder and run the optimize script

```shell
cp -r weights/MiniCPM-V-2_6 weights/MiniCPM-V-2_6-opt
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
python exam/minicpmv_opt.py
```

#### replace weights/MiniCPM-V-2_6-opt/resampler.py

```bash
cp plugin/resampler.py weights/MiniCPM-V-2_6-opt/
```

## Quantization

### OCRBench

#### OCRBench w8a8 + w4a8

```shell
export PYTHONPATH=.
python exam/quant_minicpmv.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention  --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name OCRBench --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
```

#### OCRBench w4a8 + w4a8

```shell
export PYTHONPATH=.
python exam/quant_minicpmv.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention  --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name OCRBench --nsamples 128 --calib_num 128 --online_visual_hadamard --visual_split
```

### MME

#### MME w8a8 + w4a8

```shell
export PYTHONPATH=.
python exam/quant_minicpmv.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention  --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name MME --nsamples 256 --calib_num 512 --online_visual_hadamard --visual_split
```

#### MME w4a8 + w4a8

```shell
export PYTHONPATH=.
python exam/quant_minicpmv.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --quant --quant_llm --quant_visual_clip --quant_cross_attention  --visual_w_clip --llm_w_clip --visual_static --llm_static --online_llm_hadamard --act_order --dataset_name MME --nsamples 256 --calib_num 512 --online_visual_hadamard --visual_split
```

### TextVQA

#### TextVQA w8a8 + w4a8

```shell
export PYTHONPATH=.
python exam/quant_minicpmv.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --visual_static --visual_w_clip --llm_w_clip --act_order --online_llm_hadamard  --llm_static --dataset_name TextVQA_VAL --online_visual_hadamard --visual_split --calib_num 256 --nsamples 128
```

#### TextVQA w4a8 + w4a8

```shell
export PYTHONPATH=.
python exam/quant_minicpmv.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --visual_static --visual_w_clip --llm_w_clip --act_order --online_llm_hadamard  --llm_static --dataset_name TextVQA_VAL --online_visual_hadamard --visual_split --calib_num 256 --nsamples 128
```

### DocVQA

#### DocVQA w8a8 + w4a8

```shell
export PYTHONPATH=.
python exam/quant_minicpmv.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_bits 8 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --visual_static --visual_w_clip --llm_w_clip --act_order --online_llm_hadamard  --llm_static --dataset_name DocVQA_VAL --online_visual_hadamard --visual_split --calib_num 256 --nsamples 128
```

#### DocVQA w4a8 + w4a8

```shell
export PYTHONPATH=.
python exam/quant_minicpmv.py --rotate --rotate_visual_clip --rotate_visual_cross_attn --rotate_llm --quant --quant_llm --quant_visual_clip --quant_cross_attention --visual_w_bits 4 --visual_a_bits 8 --llm_w_bits 4 --llm_a_bits 8 --visual_static --visual_w_clip --llm_w_clip --act_order --online_llm_hadamard  --llm_static --dataset_name DocVQA_VAL --online_visual_hadamard --visual_split --calib_num 256 --nsamples 128
```
