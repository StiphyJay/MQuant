# MQuant

Offical code for **ACM MM2025** paper [**MQuant:
Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization**](https://arxiv.org/abs/2502.00425)  ([Arxiv](https://arxiv.org/abs/2502.00425))

## News
2025.08.07: 🔥🔥🔥 MQuant for Qwen-VL2 has been released. Looking forward to your response!

2025.08.05: 🔥🔥🔥 MQuant for Intern-VL2 and MiniCPM-V has been released. 

2025.08.04: 🔥🔥🔥 MQuant for Qwen-VL has been released.

2025.07.06: 🔥🔥🔥 MQuant has been accepted by ACM MM2025.

## ToDo List
- [ ] support more MLLMs
- [ ] release the quantization code for Qwen-VL2
- [x] release the quantization code for Intern-VL2, MiniCPM-V 
- [x] release the quantization code for Qwen-VL
- [x] release the core code after the paper is accepted
- [x] update acknowledgement
- [x] release the paper link

## Highlight

- MQuant is the first quantization solution for Multimodal large language models applicable to 5 mainstream MLLMs.
- MQuant proposes the **Modality-Specific Static Quantization (MSQ)** to significantly reduce the Time-to-First-Token (TTFT) and **Rotation Magnitude Suppression (RMS)** to mitigate weight outliers.
- MQuant achieves near-floating-point accuracy (**<1%** degradation) while reducing inference latency by up to **30%** on 5 mainstram MLLMs (Qwen-VL/Intern-VL/Qwen2-VL/GLM-4V/MiniCPM-V) under **W4A8** setting.

## Quick Start

### 1. Installation

[see here](docs/install.md)

### 2. Quant Model

#### 1. QwenVL

[see here](docs/qwenvl.md)

#### 2. InternVL2

[see here](docs/internvl.md)

#### 3. Minicpmv

[see here](docs/minicpmv.md)

#### 4. Qwen2VL

[see here](docs/qwen2vl.md)

## Contact

Any questions or suggestions are welcome! Jiangyong Yu [jiangyongyufocus@gmail.com](mailto:jiangyongyufocus@gmail.com), Sifan Zhou [sifanjay@gmail.com](mailto:sifanjay@gmail.com), Dawei Yang[dawei.yang@houmo.ai](mailto:dawei.yang@houmo.ai).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=StiphyJay/MQuant&type=Date)](https://star-history.com/#StiphyJay/MQuant&Date)

## Acknowledgement

Our implementation is based on [Quarot](https://github.com/spcl/QuaRot), [GPTQ](https://github.com/IST-DASLab/gptq) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Thanks for the great open-source work!

## Citation

If you think our paper or code is helpful, please consider citing our work.

```
@inproceedings{yu2025mquant,
      title={MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization}, 
      author={JiangYong Yu and Sifan Zhou and Dawei Yang and Shuo Wang and Shuoyu Li and Xing Hu and Chen Xu and Zukang Xu and Changyong Shu and Zhihang Yuan},
      booktitle={Proceedings of the 33rd ACM international conference on multimedia (MM'25)},
      year={2025}
}
```

## License

MQuant is release under MIT license (see [LICENSE](LICENSE)).

