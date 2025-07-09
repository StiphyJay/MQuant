
2025.07.06: 🔥🔥🔥 MQuant has been accepted by accppted by ACM MM2025.

# MQuant

Offical code for for ACM MM2025 paper MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization.*([Paper](https://arxiv.org/abs/2502.00425))*

[**MQuant:
Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization**](https://arxiv.org/abs/2502.00425)\


## Highlight
- MQuant is the first quantization solution for Multimodal large language models applicable to 5 mainstream MLLMs.
- MQuant proposes the **Modality-Specific Static Quantization (MSQ)** to significantly reduce the Time-to-First-Token (TTFT) and **Rotation Magnitude Suppression (RMS)** to mitigate weight outliers.
- MQuant achieves near-floating-point accuracy (**<1%** degradation) while reducing inference latency by up to **30%** on 5 mainstram MLLMs (Qwen-VL/Intern-VL/Qwen2-VL/GLM-4V/MiniCPM-V) under **W4A8** setting.

## ToDo List
- [ ] release the quantization code for other MLLMs
- [ ] release the quantization code for Qwen-VL
- [ ] release the core code after the paper accepted
- [ ] update acknowledgement
- [x] release the paper link

## Contact
Any questions or suggestions are welcome! [Jiangyong Yu] [jiangyongyufocus@gmail.com](mailto:jiangyongyufocus@gmail.com)
, Dawei Yang[dawei.yang@houmo.ai](mailto:dawei.yang@houmo.ai), Sifan Zhou [sifanjay@gmail.com](mailto:sifanjay@gmail.com)

## Abstract
Recently, multimodal large language models (MLLMs) have garnered widespread attention due to their ability to perceive and understand multimodal signals. However, their large parameter sizes and substantial computational demands severely hinder their practical deployment and application. While quantization is an effective way to reduce model size and inference latency, its application to MLLMs remains underexplored. In this paper, we conduct an in-depth analysis of MLLMs quantization and identify several challenges: slow inference speed of the visual tokens, distributional differences across modalities, and visual outlier clipping degrades performance. To address these challenges, we propose MQuant, a quantization framework tailored for MLLMs. Specifically, 1) we design Modality-specific Quantization (MSQ) and Attention-Invariant Flexible Switching (AIFS) to support per-tensor static quantization and facilitate efficient inference. 2) we introduce a unified LayerNorm-to-RMSNorm transformation, achieving seamless integration of the MLLM vision encoder with Hadamard rotation. 3) we propose Rotation Magnitude Suppression (RMS) to mitigate outliers introduced by Hadamard rotation. Experiments conducted on five mainstream MLLMs demonstrate the superior performance and broad applicability of MQuant. For example, it maintains around 98% of the floating-point accuracy under the W4A8 setting. To the best of our knowledge, MQuant is the first quantization solution for MLLMs, paving the way for future advancements in their application.


  
## License
MQuant is release under MIT license (see [LICENSE](LICENSE)).

## Citation
If you think our paper or code is helpful, please consider citing our work.
```
@misc{yu2025mquantunleashinginferencepotential,
      title={MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization}, 
      author={JiangYong Yu and Sifan Zhou and Dawei Yang and Shuo Wang and Shuoyu Li and Xing Hu and Chen Xu and Zukang Xu and Changyong Shu and Zhihang Yuan},
      year={2025},
      eprint={2502.00425},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.00425}, 
}
```

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=StiphyJay/MQuant&type=Date)](https://star-history.com/#StiphyJay/MQuant&Date)
