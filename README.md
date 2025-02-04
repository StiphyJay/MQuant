# MQuant
Offical code for MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization

[**MQuant:
Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization**](https://arxiv.org/abs/2502.00425)\
*Jiangyong Yu, Sifan Zhou, Dawei Yang, Shuoyu Li, Shuo Wang, Xing Hu, Chen Xu, Zukang Xu, Changyong Shu and Zhihang Yuan*\
Houmo AI, Southeast University, Xian Jiaotong University,\
*Paper ([arXiv 2302.02367](https://arxiv.org/abs/2502.00425))*



## ToDo List
- [ ] release the quantization code
- [x] release the paper linke

## Contact
Any questions or suggestions are welcome! Jiangyong Yu[sifanjay@gmail.com](mailto:sifanjay@gmail.com)
, Sifan Zhou [sifanjay@gmail.com](mailto:sifanjay@gmail.com)

## Abstract
Recently, multimodal large language models (MLLMs) have garnered widespread attention due to their ability to perceive and understand multimodal signals. However, their large parameter sizes and substantial computational demands severely hinder their practical deployment and application. While quantization is an effective way to reduce model size and inference latency, its application to MLLMs remains underexplored. In this paper, we conduct an in-depth analysis of MLLMs quantization and identify several challenges: slow inference speed of the visual tokens, distributional differences across modalities, and visual outlier clipping degrades performance. To address these challenges, we propose MQuant, a quantization framework tailored for MLLMs. Specifically, 1) we design Modality-specific Quantization (MSQ) and Attention-Invariant Flexible Switching (AIFS) to support per-tensor static quantization and facilitate efficient inference. 2) we introduce a unified LayerNorm-to-RMSNorm transformation, achieving seamless integration of the MLLM vision encoder with Hadamard rotation. 3) we propose Rotation Magnitude Suppression (RMS) to mitigate outliers introduced by Hadamard rotation. Experiments conducted on five mainstream MLLMs demonstrate the superior performance and broad applicability of MQuant. For example, it maintains around 98% of the floating-point accuracy under the W4A8 setting. To the best of our knowledge, MQuant is the first quantization solution for MLLMs, paving the way for future advancements in their application.


  
## License
MQuant is release under MIT license (see [LICENSE](LICENSE)).

## Citation
If you think our paper or code is helpful, please consider citing our work.
```
@article{Yu2025Mquant,
title={{MQ}uant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization},
author={Jiangyong Yu, Sifan Zhou, Dawei Yang, Shuoyu Li, Shuo Wang, Xing Hu, XUCHEN, Zukang Xu, Changyong Shu and Zhihang Yuan},
year={2025},
}
```

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=StiphyJay/MQuant&type=Date)](https://star-history.com/#StiphyJay/MQuant&Date)
