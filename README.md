# MQuant

Offical code for **ACM MM2025** paper [**MQuant:
Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization**](https://arxiv.org/abs/2502.00425)  ([Arxiv](https://arxiv.org/abs/2502.00425))

## News

2025.07.06: 🔥🔥🔥 MQuant has been accepted by ACM MM2025.

## ToDo List
- [ ] release the quantization code for other MLLMs
- [ ] release the quantization code for Qwen-VL
- [ ] release the core code after the paper accepted
- [ ] update acknowledgement
- [x] release the paper link


## Highlight
- MQuant is the first quantization solution for Multimodal large language models applicable to 5 mainstream MLLMs.
- MQuant proposes the **Modality-Specific Static Quantization (MSQ)** to significantly reduce the Time-to-First-Token (TTFT) and **Rotation Magnitude Suppression (RMS)** to mitigate weight outliers.
- MQuant achieves near-floating-point accuracy (**<1%** degradation) while reducing inference latency by up to **30%** on 5 mainstram MLLMs (Qwen-VL/Intern-VL/Qwen2-VL/GLM-4V/MiniCPM-V) under **W4A8** setting.


## Quick Start

## Contact
Any questions or suggestions are welcome! Jiangyong Yu [jiangyongyufocus@gmail.com](mailto:jiangyongyufocus@gmail.com), Sifan Zhou [sifanjay@gmail.com](mailto:sifanjay@gmail.com), Dawei Yang[dawei.yang@houmo.ai](mailto:dawei.yang@houmo.ai).

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=StiphyJay/MQuant&type=Date)](https://star-history.com/#StiphyJay/MQuant&Date)

  
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
## License
MQuant is release under MIT license (see [LICENSE](LICENSE)).

