# LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory

> **Notice: This is a reimplementation of LoGeR; complete code and models will be released upon approval.**

LoGeR processes long video streams in chunks with a hybrid memory design to improve large-scale geometric reconstruction quality and consistency.

[**LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory**](https://arxiv.org/abs/2603.03269) [*Junyi Zhang*](https://junyi42.github.io/), [*Charles Herrmann*](https://scholar.google.com/citations?user=LQvi5XAAAAAJ), [*Junhwa Hur*](https://hurjunhwa.github.io/), [*Chen Sun*](https://chensun.me/index.html), [*Ming-Hsuan Yang*](https://faculty.ucmerced.edu/mhyang/), [*Forrester Cole*](https://scholar.google.com/citations?user=xZRRr-IAAAAJ&hl), [*Trevor Darrell*](https://people.eecs.berkeley.edu/~trevor/), [*Deqing Sun*](https://deqings.github.io/)
| [**[Project Webpage]**](https://LoGeR-project.github.io/) | [**[arXiv]**](https://arxiv.org/abs/2603.03269)

<p align="center">
  <img src="https://loger-project.github.io/figs/fig1_teaser.png" alt="LoGeR Teaser" width="100%">
</p>

## Installation

```bash
git clone https://github.com/junyi42/LoGeR
cd LoGeR
conda create -n loger python=3.11 cmake=3.14.0
conda activate loger
pip install -e .
```

The reusable Python package now lives under `src/loger`.

## Checkpoint Download

Checkpoints are hosted on [Hugging Face](https://huggingface.co/Junyi42/LoGeR):


Please place files as:
- `ckpts/LoGeR/latest.pt`
- `ckpts/LoGeR_star/latest.pt`

Example commands:

```bash
wget -O ckpts/LoGeR/latest.pt "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR/latest.pt?download=true"
wget -O ckpts/LoGeR_star/latest.pt "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR_star/latest.pt?download=true"
```

## Demo

For demo usage, please directly refer to:

- [`demo_run.sh`](demo_run.sh)

## Evaluation

For evaluation instructions, please refer to:

- [`eval/eval.md`](eval/eval.md)

## Citation

If you find our work useful, please cite:

```bibtex
@article{zhang2026loger,
  title={LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory},
  author={Zhang, Junyi and Herrmann, Charles and Hur, Junhwa and Sun, Chen and Yang, Ming-Hsuan and Cole, Forrester and Darrell, Trevor and Sun, Deqing},
  journal={arXiv preprint arXiv:2603.03269},
  year={2026}
}
```

## Acknowledgments
Our code is based on [Pi3](https://github.com/yyfz/Pi3) and [LaCT](https://github.com/a1600012888/LaCT), our camera pose estimation evaluation script is based on [TTT3R](https://github.com/Inception3D/TTT3R) & [VBR](https://github.com/rvp-group/vbr-slam-benchmark), and our visualization code is based on [Viser](https://github.com/nerfstudio-project/viser). We thank the authors for their excellent work!
