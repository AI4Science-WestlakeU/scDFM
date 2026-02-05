<p align="center">
  <img src="assets/logo.png" alt="scDFN logo" width="400" />
</p>

# scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2601.01829-b31b1b?logo=arxiv)](https://openreview.net/forum?id=QSGanMEcUV)
[![Codebase](https://img.shields.io/badge/Codebase-GitHub-181717?logo=github)](https://github.com/AI4Science-WestlakeU/scDFM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?logo=open-source-initiative&logoColor=white)](LICENSE)


Official repo for the paper [scDFM](URL), ICLR 2026. <br />
Chenglei Yu<sup>∗1,2</sup>, [Chuanrui Wang](https://wang-cr.github.io/)<sup>∗1</sup>, Bangyan Liao<sup>∗1,2</sup> & [Tailin Wu](https://tailin.org/)<sup>†1</sup>.<br />

<sup>1</sup>School of Engineering, Westlake University; 
<sup>2</sup>Zhejaing University;

</sup>*</sup>Equal contribution, </sup>†</sup>Corresponding authors

----

## Overview 
We propose a novel distributional flow matching framework (scDFM) for robust single-cell perturbation prediction, which models the full distribution of perturbed cellular expression profiles conditioned on control states, thereby overcoming limitations of existing methods that rely on cell-level correspondences and fail to capture population-level transcriptional shifts.

Framework of paper:

<a href="url"><img src="assets/fig1.png" align="center" width="600" ></a>

## Install dependencies 
```
conda env create -f environment.yml
```

##  ⏬ Dataset download

Put dataset into data file:

- [Norman](https://figshare.com/articles/dataset/Norman_et_al_2019_Science_labeled_Perturb-seq_data/24688110?utm_source=chatgpt.com&file=43390776)
- [Combosciplex subset of sciplex v3](https://figshare.com/articles/dataset/combosciplex/25062230?file=44229635)

Example directory layout after download (relative to repo root):
```
scDFM/
├─ data/
│  ├─ norman.h5ad
│  └─ combosciplex.h5ad
├─ src/
│  └─ ...
└─ run.sh
```



## 📥 Training

An example on additive task.
```bash
bash run.sh
```

## 🫡 Citation

If you find our work and/or our code useful, please cite us via:

```bibtex
comming soon
```

## 📚 Related Resources

- AI for Scientific Simulation and Discovery Lab: https://github.com/AI4Science-WestlakeU
