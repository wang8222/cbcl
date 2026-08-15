# Cross-Branch Contrastive Learning Algorithm in Heterogeneous Networks

This repository contains the source code, configuration files, data preparation
utilities, and evaluation scripts for the paper:

**Cross-Branch Contrastive Learning Algorithm in Heterogeneous Networks**

## Paper Information

**Manuscript ID:** 10895

## Authors and Affiliations

**Authors:** Runfeng Wang, Xingyang Zhao, Bingyu Zhang, and Zengxin Kang.

Shijiazhuang Posts and Telecommunications Technical College, Shijiazhuang, China.

## Overview

Cross-Branch Contrastive Learning (CBCL) is a self-supervised representation
learning framework for heterogeneous information networks (HINs).

CBCL contains two complementary branches:

1. **Structural-view branch**, which models direct heterogeneous relations and
   meta-path-induced structural dependencies.
2. **Path-semantic branch**, which learns meta-path-specific semantic
   representations and models their interactions using Transformer-based
   semantic fusion.

The two representations are jointly optimized through bidirectional
cross-branch contrastive learning.

Experiments are conducted on three benchmark heterogeneous information
network datasets:

- ACM
- DBLP
- AMiner

The repository supports node classification and node clustering evaluation.

---

## Repository Structure

```text
CBCL_repository/
├── README.md
├── requirements.txt
├── main.py
│
├── configs/
│   ├── acm.yaml
│   ├── dblp.yaml
│   └── aminer.yaml
│
├── data/
│   └── data
│
├── models/
│   ├── __init__.py
│   ├── cbcl.py
│   ├── structural_branch.py
│   ├── semantic_branch.py
│   ├── contrastive.py
│   └── layers.py
│
├── preprocessing/
│   ├── __init__.py
│   ├── feature_transform.py
│   ├── metapath.py
│   └── pathsim.py
│
├── train/
│   ├── __init__.py
│   └── trainer.py
│
├── evaluation/
│   ├── __init__.py
│   ├── classification.py
│   └── clustering.py
│
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── logger.py
│   └── seed.py
│
├── scripts/
│   ├── inspect_data.py
│   └── make_toy_acm.py
│
└── tests/
    ├── conftest.py
    └── test_smoke.py
```

## File Description

* `main.py`: Main program for CBCL training and evaluation.
* `configs/`: Configuration files for ACM, DBLP, and AMiner.
* `models/`: Implementation of CBCL, including the structural branch, semantic branch, and contrastive-learning modules.
* `preprocessing/`: Feature transformation, meta-path construction, and PathSim-based neighbor preprocessing.
* `train/`: Training procedure of CBCL.
* `evaluation/`: Node classification and clustering evaluation.
* `utils/`: Data loading, configuration, logging, and random-seed utilities.
* `scripts/`: Dataset inspection and auxiliary scripts.
* `tests/`: Basic tests for checking the implementation.
* `requirements.txt`: Required Python packages.

## Dataset

The experiments use three benchmark datasets: **ACM, DBLP, and AMiner**.

Because the complete datasets are too large to be stored directly in this GitHub repository, the download link is provided in:

```text
data/data
```

After downloading, please organize the datasets as:

```text
data/
├── ACM/
├── DBLP/
└── AMiner/
```

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Evaluation

The learned representations are evaluated on:

* Node classification: Macro-F1, Micro-F1, and AUC.
* Node clustering: NMI and ARI.

The corresponding experimental results are reported in Tables II and III of the paper.

## Contact

For questions about the code or experiments, please contact:

E-mail: [1293221297@qq.com](mailto:1293221297@qq.com)
