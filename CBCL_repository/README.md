# Cross-Branch Contrastive Learning Algorithm in Heterogeneous Networks

This repository contains the source code, configuration files, data preparation
utilities, and evaluation scripts for the paper:

**Cross-Branch Contrastive Learning Algorithm in Heterogeneous Networks**

## Paper Information

**Manuscript ID:** [10895]

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
