
CBCL/
├── README.md
├── requirements.txt
├── main.py
├── configs/
│   ├── acm.yaml
│   ├── dblp.yaml
│   └── aminer.yaml
├── data/
│   ├── ACM/
│   ├── DBLP/
│   └── AMiner/
├── models/
│   ├── cbcl.py
│   ├── structural_branch.py
│   ├── semantic_branch.py
│   ├── contrastive.py
│   └── layers.py
├── preprocessing/
│   ├── feature_transform.py
│   ├── metapath.py
│   └── pathsim.py
├── train/
│   └── trainer.py
├── evaluation/
│   ├── classification.py
│   └── clustering.py
├── utils/
│   ├── config.py
│   ├── data.py
│   ├── logger.py
│   └── seed.py
├── scripts/
│   ├── inspect_data.py
│   └── make_toy_acm.py
└── tests/
    └── test_smoke.py
```

