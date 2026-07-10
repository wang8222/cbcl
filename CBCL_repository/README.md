# CBCL: Cross-Branch Contrastive Learning

这是依据论文 **Cross-Branch Contrastive Learning Algorithm in Heterogeneous Networks** 重新整理的可运行 PyTorch 实现。代码对应论文中的四个核心部分：

1. 异构节点特征映射与 PathSim Top-K 邻居筛选；
2. 关系感知结构分支；
3. 元路径 GAT + Transformer 路径语义分支；
4. 元路径相关性扩展正样本的双向跨分支对比学习。

## 1. 目录结构

```text
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

## 2. 你的 ACM 数据放置方式

把截图中的文件直接复制到 `data/ACM/`。代码兼容 Windows 隐藏扩展名的情况，并会自动尝试 `.npy`、`.npz`、`.txt` 和无扩展名文件。

推荐文件：

```text
data/ACM/
├── a_feat.npz
├── p_feat.npz
├── labels.npy
├── nei_a.npy
├── nei_s.npy
├── pa.txt
├── ps.txt
├── pap.npz
├── psp.npz
├── pos.npz
├── train_20.npy  val_20.npy  test_20.npy
├── train_40.npy  val_40.npy  test_40.npy
└── train_60.npy  val_60.npy  test_60.npy
```

`graph.bin` 可以保留，但本实现不依赖 DGL，也不读取它。结构边优先从 `pa.txt`、`ps.txt` 读取；若文本文件不可用，则回退到 `nei_a.npy`、`nei_s.npy`。

## 3. 安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## 4. 先检查你的数据

```bash
python scripts/inspect_data.py --data-dir data/ACM
```

该脚本会打印每个文件的真实路径、格式、shape、dtype 和稀疏矩阵 nnz，便于发现扩展名或文件格式问题。

## 5. 运行 ACM

```bash
python main.py --config configs/acm.yaml
```

常用覆盖参数：

```bash
python main.py --config configs/acm.yaml --device cuda:0 --epochs 300
python main.py --config configs/acm.yaml --data-dir D:/dataset/ACM --device cpu
```

训练输出默认保存到 `runs/acm/`：

- `best_model.pt`
- `embeddings.npy`
- `metrics.json`
- `history.json`
- `train.log`

## 6. 没有真实数据时做冒烟测试

```bash
python scripts/make_toy_acm.py --output data/ACM_toy
python main.py --config configs/acm.yaml --data-dir data/ACM_toy --epochs 3 --device cpu
pytest -q
```

## 7. 论文公式与代码对应

| 论文内容 | 代码 |
|---|---|
| 式 (2) 类型特定特征变换 | `preprocessing/feature_transform.py` |
| 式 (3) PathSim | `preprocessing/pathsim.py` |
| 式 (4) 元路径邻接归一化与融合 | `preprocessing/metapath.py`, `models/structural_branch.py` |
| 式 (5)-(6) 结构分支 | `models/structural_branch.py` |
| 式 (7)-(13) 路径语义分支 | `models/semantic_branch.py`, `models/layers.py` |
| 式 (14) 正样本扩展 | `preprocessing/metapath.py` |
| 式 (15)-(17) 双向对比损失 | `models/contrastive.py`, `models/cbcl.py` |

## 8. 数据兼容说明

- `labels`、`train_20` 等即使 Windows 资源管理器隐藏了 `.npy`，加载器也会自动找到。
- `nei_a.npy`、`nei_s.npy` 可为 object array，每个元素是一组邻居编号。
- `pap.npz`、`psp.npz` 由 `scipy.sparse.save_npz` 保存时可直接读取。
- `pa.txt`、`ps.txt` 支持空格、Tab 或逗号分隔，并尝试识别 0/1 起始编号与全局偏移编号。
- `pos.npz` 仅用于兼容旧数据，不直接作为训练正样本。CBCL 按论文从 PathSim 过滤后的元路径邻接重新构造正样本。

## 9. 重要说明

论文中的完整复现实验还依赖具体随机种子、原始数据预处理、每条元路径的 Top-K 和训练轮数。默认配置提供合理起点，但不承诺一次运行就精确得到论文表格中的同一小数值。
