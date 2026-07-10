# Verification record

Verification environment: Python 3.13, PyTorch CPU.

Commands executed successfully:

```bash
python -m compileall -q .
pytest -q
# 2 passed

python scripts/make_toy_acm.py --output /tmp/CBCL_toy --papers 48
python scripts/inspect_data.py --data-dir /tmp/CBCL_toy
python main.py --config configs/acm.yaml \
  --data-dir /tmp/CBCL_toy \
  --output-dir /tmp/CBCL_run \
  --epochs 2 \
  --device cpu
```

Generated outputs were checked and were non-empty:

- `best_model.pt`
- `embeddings.npy`
- `history.json`
- `metrics.json`
- `train.log`

The smoke dataset uses the same ACM filename/layout family shown by the user:
`p_feat.npz`, `a_feat.npz`, `labels.npy`, `nei_a.npy`, `nei_s.npy`,
`pa.txt`, `ps.txt`, `pap.npz`, `psp.npz`, `pos.npz`, and split `.npy` files.

Important: the user's actual binary dataset files were not available in the conversation;
only a screenshot of the directory was available. Therefore this verification proves code
integrity and format-compatible end-to-end execution, but not exact reproduction of the
paper's numerical results on the user's private copy of ACM.
