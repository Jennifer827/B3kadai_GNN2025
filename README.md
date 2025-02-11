# B3kadai_GNN2025

## 環境

```bash
    docker compose run --rm --service-ports --build work
```

## 実行

- GCN と GAT の比較 (Cora dataset, CiteSeer dataset)

```bash
    cd src
    python3 src/[GCN, GAT].py
```

- RGCN の実験 (AIFB, MUTAG, BGS, AM dataset)
  - AM dataset は 4090 ではメモリ不足

```bash
    cd src/torch-rgcn
    python3 experiments/classify_nodes.py with configs/e-rgcn/nc-[AIFB, MUTAG, BGS, AM].yaml
```
