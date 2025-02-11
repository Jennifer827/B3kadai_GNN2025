# B3kadai_GNN2025

## 環境

```bash
    git clone https://github.com/Jennifer827/B3kadai_GNN2025
    cd src/torch-rgcn/
    bash get_data.sh
    cd ../../
    docker compose run --rm --service-ports --build work
```

## 実行

- GCN と GAT の比較 (Cora dataset, CiteSeer dataset)
  - データセットは GCN.py, GAT.py で変更
  - 各パラメータも変更可能

```bash
    python3 [GCN, GAT].py
```

- RGCN の実験 (AIFB, MUTAG, BGS, AM dataset)
  - AM dataset は 4090 ではメモリ不足, A6000 ならできそう

```bash
    cd torch-rgcn/
    pip install -e .
    python3 experiments/classify_nodes.py with configs/e-rgcn/nc-[AIFB, MUTAG, BGS, AM].yaml
```

- GCN の実験 (AIFB, MUTAG, BGS, AM dataset)
  - AM, BGS dataset は 4090 ではできなかった.

```bash
    cd torch-rgcn/
    python3 experiments/GCN_classify_nodes.py with configs/gcn/nc-[AIFB, MUTAG, BGS, AM].yaml
```

## 使用したリポジトリ

- GCN, GAT: https://github.com/joisino/gnnbook
- R-GCN: https://github.com/thiviyanT/torch-rgcn

## 元論文

- GCN: https://arxiv.org/abs/1609.02907
- GAT: https://arxiv.org/abs/1710.10903
- R-GCN: https://arxiv.org/abs/1703.06103
