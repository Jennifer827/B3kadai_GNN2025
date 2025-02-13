# B3kadai_GNN2025

## 環境構築

```bash
    git clone https://github.com/Jennifer827/B3kadai_GNN2025
    cd src/torch-rgcn/
    bash get_data.sh # データセットのダウンロード
    cd ../../
    docker compose run --rm --service-ports --build work # 2回目以降はこれを実行するだけでOK
```

## 実行

### 実験 1 : GCN と GAT の性能比較

- データセットは GCN.py, GAT.py のコメントアウト部分で変更可能
  - Cora, CiteSeer の 2 種類
- 各パラメータも変更可能
  - optimizer の変更 : コメントアウト部分を変更
  - 学習率 : lr
  - 中間層の次元 : "model = GCN(dataset.num_node_features, 16, dataset.num_classes)" の 16 の部分
  - エポック数 : "train (500)"の 500 の部分

```bash
    python3 GCN.py
    python3 GAT.py
```

### 実験 2 : RGCN と GCN の性能比較

- データセット : AIFB, MUTAG, BGS, AM をコマンドライン引数で指定

- RGCN の場合
  - AM dataset は 4090 ではメモリ不足, A6000 ならできそう
  - パラメータ等の変更 : torch-rgcn/configs/e-rgcn/nc-[dataset].yaml で設定可能

```bash
    cd torch-rgcn/
    pip install -e . # (docker compose run 実行後初回のみ)
    python3 experiments/classify_nodes.py with configs/e-rgcn/nc-[AIFB, MUTAG, BGS, AM].yaml
```

- GCN の場合
  - AM, BGS dataset は 4090 ではできなかった.
  - パラメータ等の変更 : torch-rgcn/configs/gcn/nc-[dataset].yaml で変更可能

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
