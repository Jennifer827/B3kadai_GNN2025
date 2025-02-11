from utils.misc import create_experiment
from utils.data import load_node_classification_data
from sklearn.metrics import accuracy_score
from statistics import stdev
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

"""
Graph Convolutional Network (GCN) for node classification.
この実装では、関係情報を無視して通常のGCNによるノード分類を行います。
"""

# Sacred を用いて実験を管理
ex = create_experiment(name='GCN Node Classification', database='node_class')

class GCN(torch.nn.Module):
    def __init__(self, in_d, mid_d, out_d):
        super().__init__()
        self.conv1 = GCNConv(in_d, mid_d)
        self.conv2 = GCNConv(mid_d, out_d)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        emb = x.detach()  # 中間層の埋め込み（必要に応じて利用）
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), emb

@ex.capture
def train_model(dataset,
                training,
                gcn,
                evaluation,
                repeat,
                _run):
    # 訓練設定のデフォルト値をセット
    epochs = training["epochs"] if "epochs" in training else 50
    nhid = gcn["hidden_size"] if "hidden_size" in gcn else 16
    final_run = evaluation["final_run"] if "final_run" in evaluation else False

    # --- データの読み込み ---
    # load_node_classification_data の返り値は以下の通り:
    #   triples: (src, relation, dst) のリスト（関係は無視する）
    #   (n2i, i2n): ノード名とインデックスの対応
    #   (r2i, i2r): 関係の対応
    #   train, test: {node_name: class} の辞書
    triples, (n2i, i2n), (r2i, i2r), train, test = load_node_classification_data(
        dataset["name"], use_test_set=final_run, prune=dataset["prune"]
    )

    # --- グラフ構築 ---
    # triples から (src, dst) エッジを抽出し、無向グラフとするため (dst, src) も追加
    edges = []
    for src, r, dst in triples:
        edges.append((src, dst))
        edges.append((dst, src))
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # shape: [2, num_edges]
    num_nodes = len(n2i)
    # ※各ノードの特徴量が存在しないため、ここでは one-hot 表現 (単位行列) を使用
    x = torch.eye(num_nodes, dtype=torch.float)

    # --- ラベルとマスクの作成 ---
    # すべてのノードのラベルは -1 で初期化し、訓練・テストノードには正しいラベルを設定
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for name, label in train.items():
        idx = n2i[name]
        y[idx] = label
        train_mask[idx] = True

    for name, label in test.items():
        idx = n2i[name]
        y[idx] = label
        test_mask[idx] = True

    # PyG の Data オブジェクトの作成
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    # GPU の有無チェック
    use_cuda = training["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    data = data.to(device)

    # --- クラス数の決定 ---
    # train, test 辞書に含まれるクラスからユニークなクラス数を算出
    classes = set(list(train.values()) + list(test.values()))
    num_classes = len(classes)

    # --- モデル構築 ---
    # 入力次元は one-hot 表現の場合 num_nodes、出力はクラス数
    in_dim = data.num_node_features  # この場合、in_dim = num_nodes
    model = GCN(in_dim, nhid, num_classes)
    if use_cuda:
        model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

    print("num_nodes:", num_nodes, "num_classes:", num_classes)

    # --- 学習 ---
    def train(epochs):
        model.train()
        for e in range(epochs):
            optimizer.zero_grad()
            out, _ = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if (e + 1) % 10 == 0:
                print(f"Epoch {e+1}/{epochs}, Loss: {loss.item():.4f}")

    train(epochs)

    # --- 評価 ---
    model.eval()
    with torch.no_grad():
        logits, _ = model(data)
        pred = logits.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        total = int(data.test_mask.sum())
        acc = correct / total if total > 0 else 0
    print("Test Accuracy:", acc)
    return acc


@ex.automain
def repeat(_run, repeats=1):
    test_accuracies = []
    for i in range(1, repeats + 1):
        test_accuracy = train_model(repeat=i)
        test_accuracies.append(test_accuracy)

    avg = sum(test_accuracies) / len(test_accuracies)
    std = stdev(test_accuracies) if len(test_accuracies) != 1 else 0
    ste = std / (len(test_accuracies) ** 0.5)

    avg = round(avg, 2)
    ste = round(ste, 2)

    _run.log_scalar("test.accuracy", avg)
    _run.log_scalar("test.accuracy_ste", ste)
    _run.log_scalar("repeats", repeats)

    print(f'[Summary] Test Accuracy: {avg:.2f} -/+ {ste:.2f} {f"({repeats} runs)" if repeats > 1 else ""}')
