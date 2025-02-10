import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

dataset = Planetoid(root='/tmp/Cora', name='Cora')
# dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
n = dataset[0].num_nodes

print("グラフの数: ", len(dataset))
print("クラスの数:",dataset.num_classes)
print("ノードの特徴量の次元: ", dataset.num_node_features) # Cora: 1433種類の特定のワードが論文中に含まれているか 0/1データとなっている
print("1つめのグラフ: ", dataset[0])

# グラフ畳み込みネットワークの定義
class GCN(torch.nn.Module):
    def __init__(self, in_d, mid_d, out_d):
        super().__init__()
        self.conv1 = GCNConv(in_d, mid_d)
        self.conv2 = GCNConv(mid_d, out_d)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        emb = x.detach()
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), emb

model = GCN(dataset.num_node_features, 16, dataset.num_classes)

data = dataset[0]
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

def train(epoch):
    model.train()
    for epoch in range(epoch):
        optimizer.zero_grad()
        out = model(data)[0]
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
train(500)

model.eval()
pred = model(data)[0].argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(acc)