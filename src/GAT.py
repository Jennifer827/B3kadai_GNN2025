import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
# dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
data = dataset[0]

# GATの定義
class GAT(torch.nn.Module):
    def __init__(self, in_d, mid_d, out_d, heads, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_d, mid_d, heads=heads, dropout=dropout)
        self.conv2 = GATConv(mid_d * heads, out_d, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
model = GAT(dataset.num_node_features, 8, dataset.num_classes, 8)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # 一般的なoptimizer

def train(epoch):
    model.train()
    for epoch in range(epoch):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
train(500)

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(acc)