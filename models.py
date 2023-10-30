import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn


class Predictor(nn.Module):
    def __init__(self, in_features):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)


class GCN(nn.Module):
    def __init__(self, nfeats, nhids) -> None:
        super().__init__()

        self.conv1 = tgnn.GCNConv(nfeats, nhids)
        self.conv2 = tgnn.GCNConv(nhids, nhids)
        self.predictor = Predictor(nhids)

    def forward(self, edge_index, x):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return self.predictor(x)


class GAT(nn.Module):
    def __init__(self, nfeats, nhids, nheads=8) -> None:
        super().__init__()

        self.conv1 = tgnn.GATConv(nfeats, nhids, heads=nheads, concat=True)
        self.conv2 = tgnn.GATConv(
            nhids * nheads, nhids, heads=nheads, concat=False)
        self.predictor = Predictor(nhids)

    def forward(self, edge_index, x):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return self.predictor(x)


class SAGE(nn.Module):
    def __init__(self, nfeats, nhids) -> None:
        super().__init__()

        self.conv1 = tgnn.SAGEConv(nfeats, nhids)
        self.conv2 = tgnn.SAGEConv(nhids, nhids)
        self.predictor = Predictor(nhids)

    def forward(self, edge_index, x):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return self.predictor(x)
