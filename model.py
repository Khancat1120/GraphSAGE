import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv
from dgl.data import KarateClubDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, aggregator_type='mean'):
        super().__init__()
        self.sage_1 = SAGEConv(in_features, hidden_features, aggregator_type)
        self.sage_2 = SAGEConv(hidden_features, out_features, aggregator_type)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, g, x, etype=None):
        x = x.to(self.device)
        g = g.to(self.device)
        x = self.sage_1(g, x)
        x = self.sage_2(g, x)
        return F.normalize(x, p=2, dim=1)
# 定义GraphSAGE模型
# class GraphSAGE(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats, aggregator_type='mean'):
#         super(GraphSAGE, self).__init__()
#         self.layers = nn.ModuleList()
#         self.sage_1 = SAGEConv(in_feats, hidden_feats, aggregator_type)
#         self.sage_2 = SAGEConv(hidden_feats, out_feats, aggregator_type)
#
#     def forward(self, g, features, etype):
#         res = features
#         res = self.sage_1(g, res)
#         res = self.sage_2(g, res)
#         res = F.dropout(res, p=0.1, training=self.training)
#
#         return F.normalize(res, p=2, dim=1)
