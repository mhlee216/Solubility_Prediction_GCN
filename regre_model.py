import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool

n_features = 75
conv_dim1 = 64
conv_dim2 = 64
conv_dim3 = 64
concat_dim = 64
pred_dim1 = 64
pred_dim2 = 64
pred_dim3 = 64
out_dim = 1

class GCNlayer(nn.Module):
    def __init__(self, n_features, conv_dim1, conv_dim2, conv_dim3, concat_dim, dropout, conv):
        super(GCNlayer, self).__init__()
        self.n_features = n_features
        self.conv_dim1 = conv_dim1
        self.conv_dim2 = conv_dim2
        self.conv_dim3 = conv_dim3
        self.concat_dim =  concat_dim
        self.dropout = dropout
        self.conv = conv
        
        if self.conv == 'GCNConv':
            self.conv1 = GCNConv(self.n_features, self.conv_dim1, cached=False)
            nn.init.xavier_uniform_(self.conv1.weight)
            self.bn1 = BatchNorm1d(self.conv_dim1)
            self.conv2 = GCNConv(self.conv_dim1, self.conv_dim2, cached=False)
            nn.init.xavier_uniform_(self.conv2.weight)
            self.bn2 = BatchNorm1d(self.conv_dim2)
            self.conv3 = GCNConv(self.conv_dim2, self.conv_dim3, cached=False)
            nn.init.xavier_uniform_(self.conv3.weight)
            self.bn3 = BatchNorm1d(self.conv_dim3)
            self.conv4 = GCNConv(self.conv_dim3, self.concat_dim, cached=False)
            nn.init.xavier_uniform_(self.conv4.weight)
            self.bn4 = BatchNorm1d(self.concat_dim)
            
        elif self.conv == 'ARMAConv':
            self.conv1 = ARMAConv(self.n_features, self.conv_dim1)
            self.bn1 = BatchNorm1d(self.conv_dim1)
            self.conv2 = ARMAConv(self.conv_dim1, self.conv_dim2)
            self.bn2 = BatchNorm1d(self.conv_dim2)
            self.conv3 = ARMAConv(self.conv_dim2, self.conv_dim3)
            self.bn3 = BatchNorm1d(self.conv_dim3)
            self.conv4 = ARMAConv(self.conv_dim3, self.concat_dim)
            self.bn4 = BatchNorm1d(self.concat_dim)
            
        elif self.conv == 'SAGEConv':
            self.conv1 = SAGEConv(self.n_features, self.conv_dim1)
            self.bn1 = BatchNorm1d(self.conv_dim1)
            self.conv2 = SAGEConv(self.conv_dim1, self.conv_dim2)
            self.bn2 = BatchNorm1d(self.conv_dim2)
            self.conv3 = SAGEConv(self.conv_dim2, self.conv_dim3)
            self.bn3 = BatchNorm1d(self.conv_dim3)
            self.conv4 = SAGEConv(self.conv_dim3, self.concat_dim)
            self.bn4 = BatchNorm1d(self.concat_dim)
        
    def forward(self, data, device):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = global_add_pool(x, data.batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class FClayer(nn.Module):
    def __init__(self, concat_dim, pred_dim1, pred_dim2, pred_dim3, out_dim, dropout):
        super(FClayer, self).__init__()
        self.concat_dim = concat_dim
        self.pred_dim1 = pred_dim1
        self.pred_dim2 = pred_dim2
        self.pred_dim3 = pred_dim3
        self.out_dim = out_dim
        self.dropout = dropout

        self.fc1 = Linear(self.concat_dim*2, self.pred_dim1)
        self.bn1 = BatchNorm1d(self.pred_dim1)
        self.fc2 = Linear(self.pred_dim1, self.pred_dim2)
        self.bn2 = BatchNorm1d(self.pred_dim2)
        self.fc3 = Linear(self.pred_dim2, self.pred_dim3)
        self.fc4 = Linear(self.pred_dim3, self.out_dim)
    
    def forward(self, data):
        x = F.relu(self.fc1(data))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc4(x)
        return x

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.dropout = args.dropout
        self.conv = args.conv
        
        self.conv1 = GCNlayer(n_features, 
                              conv_dim1, 
                              conv_dim2, 
                              conv_dim3, 
                              concat_dim, 
                              self.dropout, 
                              self.conv)
        self.conv2 = GCNlayer(n_features, 
                              conv_dim1, 
                              conv_dim2, 
                              conv_dim3, 
                              concat_dim, 
                              self.dropout, 
                              self.conv)
        self.fc = FClayer(concat_dim, 
                          pred_dim1, 
                          pred_dim2, 
                          pred_dim3, 
                          out_dim, 
                          self.dropout)
        
    def forward(self, solute, solvent, device):
        x1 = self.conv1(solute, device)
        x2 = self.conv2(solvent, device)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
