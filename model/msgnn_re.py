import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.layers_re import Fssn_layers, AttentionLayer, Dscrmnt_v2, AttentionLayer_wosubgraph, Fssn_layers_wosubgraph

    
class MSGNN(nn.Module):
    def __init__(self, ntype, in_dims, hidden_dim, out_dim, nheads, dropout_rate, alpha, beta, device, nlayers = 2, graph_readout = 'max', ablation = None):
        super(MSGNN, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(alpha)
        self.ntype = ntype
        self.device = device
        self.beta = beta
        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.nlayers = nlayers - 1 # exclude output layer
        if self.nlayers < 0:
            self.nlayers = 0 
        
        self.W = self._build_W(self.in_dims)
        
        global_readout = False if ablation == 'global' else True
        
        self.Att = nn.ModuleList([AttentionLayer(self.ntype, hidden_dim, dropout_rate, alpha, beta, nheads, device, global_readout) for _ in range(self.nlayers)])
        self.Att_out = AttentionLayer(self.ntype, hidden_dim, dropout_rate, alpha, beta, nheads, device, global_readout)

        self.fssn_layer = nn.ModuleList([Fssn_layers(self.ntype, alpha, device, concat=True) for _ in range(self.nlayers)])
        self.fssn_layer_out = Fssn_layers(self.ntype, alpha, device, concat=False)

        self.fc = nn.ModuleList([nn.Linear(in_features = hidden_dim * nheads, out_features = hidden_dim) for _ in range(self.nlayers)])
        self.fc_out = nn.Linear(in_features = hidden_dim, out_features = self.out_dim)
        
        self.DNT = Dscrmnt_v2(self.ntype, self.out_dim, alpha, device, graph_readout, ablation)
    
    
    def _build_W(self, in_dims):
        Wr_dict = {key: [nn.Linear(in_dims[key],self.hidden_dim), self.LeakyReLU] for key in range(len(in_dims))}
        Wr = nn.ModuleDict({str(key):nn.Sequential(*value) for key,value in Wr_dict.items()})
        return Wr

    def forward(self, inputs):
        features, batch, perturbation = inputs
        attentions = []
        
        features = [self.W[str(p)](features[p]) for p in range(self.ntype)] # type-specific projection
        features = torch.cat(features, dim = 0)
        
        for layer_num in range(self.nlayers):
            attentions = self.Att[layer_num](features, batch, attentions)
            features = self.fssn_layer[layer_num](batch, features, attentions)
            features = self.fc[layer_num](features)
            features = self.LeakyReLU(features)
            
        attentions = self.Att_out(features, batch, attentions)
        features = self.fssn_layer_out(batch, features, attentions)       
        features = self.fc_out(features)
        
        features = F.normalize(features, p=2, dim=1)
        
        if perturbation is not None:                                        
            logits_node, logits_graph = self.DNT(features, perturbation)
            logits_node = logits_node.flatten()
            logits_graph = logits_graph.flatten()
        else:
            logits_node = []
            logits_graph = []

        return features, logits_node, logits_graph


class MSGNN_wosub(nn.Module):
    def __init__(self, ntype, in_dims, hidden_dim, out_dim, nheads, dropout_rate, alpha, beta, device, nlayers = 2, graph_readout = 'max', ablation = None):
        super(MSGNN_wosub, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(alpha)
        self.ntype = ntype
        self.device = device
        self.beta = beta
        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.nlayers = nlayers - 1 # exclude output layer
        if self.nlayers < 0:
            self.nlayers = 0 
        
        self.W = self._build_W(self.in_dims)
        
        global_readout = True

        self.Att = nn.ModuleList([AttentionLayer_wosubgraph(self.ntype, hidden_dim, dropout_rate, alpha, beta, nheads, device, global_readout) for _ in range(self.nlayers)])
        self.Att_out = AttentionLayer_wosubgraph(self.ntype, hidden_dim, dropout_rate, alpha, beta, nheads, device, global_readout)

        self.fssn_layer = nn.ModuleList([Fssn_layers_wosubgraph(self.ntype, alpha, device, concat=True) for _ in range(self.nlayers)])
        self.fssn_layer_out = Fssn_layers_wosubgraph(self.ntype, alpha, device, concat=False)

        self.fc = nn.ModuleList([nn.Linear(in_features = hidden_dim * nheads, out_features = hidden_dim) for _ in range(self.nlayers)])
        self.fc_out = nn.Linear(in_features = hidden_dim, out_features = self.out_dim)
        
        self.DNT = Dscrmnt_v2(self.ntype, self.out_dim, alpha, device, graph_readout, ablation)
    
    
    def _build_W(self, in_dims):
        Wr_dict = {key: [nn.Linear(in_dims[key],self.hidden_dim), self.LeakyReLU] for key in range(len(in_dims))}
        Wr = nn.ModuleDict({str(key):nn.Sequential(*value) for key,value in Wr_dict.items()})
        return Wr

    def forward(self, inputs):
        features, batch, perturbation = inputs
        attentions = []
        
        features = [self.W[str(p)](features[p]) for p in range(self.ntype)]
        features = torch.cat(features, dim = 0)
        
        for layer_num in range(self.nlayers):
            attentions, neighbors = self.Att[layer_num](features, batch, attentions)
            features = self.fssn_layer[layer_num](neighbors, features, attentions)
            features = self.fc[layer_num](features)
            features = self.LeakyReLU(features)
            
        attentions, neighbors = self.Att_out(features, batch, attentions)
        features = self.fssn_layer_out(neighbors, features, attentions)       
        features = self.fc_out(features)
        
        features = F.normalize(features, p=2, dim=1)
        
        if perturbation is not None:                                        
            logits_node, logits_graph = self.DNT(features, perturbation)
            logits_node = logits_node.flatten()
            logits_graph = logits_graph.flatten()
        else:
            logits_node = []
            logits_graph = []

        return features, logits_node, logits_graph       