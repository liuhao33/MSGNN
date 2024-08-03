import numpy as np
import torch     
import torch.nn as nn
import torch.nn.functional as F

class OHNN_layers(nn.Module):
    def __init__(self, ntype, alpha, device, agg = 'max', concat=True):
        super(OHNN_layers, self).__init__()

        self.concat = concat
        self.ntype = ntype
        self.device = device
        self.agg = agg
        self.leakyrelu = nn.LeakyReLU(alpha)
    
    def aggregator(self, batch, att_feat):
        nodes = torch.unique(batch)
        batch_nodes = batch.T.flatten()
        
        if self.agg == 'avg':
            node_aggregation = torch.zeros((len(nodes),att_feat.shape[0]), dtype = torch.float32) # 4_unique(batch) x batch
            for node in nodes:
                mean = 1/(batch_nodes == nodes).sum()
                node_aggregation[node][(nodes == node)] = mean
                
            aggregated = torch.matmul(node_aggregation.to(self.device), att_feat.transpose(0,1)) # should be 4_heads x 4_unique(batch) x 334
            
        if self.agg == 'max':
            aggregated = torch.stack([torch.max(att_feat[batch_nodes == node], dim = 0)[0] for node in nodes])
            aggregated = aggregated.transpose(0,1) # 4heads x unique_nodes x dims
            
        return aggregated


    def forward(self, batch, batch_features, att_weights):
        filters = ~(torch.eye(self.ntype).bool())
        feat = torch.cat([F.embedding(batch[:,filter], batch_features) for filter in filters])
        att_feat = torch.matmul(att_weights,feat) # batch x 4_heads x 334
             
        aggregated = self.aggregator(batch, att_feat)  # aggregated is 4_heads x 4_unique(batch) x dim
        batch_features = batch_features + aggregated # skip-connection for x, (unique(batch) x dim) + (4_heads x unique(batch) x dims) = (4_heads x unique(batch) x dims)
        batch_features = batch_features.transpose(0,1) # unique(batch) x 4_heads x dim
            
        if self.concat:
            batch_features = self.leakyrelu(batch_features) # activation
            batch_features = batch_features.reshape(batch_features.shape[0],-1) # 4_unique(batch) x (4_heads x 334)
        else: 
            batch_features = torch.mean(batch_features,dim=1)  # 4_unique(batch) x 334)

        return batch_features
    

class AttentionLayer_wg(nn.Module):
    def __init__(self, ntype, hidden_dim, dropout_rate, alpha, beta, nheads, device, readout = 'avg'):   # embedding中的维度一致吗？
        super(AttentionLayer_wg, self).__init__()
        self.ntype = ntype
        self.batch_size = None    # 即num of subgraphs
        self.beta = beta  # 注意力残差连接参数
        self.nheads = nheads       # 是否不需要nheads
        self.device = device
        self.readout = readout
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout_rate)
        self.a = nn.ParameterList([nn.Parameter(torch.empty(size=(nheads, 3 * hidden_dim), device=device))
                                   for i in range(ntype)]) # self.a的size = nheads * 3 hidden * 1
        for i in range(ntype):
            nn.init.xavier_uniform_(self.a[i], gain=1.414)

    # subgraph的每一行都是一个本体子图，存储了该本体子图embedding的索引号，该索引号负责从feats中获取获取该本体子图的特征
    # _prepare_attentional_mechanism_input 负责对本体子图的embeddings作拼接
    def forward(self, feats, subgraphs, attentions):
        self.batch_size = len(subgraphs)

        subgraphs_features = torch.stack([feats[p] for p in subgraphs])
        a_input = self._prepare_attentional_mechanism_input(feats, subgraphs_features, subgraphs)
        # a_input 的shape = (4*batch_size，subgraph_dim-1, 2*features_dim)
        att_temp = torch.cat([torch.matmul(a_input[i*self.batch_size:(i+1)*self.batch_size], self.a[i].T) for i in range(self.ntype)])
        #  a_input是(batch_size, subgraph_dim-1, 3×F)的张量，a[i]是shape为(nheads, 3×F)的张量
        att_temp = att_temp.transpose(1,2)
        # att_temp的shape = (4*batch_size, nheads, subgraph_dim-1)
        att_temp = self.leakyrelu(att_temp)

        att_temp = F.softmax(att_temp, dim=2)
        # attention = attention.reshape(self.batch_size, self.nheads, -1)    # attention的shape = (batch_size, nheads, subgraph_dim-1)
        att_temp = self.dropout(att_temp)

        if len(attentions) != 0:
            attentions = (1 - self.beta) * att_temp + self.beta * attentions.detach()
        else:
            attentions = att_temp

        return attentions

    def _prepare_attentional_mechanism_input(self, feat, subgraphs_features, subgraphs):
        N = self.ntype  # == subgraphs_features.size()[1]   # subgraph_dim，一个subgraph包含的节点数，固定值
        batch_len = len(subgraphs)
        all_combinations_matrix = []
        
        for i in range(N):
            type_selector = torch.zeros(N).bool()
            type_selector[i] = True
            subgraph_repeated_in_chunks = subgraphs_features[:, type_selector] # shape = batch x dim
            subgraph_repeated_in_chunks = subgraph_repeated_in_chunks.repeat_interleave(N-1, dim=0).reshape(self.batch_size, N-1, -1) # shape = batch x N-1 x 2*dim
            # 每个batch_size里返回的数据形式：
            #       e_type, e_type, ..., e_type
            subgraph_repeated_in_chunks = torch.cat([subgraph_repeated_in_chunks, subgraphs_features[:, ~type_selector]], dim=2) # shape = batch x N-1 x dim
            # print(all_combinations_matrix.size())
            # all_combinations_matrix 的shape = (batch_size, N-1, 2*features_dim)
            # 每个batch_size里返回的数据形式：
            #   e_type || e1
            #   e_type || e2
            #   ...
            #   e_type || e_(type - 1)
            #   e_type || e_(type+1)
            #   ...
            #   e_type || eN
            all_combinations_matrix.append(subgraph_repeated_in_chunks)
        all_combinations_matrix = torch.cat(all_combinations_matrix)

        global_read_out = self.graph_read_out(feat, subgraphs)
        global_read_out = global_read_out.repeat(all_combinations_matrix.shape[0] * all_combinations_matrix.shape[1]).reshape(all_combinations_matrix.shape[0],all_combinations_matrix.shape[1],global_read_out.shape[0])
        # shape = (batch_size, N-1, features_dim)
        all_combinations_matrix_with_global = torch.cat([all_combinations_matrix, global_read_out], dim=2) # shape = (batch_size, N-1, 3*features_dim)

        return all_combinations_matrix_with_global
    
    def graph_read_out(self, feat, subgraphs):
        if self.readout == 'max':
            read_out = torch.max(feat, dim = 0)[0]
                
        if self.readout == 'sum':
            read_out = torch.sum(feat, dim = 0)
        
        if self.readout == 'avg':
            read_out = torch.mean(feat, dim = 0)

        return read_out


class OPD_layer_v2(nn.Module):
    def __init__(self, ntype, out_dim, alpha, device, readout):
        super(OPD_layer_v2, self).__init__()
        self.ntype = ntype
        self.device = device
        self.out_dim = out_dim

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(0.5)
        self.fc_graph = nn.Linear(out_dim,out_dim)
        self.fc_node = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(self.ntype)])
        self.OPD_out_node = nn.Linear(out_dim, 1)
        self.OPD_out_graph = nn.Linear(out_dim, 1)
        self.readout = readout
        self.pooling = nn.MaxPool1d(kernel_size=out_dim)
    
    def graph_read_out(self, feats, perturbation):
        read_out = []
        if self.readout == 'max':
            for graph in perturbation:
                graph_embedd = feats[graph]
                read_out.append(torch.max(graph_embedd, dim = 0)[0])
            read_out = torch.stack(read_out)
                
        if self.readout == 'sum':
            graph_aggregation = torch.zeros((len(perturbation),feats.shape[0]), dtype = torch.float32)
            for graph_id in range(len(perturbation)):
                graph_aggregation[graph_id][perturbation[graph_id]] = 1
            read_out = torch.matmul(graph_aggregation.to(self.device), feats)
        
        if self.readout == 'avg':
            graph_aggregation = torch.zeros((len(perturbation),feats.shape[0]), dtype = torch.float32)
            for graph_id in range(len(perturbation)):
                graph_aggregation[graph_id][perturbation[graph_id]] = 1/self.ntype
            read_out = torch.matmul(graph_aggregation.to(self.device), feats)

        if self.readout == 'max_zero':
            for graph in perturbation:
                zero_id = graph==-1
                graph_embedd = feats[graph[~zero_id]]
                for _ in graph[zero_id]:
                    graph_embedd = torch.cat([graph_embedd,torch.zeros((1,graph_embedd.shape[1]),device = graph_embedd.device)])
                read_out.append(torch.max(graph_embedd, dim = 0)[0])
            read_out = torch.stack(read_out)
            
            pass
        
        
        return read_out
    
    def forward(self, feats, perturbation):
        
        read_out = self.graph_read_out(feats, perturbation)
        # read_out = self.dropout(read_out)
        read_out_graph = self.fc_graph(read_out)
        # read_out_graph = self.leakyrelu(read_out_graph)
        read_out_node = torch.stack([p(read_out_graph) for p in self.fc_node])
        # node_out_node = self.leakyrelu(node_out_node)
        read_out_node = read_out_node.transpose(0,1).reshape(-1,self.out_dim)
       
        logits_node = self.OPD_out_node(read_out_node)
        logits_graph = self.OPD_out_graph(read_out_graph)
        return logits_node,logits_graph 