import torch
import numpy as np
import sys



class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log .flush()
    def flush(self):
	    pass
 


class batch_generator:
    def __init__(self, graphs, features, device, batch_size=32, shuffle=True): 
        if graphs is not None:
            self.num_data = len(graphs) 
            self.graphs = graphs
            self.batches = np.copy(graphs)
            self.features = features
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        self.device = device
        if shuffle:
            np.random.shuffle(self.batches)
        self.batches = torch.tensor(self.batches, dtype = torch.int64, device=device)   
        
        ntype_counts = np.array([p.shape[0] for p in self.features], dtype=np.int64)
        prefix_operator = np.ones((len(ntype_counts), len(ntype_counts)))
        prefix_operator = np.tril(prefix_operator, k=-1)   
        self.feat_id_prefix = prefix_operator.dot(ntype_counts).astype(int)

    def next(self):            
        if self.num_iterators_left() <= 0:
            self.reset()
        self.iter_counter += 1                  
        batch = self.batches[(self.iter_counter - 1) * self.batch_size : self.iter_counter * self.batch_size]        
        batch_features = [self.features[i][(torch.unique(batch[:, i]) - self.feat_id_prefix[i])] for i in range(len(self.features))]         
        
        nodes = torch.unique(batch)
        batch_new = batch.clone()
        for node_ix in range(len(nodes)):
            batch_new[batch == nodes[node_ix]] = node_ix # assign new ids of batch
        return batch_new, batch_features, batch       

    def num_iterations(self):   
        return int(np.floor(self.num_data / self.batch_size))  

    def num_iterators_left(self):      
        return self.num_iterations() - self.iter_counter

    def reset(self):       
        if self.shuffle:
            self.batches = np.copy(self.graphs)
            np.random.shuffle(self.batches)
            self.batches = torch.tensor(self.batches, dtype = torch.int64, device = self.device)
        self.iter_counter = 0
        
        

class representations_generator:
    def __init__(self, graphs, features, type_mask, device, max_size=2048):    
        if graphs is not None:
            self.graphs = graphs
            self.features = features
        self.type_mask = type_mask
        self.batch_tmp = []
        self.ntype = len(features)
        self.all_nodes = np.array(np.unique(graphs))
        self.node_idx = 0 
        self.iter_counter = -1 
        self.max_size = max_size
        self.device = device
        self.graphs = torch.tensor(self.graphs, dtype = torch.int64, device=device, requires_grad = False)   
        
        self.ntype_counts = np.array([p.shape[0] for p in self.features], dtype=np.int64) # 每个种类有多少节点
        prefix_operator = np.ones((len(self.ntype_counts), len(self.ntype_counts)))
        prefix_operator = np.tril(prefix_operator, k=-1)   # 下三角矩阵
        self.feat_id_prefix = prefix_operator.dot(self.ntype_counts).astype(int) # 每个种类开始的序号

    def next(self):             # 下一个迭代的batch数据
        batch = []
        node_idx_start = self.node_idx # 记录该batch开始的节点
        while len(batch)+len(self.batch_tmp) < self.max_size:
            if len(batch) == 0:
                if len(self.batch_tmp) == 0:
                    batch = self.graphs[torch.where(self.graphs[:,self.type_mask[self.all_nodes[self.node_idx]]] == self.all_nodes[self.node_idx])]
                else:
                    batch = self.batch_tmp.clone()
            else:
                batch = torch.cat((batch,self.batch_tmp),dim=0)
            self.node_idx += 1
            if self.node_idx >= len(self.all_nodes):
                break
            self.batch_tmp = self.graphs[torch.where(self.graphs[:,self.type_mask[self.all_nodes[self.node_idx]]] == self.all_nodes[self.node_idx])]
            
        if len(batch) == 0:
            batch = self.batch_tmp.clone() # 防止某些时候len(self.batch_tmp)大于max_size导致生成中断
            self.node_idx += 1
            if self.node_idx < len(self.all_nodes): # 手动迭代一次
                self.batch_tmp = self.graphs[torch.where(self.graphs[:,self.type_mask[self.all_nodes[self.node_idx]]] == self.all_nodes[self.node_idx])]

        if self.node_idx == 26126:
            a=[]
        if self.node_idx >= 26125:
            a = [1,2,3]
            
        batch_features = [self.features[i][(torch.unique(batch[:, i]) - self.feat_id_prefix[i])] for i in range(self.ntype)]  # 迭代种类筛选对应的原始特征
        nodes = torch.unique(batch)
        batch_new = batch.clone()
        for node_ix in range(len(nodes)):
            batch_new[batch == nodes[node_ix]] = node_ix # assign new ids of batch

        node_idx_start = int((nodes == self.all_nodes[node_idx_start]).nonzero()) # 记录该batch开始的节点在batch_new的id
        node_idx_end = int((nodes == self.all_nodes[self.node_idx-1]).nonzero()) # 记录该batch结束的节点在batch_new的id
        
        feature_idx = (node_idx_start, node_idx_end + 1) # 记录该batch开始的节点 与 结束的节点+1，作为保留feature的索引
        
        self.iter_counter += 1
        return batch, batch_new, batch_features, feature_idx # 返回batch数据及其特征，以及输入模型后要保留的特征索引，feature[feature_idx[0]:feature_idx[1],:]

    def node_left(self):       # 剩余节点数
        return len(self.all_nodes) - self.node_idx
    
    def num_iter(self):
        return self.iter_counter