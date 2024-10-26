import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn  
import torch.nn.functional as F
import re
import random
import itertools


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.auc = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model) 
        elif score <= self.best_score - self.delta: 
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model) 
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving checkpoint ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss



class FocalLoss(nn.Module):  
    def __init__(self, alpha=0.8, gamma=2.0):  
        super(FocalLoss, self).__init__()  
        self.alpha = alpha  
        self.gamma = gamma  
  
    def forward(self, inputs, targets):  
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class Schema_perturbation(object):
    def __init__(self, subgraphs_all, ratio, device, mode = 'default'):
        super(Schema_perturbation, self).__init__()
        self.N_neg = None
        self.all_nodes = np.unique(subgraphs_all)
        self.ntype = subgraphs_all.shape[1]
        self.subgraphs = subgraphs_all
        self.device = device
        self.ratio = ratio if ratio < 0.3 else 0.3
        self.N_per = max(int(self.ntype * ratio), 1)  # N_per != 0
        self.N_neg = None
        self.N_candidates_len = 4
        self.mode = mode

        self.unique_nodes_batch = []
    
    def perturbation(self, batch_graph):
        perturbation_candidates = []
        cnt = self.N_per
        if self.mode == 'zero':
            pert_idx = []    
            while cnt > 0 :
                cols = list(range(self.ntype))
                comb = list(itertools.combinations(cols, cnt))
                pert_idx += comb
                cnt -=1 
            clean_candidates = [] 
            for pert in pert_idx:
                subgraph_pert = batch_graph.clone()
                for col in pert:
                    col_pert = torch.Tensor([-1]*subgraph_pert.shape[0])
                    subgraph_pert[:,col] = col_pert 
                perturbation_candidates.append(subgraph_pert) 
                clean_candidates.append(batch_graph)
            perturbation_candidates = torch.cat(perturbation_candidates)
            clean_candidates = torch.cat(clean_candidates)
            shuffle_idx = torch.randperm(perturbation_candidates.shape[0])        
            perturbation_candidates = perturbation_candidates[shuffle_idx]
            clean_candidates = clean_candidates[shuffle_idx]    
        else:
            if self.mode == 'random':
                nodes = [torch.unique(batch_graph) for _ in range(self.ntype)]
            if self.mode == 'default':
                nodes = [torch.unique(batch_graph[:,col]) for col in range(self.ntype)]      
            pert_idx = []    
            while cnt > 0 :
                cols = list(range(self.ntype))
                comb = list(itertools.combinations(cols, cnt))
                pert_idx += comb
                cnt -=1   
            
            clean_candidates = [] 
            for pert in pert_idx:
                subgraph_pert = batch_graph.clone()
                for col in pert:
                    prob = torch.Tensor([1/len(nodes[col])]*len(nodes[col]))
                    col_pert = nodes[col][torch.multinomial(prob, self.N_neg, replacement=True)]
                    subgraph_pert[:,col] = col_pert 
                perturbation_candidates.append(subgraph_pert) 
                clean_candidates.append(batch_graph)
            perturbation_candidates = torch.cat(perturbation_candidates)
            clean_candidates = torch.cat(clean_candidates)
            shuffle_idx = torch.randperm(perturbation_candidates.shape[0])        
            perturbation_candidates = perturbation_candidates[shuffle_idx]
            clean_candidates = clean_candidates[shuffle_idx]
            
        return perturbation_candidates,clean_candidates


    def get_perturbation(self, batch_graph):
        self.N_neg = len(batch_graph)
        perturbation = [batch_graph]
        clean=[batch_graph]
        while len(perturbation[0]) < self.N_neg * 2:
            candidates,clean_candidates = self.perturbation(batch_graph)
            candidates_eq_batch_graph = torch.eq(candidates[:,None,:], batch_graph).all(dim=2)
            candidates_eq_batch_graph = torch.sum(candidates_eq_batch_graph,dim=1).bool()
            candidates = candidates[~candidates_eq_batch_graph]
            clean_candidates = clean_candidates[~candidates_eq_batch_graph]
            perturbation.append(candidates)
            clean.append(clean_candidates)
            perturbation = [torch.cat((perturbation[0], perturbation[1]), dim = 0)]
            clean = [torch.cat((clean[0], clean[1]), dim = 0)]
        
        perturbation = perturbation[0][:self.N_neg*2]
        clean = clean[0][:self.N_neg*2]
        label = np.array([1] * self.N_neg + [0] * self.N_neg)
        
        shuffle_idx = torch.randperm(perturbation.shape[0])
        perturbation = perturbation[shuffle_idx]
        clean = clean[shuffle_idx]
        label = label[shuffle_idx]
        node_label = (perturbation!=clean).byte()

        return perturbation, node_label,label
    

    def perturbation2(self, batch_graph):
        perturbation_candidates = []
        node_label_candidates = []         
        col_dic = {}
        cnt = self.N_per
        if self.mode == 'easy':
            nodes = [torch.unique(batch_graph) for _ in range(self.ntype)]
        if self.mode == 'normal':
            nodes = [torch.unique(batch_graph[:,col]) for col in range(self.ntype)]
        
        pert_idx = []    
        while cnt > 0 :
            cols = list(range(self.ntype))
            comb = list(itertools.combinations(cols, cnt))
            pert_idx += comb
            cnt -=1

        for col in range(self.ntype):
            prob = torch.Tensor([1/len(nodes[col])]*len(nodes[col]))
            col_pert = nodes[col][torch.multinomial(prob, self.N_candidates_len*self.N_neg, replacement=True)]
            col_dic[col] = col_pert
        
        for pert in pert_idx:
            subgraph_pert = batch_graph.clone()
            pert_pattern = torch.zeros(self.ntype, device=self.device).reshape(1,-1)
            for col in pert:
                subgraph_pert[:,col] = col_dic[col][torch.randperm(col_dic[col].shape[0])]
                pert_pattern[:,col] = 1
            subgraph_pert = subgraph_pert.unique(dim = 0)
            perturbation_candidates.append(subgraph_pert)
            node_label_candidates.append(torch.cat([pert_pattern]*subgraph_pert.shape[0]))

        perturbation_candidates = torch.cat(perturbation_candidates)
        node_label_candidates = torch.cat(node_label_candidates)
        
        shuffle_idx = torch.randperm(perturbation_candidates.shape[0])
        perturbation_candidates = perturbation_candidates[shuffle_idx]
        node_label_candidates = node_label_candidates[shuffle_idx]
        
        return perturbation_candidates, node_label_candidates
    

    def get_perturbation2(self, batch_graph):
        self.N_neg = len(batch_graph)
        perturbation = [batch_graph]
        # clean=[batch_graph]
        node_label = [torch.zeros_like(batch_graph)]
        batch_graph_extend = torch.cat([batch_graph]*self.N_candidates_len)
        while len(perturbation[0]) < 2 * self.N_neg:
            candidates, node_label_candidates = self.perturbation2(batch_graph_extend)
            candidates_eq_batch_graph = torch.eq(candidates[:,None,:], batch_graph_extend).all(dim=2)
            candidates_eq_batch_graph = torch.sum(candidates_eq_batch_graph,dim=1).bool()
            candidates = candidates[~candidates_eq_batch_graph] # remove pos sample
            node_label_candidates = node_label_candidates[~candidates_eq_batch_graph]

            perturbation.append(candidates)
            node_label.append(node_label_candidates)
            perturbation = [torch.cat((perturbation[0], perturbation[1]), dim = 0)]
            node_label = [torch.cat((node_label[0], node_label[1]), dim = 0)]
        
        perturbation = perturbation[0][:self.N_neg*2]
        node_label = node_label[0][:self.N_neg*2]
        label = np.array([1] * self.N_neg + [0] * self.N_neg)
        
        shuffle_idx = torch.randperm(perturbation.shape[0])
        perturbation = perturbation[shuffle_idx]
        node_label = node_label[shuffle_idx]
        label = label[shuffle_idx]

        return perturbation, node_label, label


def sparse_transform(sparse_matrix, device):
    indices = sparse_matrix.nonzero()
    values = sparse_matrix[indices].tolist()[0]
    size = sparse_matrix.shape
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size, device=device)
    return sparse_tensor
    
    
def link_prediction_sp(test_subgraph, adjM, embedding): 
    nodes = np.unique(test_subgraph)
    mask = sp.lil_matrix(adjM.shape, dtype=bool) # lil_matrix or csr_matrix depends on tasks
    mask[np.ix_(nodes,nodes)] = True
    adjM = adjM.multiply(mask.tocsr()) 
    node2index = {n:i for i,n in enumerate(nodes)}
    
    pos_data = []
    neg_candidates = []
    
    for head in nodes:
        pos_tail_set = np.array(sp.find(adjM[head]))[1]
        pos_data = pos_data + [[node2index[head], node2index[p]] for p in pos_tail_set]

        pos_tail_id = np.ones_like(nodes,dtype=bool)
        pos_tail_id[[node2index[p] for p in pos_tail_set]] = False

        neg_tail_set = nodes[pos_tail_id]
        neg_candidates = neg_candidates + [[node2index[head], node2index[p]] for p in neg_tail_set]
    
    pos_data = np.array(pos_data)
    neg_sample_num = len(pos_data)
    idx = np.random.choice(len(neg_candidates), neg_sample_num, replace=False)
    neg_data = np.array([neg_candidates[p] for p in idx])

    y_true_label = np.array([1] * len(pos_data) + [0] * len(neg_data))

    pos_embedding_head = embedding[pos_data[:,0]]
    pos_embedding_head = pos_embedding_head.reshape(-1, 1, embedding.shape[1])
    pos_embedding_tail = embedding[pos_data[:,1]]
    pos_embedding_tail = pos_embedding_tail.reshape(-1, embedding.shape[1], 1)
    pos_out = torch.bmm(pos_embedding_head, pos_embedding_tail)

    neg_embedding_head = embedding[neg_data[:,0]]
    neg_embedding_head = neg_embedding_head.reshape(-1, 1, embedding.shape[1])
    neg_embedding_tail = embedding[neg_data[:,1]]
    neg_embedding_tail = neg_embedding_tail.reshape(-1, embedding.shape[1], 1)
    neg_out = torch.bmm(neg_embedding_head, neg_embedding_tail)

    return pos_out, neg_out, y_true_label


def link_prediction(test_subgraph, adjM, embedding):   
    adjM = adjM.A
    nodes = np.unique(test_subgraph)
    mask = np.zeros_like(adjM, dtype=bool)
    mask[np.ix_(nodes,nodes)] = True
    adjM = np.logical_and(adjM, mask).astype('int') 
    node2index = {n:i for i,n in enumerate(nodes)}
    
    pos_data = []
    pos_data_head = []
    pos_data_tail = []
    
    neg_candidates = []
    neg_candidates_head = []
    neg_candidates_tail = []
    
    for head in nodes:
        for tail in nodes:
            if adjM[head][tail] == 1:
                pos_data.append([head, tail])
                pos_data_head.append(node2index[head])
                pos_data_tail.append(node2index[tail])
            elif adjM[head][tail] == 0:
                neg_candidates.append([node2index[head], node2index[tail]])
    
    pos_data = np.array(pos_data)
    neg_sample_num = len(pos_data)     
    idx = np.random.choice(len(neg_candidates), int(neg_sample_num), replace=False)
    neg_data = [neg_candidates[p] for p in sorted(idx)]
    neg_data = np.array(neg_data)
    neg_candidates_head = [p[0] for p in neg_data]
    neg_candidates_tail = [p[1] for p in neg_data]

    y_true_label = np.array([1] * len(pos_data) + [0] * len(neg_data))

    pos_embedding_head = embedding[pos_data_head]
    pos_embedding_head = pos_embedding_head.view(-1, 1, embedding.shape[1])
    pos_embedding_tail = embedding[pos_data_tail]
    pos_embedding_tail = pos_embedding_tail.view(-1, embedding.shape[1], 1)
    pos_out = torch.bmm(pos_embedding_head, pos_embedding_tail)

    neg_embedding_head = embedding[neg_candidates_head]
    neg_embedding_head = neg_embedding_head.view(-1, 1, embedding.shape[1])
    neg_embedding_tail = embedding[neg_candidates_tail]
    neg_embedding_tail = neg_embedding_tail.view(-1, embedding.shape[1], 1)
    neg_out = torch.bmm(neg_embedding_head, neg_embedding_tail)

    return pos_out, neg_out, y_true_label

