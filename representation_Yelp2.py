'''

this script is for producing representations of all nodes.

'''
import time

# import argparse
from utils.data import load_Yelp2_data
from utils.tools_re import representations_generator
import torch
from model.ohnn_re import OHNN_v3
import numpy as np


lr = 0.0001 # 0.0001
weight_decay = 0.001
repeat = 5
num_epochs = 200
batch_size = 64 # 32 64
patience = 10
perturb_ratio = 0.2
nlayers = 3

g_readout = 'avg' # 'avg' # 'avg', 'max', 'sum', 'Breadth'
ohnn_agg = 'max' # 'max', 'avg'
opd_readout = 'max' # 'max', 'sum', 'avg'

jointly = True
gamma = 0 #1e-3 jointly scaler

hidden_dim = 256
out_dim = 64
dropout_rate = 0.5
alpha = 0.01 # leakyrelu slope
beta_1 = 0.1 # 0.3 skip-connection, ratio of previous epoch weight
beta_2 = 0.7
nheads_1 = 1 # intro subgraph
nheads_2 = 0 # inter subgraph

save_postfix = 'yelp2_opd_v2re_opdv3_3l1h_0228_t1540' # rpt 9

num_rpt = 9

data_path = r'./data/Yelp2_processed/graph_split'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


features, subgraphs,subgraphs_train_val_test, train_val_test_idx, labels, adj, type_mask, ontology_pairs = load_Yelp2_data(prefix = data_path)
ontology_path = None

subgraphs_train = subgraphs_train_val_test['subgraphs_train']
subgraphs_val = subgraphs_train_val_test['subgraphs_val']
subgraphs_test = subgraphs_train_val_test['subgraphs_test']

train_nodes = train_val_test_idx['train_nodes']
val_nodes = train_val_test_idx['val_nodes']
test_nodes = train_val_test_idx['test_nodes']

ntype_counts = torch.tensor([p.shape[0] for p in features])
in_dims = [feature.shape[1] for feature in features]
features = [torch.FloatTensor(feature).to(device) for feature in features]

t_all_start = time.time()

# net = OHNN_v2(out_dim, nheads_1, adj, ontology_path, ontology_pairs, in_dims, hidden_dim, dropout_rate, alpha, beta_1, device, att_readout, ohnn_readout, opd_readout)
net = OHNN_v3(ontology_path, ontology_pairs, in_dims, hidden_dim, out_dim, nheads_1, dropout_rate, alpha, beta_1, device, nlayers, g_readout, ohnn_agg, opd_readout)
net.to(device)

net.load_state_dict(torch.load(r'./ckpt/checkpoint_{}_{}.pt'.format(save_postfix, num_rpt)))

def representations_generating_v2(net, subgraphs, features, type_mask, device, save_flag, max_size=10000, task = 'test'):
    # from utils.tools import representations_generator
    print('——__——__——__——__——__——__——__——__——__——__')
    print('generating representations')
    t_g = time.time()
    repre_g = representations_generator(subgraphs, features, type_mask, device, max_size=len(subgraphs)) # 2048
    net.eval()
    with torch.no_grad():
        while repre_g.node_left() > 0:
            batch, batch_new, batch_features, feature_idx = repre_g.next()
            
            batch_representations, y_pred, y_pred_graph = net((batch_features, batch_new, None))
            
            if repre_g.num_iter() == 0:
                node_representations = batch_representations[feature_idx[0]:feature_idx[1]]
            else: node_representations = torch.cat((node_representations,batch_representations[feature_idx[0]:feature_idx[1]]),dim=0)
    print('finish!')
    if save_flag:
        print('saving...')           
        np.save(data_path + '/representations/'+ task + '_node_representations.npy', node_representations.cpu().numpy())
    print('done! time costs:', time.time() - t_g)
    return node_representations.cpu().numpy()

if __name__ == '__main__':
    node_representations = representations_generating_v2(net, subgraphs, features, type_mask, device, save_flag = True, max_size=10000, task = 'all_opdv2')
    # node_representations = np.load(data_path + '/representations/' + 'all_opdv2' + '_node_representations.npy') # opd_v2_88

    