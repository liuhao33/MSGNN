import time
import argparse
from utils.data import load_DBLP_data2
from utils.tools_re import representations_generator
import torch
from model.msgnn_re import MSGNN
import numpy as np
weight_decay = 0.001
alpha = 0.01 # leaky slope
beta = 0.1 # skip-connection
data_path = r'./data/DBLP_processed'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
features, subgraphs, ____, __, ___, type_mask = load_DBLP_data2(prefix = data_path)
in_dims = [feature.shape[1] for feature in features]
features = [torch.FloatTensor(feature).to(device) for feature in features]
ntype = len(in_dims)

def run(nlayers, dropout_rate, hidden_dim, out_dim, num_heads, load_postfix, save_flag, max_size):
    net = MSGNN(ntype, in_dims, hidden_dim, out_dim, num_heads, dropout_rate, alpha, beta, device, nlayers)
    net.to(device)
    net.load_state_dict(torch.load(r'./ckpt/checkpoint_{}.pt'.format(load_postfix)))
    node_representations = representations_generating_v2(net, subgraphs, features, type_mask, device, save_flag, max_size, load_postfix)

def representations_generating_v2(net, subgraphs, features, type_mask, device, save_flag = True, max_size=10000, name = 'dblp'):
    # from utils.tools import representations_generator
    print('generating representations')
    t_g = time.time()
    repre_g = representations_generator(subgraphs, features, type_mask, device, max_size=max_size) # 2048
    net.eval()
    with torch.no_grad():
        while repre_g.node_left() > 0:
            _, batch_new, batch_features, feature_idx = repre_g.next()
            
            batch_representations, __, ___ = net((batch_features, batch_new, None))
            
            if repre_g.num_iter() == 0:
                node_representations = batch_representations[feature_idx[0]:feature_idx[1]]
            else: node_representations = torch.cat((node_representations,batch_representations[feature_idx[0]:feature_idx[1]]),dim=0)
    print('finish!')
    if save_flag:
        print('saving...')           
        np.save(data_path + '/representations/'+ name + '_node_representations.npy', node_representations.cpu().numpy())
    print('done! time costs:', time.time() - t_g)
    return node_representations.cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MSGNN demo for Available Check')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers.')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--hidden-dim', type=int, default=1024, help='Dimension of the node hidden state. Default is 1024.')
    parser.add_argument('--out-dim', type=int, default=256, help='Dimension of the output. Default is 256.')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of the attention heads. Default is 4.')
    parser.add_argument('--load-postfix', default='dblp', help='Postfix for the saved model. Default is dblp.')
    
    parser.add_argument('--save', action="store_true", help='Save embeddings.')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size.')
    
    args = parser.parse_args()
    
    run(args.nlayers, args.dropout, args.hidden_dim, args.out_dim, args.num_heads, args.load_postfix, args.save, args.batch_size)