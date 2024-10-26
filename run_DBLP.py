import time, sys
import argparse
from utils.data import load_DBLP_data2
from utils.tools_re import batch_generator, Logger
from utils.pytorchtools_re import EarlyStopping, FocalLoss, Schema_perturbation, link_prediction
import torch
import torch.nn.functional as F
from model.msgnn_re import MSGNN, MSGNN_wosub
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, auc, precision_recall_curve

weight_decay = 0.001
alpha = 0.01 # leaky slope
beta = 0.1 # skip-connection
gamma = 0.8454 # trade off
batch_size = 64
perturb_ratio = 0.6 # mask ratio
save_postfix = 'checkpoint'
sys.stdout = Logger(r'./train_log/' + save_postfix +'.log', sys.stdout)
data_path = r'./data/DBLP_processed'
features, subgraphs, subgraphs_train_val_test, _, adj, __ = load_DBLP_data2(data_path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
subgraphs_train = subgraphs_train_val_test['subgraphs_train']
subgraphs_val = subgraphs_train_val_test['subgraphs_val']
subgraphs_test = subgraphs_train_val_test['subgraphs_test']
in_dims = [feature.shape[1] for feature in features]
features = [torch.FloatTensor(feature).to(device) for feature in features]
ntype = len(in_dims)

def run(lr, repeat, num_epochs, patience, nlayers, dropout_rate,
                masking_mode, hidden_dim, out_dim, num_heads, ablation):
    
    if masking_mode == 'zero':
        graph_readout = 'max_zero'
    else: graph_readout = 'max'
    if ablation == 'graph':
        gamma2 = 0
    else: gamma2 = 1
    if ablation == 'node':
        gamma = 0
    else: gamma = globals()['gamma']
    if ablation == 'instance':
        model = globals()['MSGNN_wosub']
    else:
        model = globals()['MSGNN']

    auc_list = []
    pr_auc_list = []
    f1_list = []

    t_all_start = time.time()
    FLloss = FocalLoss(alpha=0.64)
    # print('postfix:', save_postfix)
    for rpt in range(repeat):
        rpt_start = time.time()
        net = model(ntype, in_dims, hidden_dim, out_dim, num_heads, dropout_rate, alpha, beta, device, nlayers, graph_readout, ablation)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=r'./ckpt/{}_{}.pt'.format(save_postfix, rpt))
        dur1_opd, dur2_opd = [], []
        batch_g = batch_generator(subgraphs_train, features, device, batch_size)
        val_g = batch_generator(subgraphs_val, features, device, 2048, shuffle=False)
        sp = Schema_perturbation(subgraphs, perturb_ratio, device, mode = masking_mode)
        
        for epoch in range(num_epochs):
            t_start = time.time()
            ## training
            net.train()  
            for iteration in range(batch_g.num_iterations()):
                # forward
                t0 = time.time()
                batch, batch_features, _ = batch_g.next()
                perturbation, node_label, label = sp.get_perturbation(batch)
                _, y_pred,y_pred_graph = net((batch_features, batch, perturbation))

                label = torch.from_numpy(label).float().to(device)
                node_label = node_label.float().to(device)
                node_label = node_label.reshape(-1,1)
                y_pred = y_pred.reshape(-1,ntype).reshape(-1,1)
                OPD_loss = gamma * FLloss(y_pred, node_label) + gamma2 * F.binary_cross_entropy_with_logits(y_pred_graph, label)

                t1 = time.time()
                dur1_opd.append(t1 - t0)
                # update
                optimizer.zero_grad()
                OPD_loss.backward()
                optimizer.step()
                t2 = time.time()
                dur2_opd.append(t2 - t1)
                # print training info
                if iteration % 50 == 0:
                    print(
                        'Epoch {:03d} | Iteration {:04d} | Train_Loss {:.4f} | Time1(s) {:.4f}/{:.3f} | Time2(s) {:.4f}/{:.3f}'.format(
                            epoch, iteration, OPD_loss.item(), np.mean(dur1_opd), np.mean(dur1_opd)*50, np.mean(dur2_opd),np.mean(dur2_opd)*50))
            # eval
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_g.num_iterations()):
                    # forward
                    batch, batch_features, _ = val_g.next()
                    perturbation, node_label, label = sp.get_perturbation(batch)
                    _, y_pred,y_pred_graph = net((batch_features, batch, perturbation))

                    t1 = time.time()
                    dur1_opd.append(t1 - t0)
                    label = torch.from_numpy(label).float().to(device)
                    node_label = node_label.float().to(device)
                    zero_rows=torch.all(node_label==0,dim=1).nonzero().squeeze()
                    node_label = node_label[~zero_rows].reshape(-1,1)
                    y_pred = y_pred.reshape(-1,ntype)[~zero_rows].reshape(-1,1)
                    OPD_loss = gamma * FLloss(y_pred, node_label) + gamma2 * F.binary_cross_entropy_with_logits(y_pred_graph, label)

                    val_loss.append(OPD_loss.item()) 

            t_end = time.time()
            # print validation info
            print('Epoch {:03d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, np.mean(val_loss), t_end - t_start))   
            early_stopping(np.mean(val_loss), net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        print('epoch iter ends, test starts!')
        t_test = time.time()
        test_g = batch_generator(subgraphs_train, features, device, 2048, shuffle = False)
        net.load_state_dict(torch.load(r'./ckpt/{}_{}.pt'.format(save_postfix, rpt)))
        net.eval()
        auc_batch_list = []
        f1_batch_list = []
        pr_auc_batch_list = []
        with torch.no_grad():
            for iteration in range(test_g.num_iterations()):
                # forward
                batch, batch_features, batch_adj = test_g.next()
                features_test, _,__ = net((batch_features, batch, None))
                pos_out, neg_out, y_true_test = link_prediction(batch_adj.cpu(), adj, features_test)
                pos_proba = torch.sigmoid(pos_out.flatten())
                neg_proba = torch.sigmoid(neg_out.flatten())
                
                y_proba_test = torch.cat([pos_proba, neg_proba])
                y_proba_test = y_proba_test.cpu()
                y_proba_test_label = torch.where(y_proba_test > sorted(y_proba_test)[len(y_proba_test)//2], torch.ones_like(y_proba_test), torch.zeros_like(y_proba_test)).int()
                auc_score = roc_auc_score(y_true_test, y_proba_test)
                f1 = f1_score(y_true_test, y_proba_test_label)
                precision, recall, _ = precision_recall_curve(y_true_test, y_proba_test)
                pr_auc = auc(recall, precision)
            
                auc_batch_list.append(auc_score)
                f1_batch_list.append(f1)
                pr_auc_batch_list.append(pr_auc)
                
            dur_test = time.time() - t_test    
        
        print('')
        print('Schema Masking Learning')
        print('AUC = {}'.format(np.mean(auc_batch_list)))
        print('PR_AUC = {}'.format(np.mean(pr_auc_batch_list)))
        print('F1 = {}'.format(np.mean(f1_batch_list)))
        print('test costs {:.4f} seconds'.format(dur_test))
        print('rpt costs {:.4f} mins'.format((time.time()-rpt_start)/60))
        print('')
        auc_list.append(np.mean(auc_batch_list))
        f1_list.append(np.mean(f1_batch_list))
        pr_auc_list.append(np.mean(pr_auc_batch_list))
    print('----------------------------------------------------------------')
    print('Schema Masking Learning Summary')
    print('total time costs {:.2f} hours'.format((time.time()-t_all_start)/3600))
    print('AUC_mean = {:.5f}, AUC_std = {:.5f}'.format(np.mean(auc_list), np.std(auc_list)))
    print('PR_AUC_mean = {:.5f}, AP_std = {:.5f}'.format(np.mean(pr_auc_list), np.std(pr_auc_list)))
    print('F1_mean = {:.5f}, AP_std = {:.5f}'.format(np.mean(f1_list), np.std(f1_list)))
    print('')
    print('AUC history = {}'.format(auc_list))
    print('PR_AUC history = {}'.format(pr_auc_list))
    print('F1 history = {}'.format(f1_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MSGNN demo for Available Check')
    
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--repeat', type=int, default=10, help='Repeat the training and testing for N times. Default is 10.')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs. Default is 200.')
    parser.add_argument('--patience', type=int, default=10, help='Patience. Default is 10.')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers.')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--masking-mode', default='default', choices=['default', 'random', 'zero'], help='Schema masking strategy.')
    parser.add_argument('--hidden-dim', type=int, default=1024, help='Dimension of the node hidden state. Default is 1024.')
    parser.add_argument('--out-dim', type=int, default=256, help='Dimension of the output. Default is 256.')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of the attention heads. Default is 4.')
    parser.add_argument('--ablation', default='None', choices=['None', 'graph', 'node', 'instance', 'global'], help='Ablation param')
    
    args = parser.parse_args()
    
    run(args.learning_rate, args.repeat, args.epoch, args.patience, args.nlayers, args.dropout,
                args.masking_mode, args.hidden_dim, args.out_dim, args.num_heads, args.ablation)