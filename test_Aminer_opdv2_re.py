import time, sys
from utils.data import load_AminerS_data
from utils.tools_re import batch_generator, Logger
from utils.pytorchtools_re import EarlyStopping, FocalLoss, Ontology_perturbation, link_prediction_by_graph2, link_prediction_by_graph
import torch
import torch.nn.functional as F
from model.ohnn_re import OHNN_v3
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, auc, precision_recall_curve

lr = 0.0003 
weight_decay = 0.001
repeat = 1
num_epochs = 200
batch_size = 64 # 32
patience = 10
perturb_ratio = 0.3
nlayers = 3 # 3 # 1

# jointly = True
gamma = 2 # 0.8454

g_readout = 'avg'
ohnn_agg = 'max'
opd_readout = 'max' # 'max'
op_mode = 'normal' #'normal'

out_dim = 256 # 192
hidden_dim = 1024
dropout_rate = 0.5 # 0.36
alpha = 0.01 # leaky slope
beta_1 = 0.1 # 0.3
beta_2 = 0 
nheads_1 = 2 # 2 # 1
nheads_2 = 0

save_postfix = 'msgnn_aminer'
sys.stdout = Logger(r'./train_log/' + save_postfix +'.log', sys.stdout)
data_path = r'./data/Aminer2_processed/graph_split'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

features, subgraphs,subgraphs_train_val_test, train_val_test_idx, labels, adj, type_mask = load_AminerS_data(prefix = data_path)

subgraphs_train = subgraphs_train_val_test['subgraphs_train']
subgraphs_val = subgraphs_train_val_test['subgraphs_val']
subgraphs_test = subgraphs_train_val_test['subgraphs_test']

train_nodes = train_val_test_idx['train_nodes']
val_nodes = train_val_test_idx['val_nodes']
test_nodes = train_val_test_idx['test_nodes']

in_dims = [feature.shape[1] for feature in features]
features = [torch.FloatTensor(feature).to(device) for feature in features]
ntype = len(in_dims)

auc_lp_list = []
pr_auc_lp_list = []
f1_lp_list = []


t_all_start = time.time()

FLloss = FocalLoss(alpha=0.64)
print('name:', save_postfix)
for rpt in range(repeat):
    rpt_start = time.time()
    net = OHNN_v3(ntype, in_dims, hidden_dim, out_dim, nheads_1, dropout_rate, alpha, beta_1, device, nlayers, g_readout, ohnn_agg, opd_readout)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=r'./ckpt/checkpoint_{}_{}.pt'.format(save_postfix, rpt))

    dur1_opd = []
    dur2_opd = []
    dur3_opd = []
    
    batch_g = batch_generator(subgraphs_train, features, device, batch_size)
    val_g = batch_generator(subgraphs_val, features, device, 1024, shuffle=False)
    op = Ontology_perturbation(subgraphs, perturb_ratio, device, mode = op_mode)
    
    for epoch in range(num_epochs):
        t_start = time.time()
        # training
        net.train()  
        for iteration in range(batch_g.num_iterations()):
            # forward
            t0 = time.time()
            batch, batch_features, _ = batch_g.next()
            perturbation, node_label, label = op.get_perturbation(batch)
            features_train, y_pred,y_pred_graph = net((batch_features, batch, perturbation))

            t1 = time.time()
            dur1_opd.append(t1 - t0)
            label = torch.from_numpy(label).float().to(device)
            node_label = node_label.float().to(device)
            node_label = node_label.reshape(-1,1)
            y_pred = y_pred.reshape(-1,ntype).reshape(-1,1)
            OPD_loss = gamma * FLloss(y_pred, node_label) + F.binary_cross_entropy_with_logits(y_pred_graph, label)


            t2 = time.time()
            dur2_opd.append(t2 - t1)
            
            # autograd
            optimizer.zero_grad()

            OPD_loss.backward()
            
            optimizer.step()

            t3 = time.time()
            dur3_opd.append(t3 - t2)

            # print training info
            if iteration % 50 == 0:
                print(
                    'Epoch {:03d} | Iteration {:04d} | Train_Loss {:.4f} | Time1(s) {:.4f}/{:.3f} | Time2(s) {:.4f}/{:.3f} | Time3(s) {:.4f}/{:.3f} | AUC node {:.4f} | AP node {:.4f}'.format(
                        epoch, iteration, OPD_loss.item(), np.mean(dur1_opd), np.mean(dur1_opd)*50, np.mean(dur2_opd),np.mean(dur2_opd)*50, np.mean(dur3_opd),np.mean(dur3_opd)*50,
                        roc_auc_score(node_label.detach().cpu(), y_pred.detach().cpu()), average_precision_score(node_label.detach().cpu(), y_pred.detach().cpu())))

        # eval
        net.eval()
        val_loss = []
        auc_opd = []
        ap_opd = []
        auc_opd_node = []
        ap_opd_node = []

        with torch.no_grad():
            for iteration in range(val_g.num_iterations()):
                # forward
                batch, batch_features, _ = val_g.next()
                perturbation, node_label, label = op.get_perturbation(batch)
                features_train, y_pred,y_pred_graph = net((batch_features, batch, perturbation))

                t1 = time.time()
                dur1_opd.append(t1 - t0)
                label = torch.from_numpy(label).float().to(device)
                node_label = node_label.float().to(device)
                zero_rows=torch.all(node_label==0,dim=1).nonzero().squeeze()
                node_label = node_label[~zero_rows].reshape(-1,1)
                y_pred = y_pred.reshape(-1,ntype)[~zero_rows].reshape(-1,1)
                OPD_loss = gamma * FLloss(y_pred, node_label) + F.binary_cross_entropy_with_logits(y_pred_graph, label)

                val_loss.append(OPD_loss.item()) 

                auc_opd.append(roc_auc_score(label.cpu(), y_pred_graph.cpu()))
                ap_opd.append(average_precision_score(label.cpu(), y_pred_graph.cpu()))
                auc_opd_node.append(roc_auc_score(node_label.cpu(), y_pred.cpu()))
                ap_opd_node.append(average_precision_score(node_label.cpu(), y_pred.cpu()))
        
        t_end = time.time()
        # print validation info

        print('Epoch {:03d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            epoch, np.mean(val_loss), t_end - t_start))
        print('AUC_opd = {}'.format(np.mean(auc_opd)))
        print('AP_opd = {}'.format(np.mean(ap_opd)))
        print('AUC_opd_node = {}'.format(np.mean(auc_opd_node)))
        print('AP_opd_node = {}'.format(np.mean(ap_opd_node)))           
        print('lr = {}, weight_decay = {}, batch_size = {}'.format(lr, weight_decay, batch_size))
        early_stopping(np.mean(val_loss), net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    print('——__——__——__——__——__——__——__——__——__——__')
    print('epoch iter ends, test starts!')
    t_test = time.time()
    test_g = batch_generator(subgraphs_train, features, device, 1024, shuffle = False)

    net.load_state_dict(torch.load(r'./ckpt/checkpoint_{}_{}.pt'.format(save_postfix, rpt)))

    net.eval()

    auc_lp_batch_list = []
    f1_lp_batch_list = []
    pr_auc_lp_batch_list = []
    
    OPD_loss_list = []
    
    with torch.no_grad():
        for iteration in range(test_g.num_iterations()):
            # forward
            batch, batch_features, batch_adj = test_g.next()
            
            perturbation = None

            features_test, y_pred,y_pred_graph = net((batch_features, batch, perturbation))
            
            pos_out, neg_out, y_true_test = link_prediction_by_graph(batch_adj.cpu(), adj, features_test)
            pos_proba = torch.sigmoid(pos_out.flatten())
            neg_proba = torch.sigmoid(neg_out.flatten())
            
            y_proba_test = torch.cat([pos_proba, neg_proba])
            y_proba_test = y_proba_test.cpu()
            
            y_proba_test_label = torch.where(y_proba_test > sorted(y_proba_test)[len(y_proba_test)//2], torch.ones_like(y_proba_test), torch.zeros_like(y_proba_test)).int()

            auc_lp = roc_auc_score(y_true_test, y_proba_test)
            f1_lp = f1_score(y_true_test, y_proba_test_label)
            precision_lp, recall_lp, _ = precision_recall_curve(y_true_test, y_proba_test)
            pr_auc_lp = auc(recall_lp, precision_lp)
        
            auc_lp_batch_list.append(auc_lp)
            f1_lp_batch_list.append(f1_lp)
            pr_auc_lp_batch_list.append(pr_auc_lp)
            
        dur_test = time.time() - t_test    
    
    print('\n')
    print('Ontology Perturbation Detections')
    print('\n')
    print('repeat {} with totally {} epoches'.format((rpt + 1), (epoch + 1)))

    print('\n')
    print('AUC_lp = {}'.format(np.mean(auc_lp_batch_list)))
    print('PR_AUC_lp = {}'.format(np.mean(pr_auc_lp_batch_list)))
    print('F1_lp = {}'.format(np.mean(f1_lp_batch_list)))

    print('\n')

    print('test costs is {:.4f} seconds'.format(dur_test))
    print('model costs is {:.2f} mins'.format((time.time() - rpt_start)/60))
    print('——__——__——__——__——__——__——__——__——__——__')
    
    auc_lp_list.append(np.mean(auc_lp_batch_list))
    f1_lp_list.append(np.mean(f1_lp_batch_list))
    pr_auc_lp_list.append(np.mean(pr_auc_lp_batch_list))

print('----------------------------------------------------------------')
print('Ontology Perturbation Detections Summary')
print('\n')
print('name:', save_postfix)
print('\n')
print('total time costs {:.2f} hours'.format((time.time()-t_all_start)/3600))
print('\n')
print('AUC_lp_mean = {:.5f}, AUC_std = {:.5f}'.format(np.mean(auc_lp_list), np.std(auc_lp_list)))
print('PR_AUC_lp_mean = {:.5f}, AP_std = {:.5f}'.format(np.mean(pr_auc_lp_list), np.std(pr_auc_lp_list)))
print('F1_lp_mean = {:.5f}, AP_std = {:.5f}'.format(np.mean(f1_lp_list), np.std(f1_lp_list)))

print('\n')
print('AUC_lp history = {}'.format(auc_lp_list))
print('PR_AUC_lp history = {}'.format(pr_auc_lp_list))
print('F1_lp history = {}'.format(f1_lp_list))

print('\n')
print('model hyper-parameter:\n')
print(f'lr = {lr}, weight_decay = {weight_decay}, batch size = {batch_size}, skip-connection = {beta_1}; {beta_2}')
print(f'nlayers = {nlayers}, out_dim = {out_dim}, att_head_1 = {nheads_1}, att_head_2 = {nheads_2}, perturb ratio = {perturb_ratio}')
print(f'hidden_dim = {hidden_dim}, dropout_rate = {dropout_rate}, leaky_slope = {alpha}, gamma = {gamma}')
print(f'F_loss alpha= 0.64, F_loss gamma = 2.0')
print('\n')