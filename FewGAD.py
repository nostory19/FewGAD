import os
import dgl
import pdb
import random
import argparse
import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler

from model import *
from utils import *
from tqdm import tqdm


def Rnce_loss(logits, lam, q):
    # 0.5, 0.3
    exps = torch.exp(logits)
    pos = -(exps[:,0])**q/q
    neg = ((lam*(exps.sum(1)))**q) / q
    loss = pos.mean() + neg.mean()
    return loss


# Set argument
def train_model(dataset, lr, num_epoch, batch_size, auc_test_rounds, t, k, alpha, beta, few_size, gamma, seed):

    dataset = dataset
    lr = lr
    weight_decay = 0.0
    seed = seed
    embedding_dim = 64
    num_epoch = num_epoch
    drop_out = 0.0
    batch_size = batch_size
    readout = 'avg'
    auc_test_rounds = auc_test_rounds
    negsamp_ratio = 1 # 4
    t = t
    k = k
    alpha = alpha
    beta = beta
    gamma = gamma
    few_size = few_size

    batch_size = batch_size
    subgraph_size = t
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set random seed
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Load and preprocess data
    adj, features, labels, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(dataset)

    adj_o = adj
    raw_features = features.todense()
    features, _ = preprocess_features(features)

    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    # """
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    hopk = k_order_neighbors(adj, k)
    subgraphs = search_path(adj, subgraph_size-1, hopk)

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    raw_features = torch.FloatTensor(raw_features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)

    model = Model(ft_size, embedding_dim, 'prelu', negsamp_ratio, readout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([negsamp_ratio])).to(device)
    cnt_wait = 0
    best = 1e9
    best_t = 0
    batch_num = nb_nodes // batch_size +1
    mse_loss = nn.MSELoss(reduction='mean')

    graph_nodes = [i for i in range(nb_nodes)]
    ano_nodes = [node for node, label in zip(graph_nodes, ano_label) if label==1]
    neg_nodes = random.sample(ano_nodes, few_size)

    # Train model
    with tqdm(total=num_epoch) as pbar:
        pbar.set_description('Training')
        for epoch in range(num_epoch):

            loss_full_batch = torch.zeros((nb_nodes,1)).to(device)
            model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)
                cur_batch_neg_nodes = [node for node in idx if node in neg_nodes]
                cur_neg_index = [idx.index(node) for node in cur_batch_neg_nodes]
                cur_neg_index = torch.tensor(cur_neg_index).to(device)

                lbl_pos = torch.ones(cur_batch_size).to(device)
                lbl_neg = torch.zeros(cur_batch_size * negsamp_ratio).to(device)
                if cur_neg_index.numel() > 0:
                    lbl_neg[cur_neg_index] = 1
                lbl = torch.cat((lbl_pos, lbl_neg)).unsqueeze(1)

                ba = []
                bf = []
                ba_neg = []
                bf_neg = []
                raw_bf1 = []

                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)


                neg_adj_zero_col = torch.zeros((few_size, subgraph_size + 1, 1))
                neg_adj_zero_col[:, -1, :] = 1.

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                    raw_cur_feat_1 = raw_features[:, subgraphs[i], :]
                    raw_bf1.append(raw_cur_feat_1)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)
                raw_bf1 = torch.cat(raw_bf1)
                raw_bf1 = torch.cat((raw_bf1[:, :-1, :], added_feat_zero_row, raw_bf1[:, -1:, :]), dim=1)

                neg_idx = neg_nodes
                for i in neg_idx:
                    cur_adj_neg = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat_neg = features[:, subgraphs[i], :]
                    ba_neg.append(cur_adj_neg)
                    bf_neg.append(cur_feat_neg)

                ba_neg = torch.cat(ba_neg)
                bf_neg = torch.cat(bf_neg)

                logits,  f_1,  = model(bf, ba, bf_neg, ba_neg, raw_bf1, cur_neg_index)

                # Fine tuning.
                pos_scores = logits[:cur_batch_size] #(200, 1)
                neg_scores = logits[cur_batch_size:cur_batch_size+cur_batch_size*(few_size+100)] #(2000, 1)
                normal_neg_scores = logits[cur_batch_size+cur_batch_size*(few_size+100):] #(200, 1)
                split_scores = torch.chunk(neg_scores, few_size+100, dim=0)
                neg_scores = torch.stack(split_scores, dim=1)# (200, 10, 1)
                neg_scores = neg_scores.squeeze(-1) #(200, 10)
                min_or_avg_scores = torch.zeros(cur_batch_size).to(device)
                for i in range(cur_batch_size):
                    pos_score = pos_scores[i]
                    cur_normal_neg = normal_neg_scores[i]
                    cur_few_neg = neg_scores[i]

                    if epoch < 100:
                        min_or_avg_scores[i] = cur_normal_neg
                    else:
                        if i in cur_neg_index:
                            min_or_avg_scores[i]  = cur_normal_neg
                            # print("in+")
                        else:
                            min_or_avg_scores[i] = gamma * neg_scores[i].min() + (1 - gamma) * cur_normal_neg

                min_or_avg_scores = min_or_avg_scores.unsqueeze(1)
                logits = torch.cat([pos_scores, min_or_avg_scores])
                loss1 = b_xent(logits, lbl)
                loss1 = torch.nanmean(loss1)
                loss2 = mse_loss(f_1[:, -2, :], raw_bf1[:, -1, :])
                loss = alpha*loss1 + beta*loss2
                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()

                if not is_final_batch:
                    total_loss += loss

                mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

                if mean_loss < best:
                    best = mean_loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), './data/'+dataset+'/best_model.pkl')
                else:
                    cnt_wait += 1

            pbar.set_postfix(loss=mean_loss)
            pbar.update(1)

    # Test model
    print('Loading {}th epoch'.format(best_t))
    print("Testing, Loading best model.pkl")
    model.load_state_dict(torch.load('./data/'+dataset+'/best_model.pkl'))

    multi_round_ano_score = np.zeros((auc_test_rounds, nb_nodes))
    multi_round_ano_score_p = np.zeros((auc_test_rounds, nb_nodes))
    multi_round_ano_score_n = np.zeros((auc_test_rounds, nb_nodes))

    with tqdm(total=auc_test_rounds) as pbar_test:
        pbar_test.set_description('Testing')
        for round in range(auc_test_rounds):

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)
                cur_batch_neg_nodes = [node for node in idx if node in neg_nodes]
                cur_neg_index = [idx.index(node) for node in cur_batch_neg_nodes]
                cur_neg_index = torch.tensor(cur_neg_index).to(device)

                ba = []
                bf = []
                ba_neg = []
                bf_neg = []
                raw_bf1 = []

                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)

                neg_adj_zero_col = torch.zeros((few_size, subgraph_size + 1, 1))
                neg_adj_zero_col[:, -1, :] = 1.
                neg_adj_zero_col = neg_adj_zero_col.to(device)



                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                    raw_cur_feat_1 = raw_features[:, subgraphs[i], :]
                    raw_bf1.append(raw_cur_feat_1)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
                raw_bf1 = torch.cat(raw_bf1)
                raw_bf1 = torch.cat((raw_bf1[:, :-1, :], added_feat_zero_row, raw_bf1[:, -1:, :]), dim=1)


                neg_idx = neg_nodes
                for i in neg_idx:
                    cur_adj_neg = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat_neg = features[:, subgraphs[i], :]
                    ba_neg.append(cur_adj_neg)
                    bf_neg.append(cur_feat_neg)

                ba_neg = torch.cat(ba_neg)
                bf_neg = torch.cat(bf_neg)

                with torch.no_grad():
                    logits, f_1 = model(bf, ba, bf_neg, ba_neg, raw_bf1, cur_neg_index)

                pos_scores = logits[:cur_batch_size]  # (200, 1)
                neg_scores = logits[cur_batch_size:cur_batch_size + cur_batch_size * (few_size+100)]  # (2000, 1)
                normal_neg_scores = logits[cur_batch_size + cur_batch_size * (few_size + 100):]  # (200, 1)
                split_scores = torch.chunk(neg_scores, few_size+100, dim=0)
                neg_scores = torch.stack(split_scores, dim=1)  # (200, 10, 1)
                neg_scores = neg_scores.squeeze(-1)
                min_or_avg_scores = torch.zeros(cur_batch_size).to(device)
                for i in range(cur_batch_size):
                    cur_normal_neg = normal_neg_scores[i]
                    cur_few_neg = neg_scores[i]
                    if i in cur_neg_index:
                        min_or_avg_scores[i] = cur_normal_neg
                    else:
                        min_or_avg_scores[i] = gamma * neg_scores[i].min() + (1 - gamma) * cur_normal_neg

                min_or_avg_scores = min_or_avg_scores.unsqueeze(1)
                logits = torch.cat([pos_scores, min_or_avg_scores])
                logits = torch.squeeze(logits)
                logits = torch.nan_to_num(logits)
                logits = torch.sigmoid(logits)

                neg_scores = logits[cur_batch_size:]  # 取所有负样本得分 (800,1)
                ano_score = -(logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
                scaler1 = MinMaxScaler()
                scaler2 = MinMaxScaler()
                pdist = nn.PairwiseDistance(p=2)
                dist1 = pdist(f_1[:, -2, :], raw_bf1[:, -1, :])
                ano_score_2 = dist1.cpu().numpy()
                ano_score_2 = scaler2.fit_transform(ano_score_2.reshape(-1, 1)).reshape(-1)

                ano_score = scaler1.fit_transform(ano_score.reshape(-1, 1)).reshape(-1)
                ano_score = alpha * ano_score + beta * ano_score_2

                multi_round_ano_score[round, idx] = ano_score
            pbar_test.update(1)

    ano_score_final = np.nanmean(multi_round_ano_score, axis=0)
    # np.save("./data/"+dataset+"/ano_score_final.npy", ano_score_final)
    # np.save("./data/"+dataset+"/ano_label.npy", ano_label)
    #
    auc = roc_auc_score(ano_label, ano_score_final)
    print('AUC:{:.4f}'.format(auc))
    return auc

