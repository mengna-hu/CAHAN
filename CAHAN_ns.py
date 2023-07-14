import time
import torch
import numpy as np

from model import CAHAN_lp
from utils import load_BioNet_data_ns, EarlyStopping, index_generator, parse_minibatch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,recall_score,accuracy_score, precision_score,classification_report


num_drug = 1482
num_dis = 793

expected_metapaths = [
    [(0, 1, 0), (0, 2, 0)],
    [(1, 0, 1), (1, 3, 1)]
]
metahpath_mapping={
    (0, 1, 0):"usu",
    (0, 1, 3, 1, 0):"usgsu",
    (0, 2, 0):"utu",
    (1, 0, 1):"sus",
    (1, 3, 1):"sgs",
    (1,0,2,0,1):"sutus"
}

def main(args):
    adjlists_ua, adjM, type_mask, train_val_test_pos_drug_dis, train_val_test_neg_drug_dis = load_BioNet_data_ns()

    num_ntype = 2
    features_list = []
    in_dims = []
    save_postfix=args['save_postfix']
    print(save_postfix)
    # one-hot vector used to node features
    if args['feats_type'] == 0:
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(args['device']))
    elif args['feats_type'] == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(args['device']))

    train_pos_drug_dis = train_val_test_pos_drug_dis['train_pos_drug_dis']
    val_pos_drug_dis = train_val_test_pos_drug_dis['val_pos_drug_dis']
    test_pos_drug_dis = train_val_test_pos_drug_dis['test_pos_drug_dis']
    train_neg_drug_dis = train_val_test_neg_drug_dis['train_neg_drug_dis']
    val_neg_drug_dis = train_val_test_neg_drug_dis['val_neg_drug_dis']
    test_neg_drug_dis = train_val_test_neg_drug_dis['test_neg_drug_dis']
    y_true_test = np.array([1] * len(test_pos_drug_dis) + [0] * len(test_neg_drug_dis))

    auc_list = []
    ap_list = []
    f1_list=[]
    precision_list=[]
    recall_list=[]
    acc_list=[]

    print("--------------------------------run_CAHAN_ns------------------------------------------")
    print(f"The current metapathes are:"
          f"{metahpath_mapping[expected_metapaths[0][0]]},"
          f"{metahpath_mapping[expected_metapaths[0][1]]},"
          f"{metahpath_mapping[expected_metapaths[1][0]]},"
          f"{metahpath_mapping[expected_metapaths[1][1]]}.")
    for j in range(args['repeat']):
    # for i in range(1):
        net = CAHAN_lp(num_metapath_list=[2, 2],
                     feats_dim_list=in_dims,
                     in_size=args['in_size'],
                     hidden_size=args['hidden_size'],
                     out_size=args['out_size'],
                     num_heads=args['num_heads'],
                     dropout=args['dropout']).to(args['device'])
        # g = [graph.to(args['device']) for graph in g]
        # loss_fcn = torch.nn.CrossEntropyLoss()

        # training loop
        optimizer = torch.optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        stopper = EarlyStopping(patience=args['patience'], verbose=True, save_path=f'./checkpoint/checkpoint_{save_postfix}_{j}.pt')

        # dur1 = []
        # dur2 = []
        # dur3 = []
        # train_pos_idx_generator = index_generator(batch_size=args['batch_size'], num_data=len(train_pos_drug_dis))
        # val_idx_generator = index_generator(batch_size=args['batch_size'], num_data=len(val_pos_drug_dis),
        #                                     shuffle=False)
        
        # for epoch in range(args['epoch']):
        # # for epoch in range(10):
        #     t_start = time.time()
        #     #training
        #     net.train()
        #     for iteration in range(train_pos_idx_generator.num_iterations()):
        #         # forward
        #         t0 = time.time()
        #         train_pos_idx_batch = train_pos_idx_generator.next()
        #         train_pos_idx_batch.sort()
        #         train_pos_drug_dis_batch = train_pos_drug_dis[train_pos_idx_batch].tolist()
        #         train_neg_idx_batch = np.random.choice(len(train_neg_drug_dis), len(train_pos_idx_batch))
        #         train_neg_idx_batch.sort()
        #         train_neg_drug_dis_batch = train_neg_drug_dis[train_neg_idx_batch].tolist()
        
        #         train_pos_g_lists, train_pos_idx_batch_mapped_lists, train_pos_indices_lists = parse_minibatch(
        #             adjlists_ua, train_pos_drug_dis_batch, args['device'])
        #         train_neg_g_lists, train_neg_idx_batch_mapped_lists, train_neg_indices_lists = parse_minibatch(
        #             adjlists_ua, train_neg_drug_dis_batch, args['device'])
        #         if epoch==1:
        #             continue
                
        #         t1 = time.time()
        #         dur1.append(t1 - t0)
        
        #         embedding_pos_drug, embedding_pos_dis = net(
        #             (train_pos_g_lists, features_list, type_mask, train_pos_idx_batch_mapped_lists, train_pos_indices_lists))
        #         embedding_neg_drug, embedding_neg_dis = net(
        #             (train_neg_g_lists, features_list, type_mask, train_neg_idx_batch_mapped_lists, train_neg_indices_lists))
        
        #         embedding_pos_drug = embedding_pos_drug.view(-1, 1, embedding_pos_drug.shape[1])
        #         embedding_pos_dis = embedding_pos_dis.view(-1, embedding_pos_dis.shape[1], 1)
        
        #         embedding_neg_drug = embedding_neg_drug.view(-1, 1, embedding_neg_drug.shape[1])
        #         embedding_neg_dis = embedding_neg_dis.view(-1, embedding_neg_dis.shape[1], 1)
        
        #         pos_out = torch.bmm(embedding_pos_drug, embedding_pos_dis)
        #         neg_out = -torch.bmm(embedding_neg_drug, embedding_neg_dis)
        #         train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
        
        #         t2 = time.time()
        #         dur2.append(t2 - t1)
        
        #         # autograd
        #         optimizer.zero_grad()
        #         train_loss.backward()
        #         optimizer.step()
        
        #         t3 = time.time()
        #         dur3.append(t3 - t2)
        
        #         # print training info
        #         if iteration % 100 == 0:
        #             print(
        #                 'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
        #                     epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
        
        #     # validation
        #     net.eval()
        #     val_loss = []
        #     with torch.no_grad():
        #         for iteration in range(val_idx_generator.num_iterations()):
        #             # forward
        #             val_idx_batch = val_idx_generator.next()  # 生成待选取的边的索引
        #             val_pos_drug_dis_batch = val_pos_drug_dis[val_idx_batch].tolist()
        #             val_neg_drug_dis_batch = val_neg_drug_dis[val_idx_batch].tolist()
        
        
        #             val_pos_g_lists, val_pos_idx_batch_mapped_lists, val_pos_indices_lists = parse_minibatch(
        #                 adjlists_ua, val_pos_drug_dis_batch, args['device'])
        #             val_neg_g_lists, val_neg_idx_batch_mapped_lists, val_neg_indices_lists = parse_minibatch(
        #                 adjlists_ua, val_neg_drug_dis_batch, args['device'])
        
        #             t1 = time.time()
        #             dur1.append(t1 - t0)
        
        #             embedding_pos_drug, embedding_pos_dis = net(
        #                 (val_pos_g_lists, features_list, type_mask, val_pos_idx_batch_mapped_lists, val_pos_indices_lists))
        #             embedding_neg_drug, embedding_neg_dis = net(
        #                 (val_neg_g_lists, features_list, type_mask, val_neg_idx_batch_mapped_lists, val_neg_indices_lists))
        
        #             embedding_pos_drug = embedding_pos_drug.view(-1, 1, embedding_pos_drug.shape[1])
        #             embedding_pos_dis = embedding_pos_dis.view(-1, embedding_pos_dis.shape[1], 1)
        #             embedding_neg_drug = embedding_neg_drug.view(-1, 1, embedding_neg_drug.shape[1])
        #             embedding_neg_dis = embedding_neg_dis.view(-1, embedding_neg_dis.shape[1], 1)
        
        #             pos_out = torch.bmm(embedding_pos_drug, embedding_pos_dis)
        #             neg_out = -torch.bmm(embedding_neg_drug, embedding_neg_dis)
        
        #             val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
        #         val_loss = torch.mean(torch.tensor(val_loss))
        
        #     t_end = time.time()
        #     # print validation info
        #     print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
        #         epoch, val_loss.item(), t_end - t_start))
        #     # early stopping
        #     if stopper(val_loss, net):
        #         print('Early stopping!')
        #         break

        test_idx_generator = index_generator(batch_size=args['batch_size'], num_data=len(test_pos_drug_dis),
                                             shuffle=False)
        stopper.load_checkpoint(net)
        net.eval()
        embeddings = {}
        pos_proba_list = []
        neg_proba_list = []
        drug_embedding_list = []
        dis_embedding_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_drug_dis_batch = test_pos_drug_dis[test_idx_batch].tolist()
                test_neg_drug_dis_batch = test_neg_drug_dis[test_idx_batch].tolist()


                test_pos_g_lists, test_pos_idx_batch_mapped_lists, test_pos_indices_lists = parse_minibatch(
                    adjlists_ua, test_pos_drug_dis_batch, args['device'])
                test_neg_g_lists, test_neg_idx_batch_mapped_lists, test_neg_indices_lists = parse_minibatch(
                    adjlists_ua, test_neg_drug_dis_batch, args['device'])

                embedding_pos_drug, embedding_pos_dis = net(
                    (test_pos_g_lists, features_list, type_mask, test_pos_idx_batch_mapped_lists, test_pos_indices_lists))
                embedding_neg_drug, embedding_neg_dis = net(
                    (test_neg_g_lists, features_list, type_mask, test_neg_idx_batch_mapped_lists, test_neg_indices_lists))

                for i, pair in enumerate(test_pos_drug_dis_batch):
                    embeddings[pair[0]] = embedding_pos_drug[i].cpu().numpy()
                    embeddings[pair[1]+num_drug] = embedding_pos_dis[i].cpu().numpy()
                for i, pair in enumerate(test_neg_drug_dis_batch):
                    embeddings[pair[0]] = embedding_neg_drug[i].cpu().numpy()
                    embeddings[pair[1]+num_drug] = embedding_neg_dis[i].cpu().numpy()

                embedding_pos_drug = embedding_pos_drug.view(-1, 1, embedding_pos_drug.shape[1])
                embedding_pos_dis = embedding_pos_dis.view(-1, embedding_pos_dis.shape[1], 1)
                embedding_neg_drug = embedding_neg_drug.view(-1, 1, embedding_neg_drug.shape[1])
                embedding_neg_dis = embedding_neg_dis.view(-1, embedding_neg_dis.shape[1], 1)

                drug_embedding_list.append(embedding_pos_drug .squeeze().cpu().numpy())
                dis_embedding_list.append(embedding_pos_dis.squeeze().cpu().numpy())
                pos_out = torch.bmm(embedding_pos_drug, embedding_pos_dis).flatten()
                neg_out = torch.bmm(embedding_neg_drug, embedding_neg_dis).flatten()

                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))

            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
            y_pred = [1 if y_proba_test[i] >= 0.5 else 0 for i in range(len(y_proba_test))]
        # np.savez(rf'./record/record_ns/CAHAN_prediction_result_{j}.npz', y_true=y_true_test, y_pred=y_proba_test)
        # np.savez(rf'./record/record_ns/CAHAN_embedding2vis_InfoMaxFusv1_{j}.npz', drug=np.concatenate(drug_embedding_list, axis=0),
        #          disease=np.concatenate(dis_embedding_list, axis=0))

        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        f1score = f1_score(y_true_test, y_pred)
        accuracy = accuracy_score(y_true_test, y_pred, normalize=True)
        precision = precision_score(y_true_test, y_pred, average='binary',pos_label=1)
        recall = recall_score(y_true_test, y_pred, average='binary')
        
        print('Link Prediction Test')
        print('AUC = %.4f,AP = %.4f, f1-socre = %.4f, accuracy= %.4f,precision =%.4f,recall =%.4f'% (auc,ap,f1score,accuracy,precision,recall))
        auc_list.append(auc)
        ap_list.append(ap)
        f1_list.append(round(f1score,4))
        acc_list.append(round(accuracy,4))
        precision_list.append(round(precision,4))
        recall_list.append(round(recall,4))

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(round(np.mean(auc_list),4), round(np.std(auc_list),4)))
    print('AP_mean = {}, AP_std = {}'.format(round(np.mean(ap_list),4), round(np.std(ap_list),4)))
    print('F1Score_mean = {}, AUC_std = {}'.format(round(np.mean(f1_list),4), round(np.std(f1_list),4)))
    print('accuracy_mean = {}, AP_std = {}'.format(round(np.mean(acc_list),4), round(np.std(acc_list),4)))
    print('precision_mean = {}, AP_std = {}'.format(round(np.mean(precision_list),4), round(np.std(precision_list),4)))    
    print('recall_mean = {}, AUC_std = {}'.format(round(np.mean(recall_list),4), round(np.std(recall_list),4)))



if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=2,
                        help='Random seed')
    parser.add_argument('--feats-type', type=int, default=0,
                        help='Type of the node features used. ' +
                             '0 - all id vectors; ' +
                             '1 - all zero vector. Default is 0.')
    parser.add_argument('--save-postfix',default='ns')  
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
