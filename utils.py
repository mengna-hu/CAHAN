import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio


def load_BioNet_data(prefix='/home/humn/baselines/CAHAN/BackupFile'):
    data_path = prefix+'/adjData.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    drug_usu= data['USU']
    drug_usgsu=data['USGSU']
    drug_utu=data['UTU']

    dis_sus=data['SUS']
    dis_sgs=data['SGS']
    dis_sgtgs=data['SUTUS']

    # c0结果：AUC_mean=1.0,AP_mean=1.0
    # adjlist_ua = [[drug_usu, drug_utu], [dis_sus, dis_sgs]]

    # c1结果：AUC_mean = 0.9999,AP_mean = 0.9999
    #adjlist_ua = [[drug_usu,drug_utgtu],[dis_sus,dis_ss]]

    # c2结果：AUC_mean=99.3785,AP_mean=99.3693
    #adjlist_ua = [[drug_usu, drug_utu], [dis_sgs, dis_ss]]

    # c3结果：AUC_mean=0.9683,AP_mean=0.9682
    #adjlist_ua = [[drug_ussu, drug_utgtu], [dis_ss, dis_sgtgs]]

    # AUC=1.0,ap=0.99999
    # adjlist_ua = [[drug_ussu, drug_utgtu], [dis_sus, dis_sgtgs]]

    #final结果：AUC_mean=0.9679,AP_mean=0.9697
    adjlist_ua = [[drug_usu, drug_utu], [dis_sus, dis_sgs]]

    adjM = sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types_NSAP.npy')
    train_val_test_pos_drug_dis = np.load(prefix + '/train_val_test_pos_drug_dis.npz')
    train_val_test_neg_drug_dis= np.load(prefix + '/train_val_test_neg_drug_dis.npz')
    # train_val_test_neg_drug_dis= np.load(prefix + '/ns_train_val_test_neg_drug_dis.npz')
    return adjlist_ua ,adjM, type_mask, train_val_test_pos_drug_dis, train_val_test_neg_drug_dis

def load_BioNet_data_ns(prefix='./BackupFile'):
    data_path = prefix+'/adjData.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    drug_usu= data['USU']
    drug_usgsu=data['USGSU']
    drug_utu=data['UTU']

    dis_sus=data['SUS']
    dis_sgs=data['SGS']
    dis_sgtgs=data['SUTUS']

    # c0结果：AUC_mean=1.0,AP_mean=1.0
    # adjlist_ua = [[drug_usu, drug_utu], [dis_sus, dis_sgs]]

    # c1结果：AUC_mean = 0.9999,AP_mean = 0.9999
    #adjlist_ua = [[drug_usu,drug_utgtu],[dis_sus,dis_ss]]

    # c2结果：AUC_mean=99.3785,AP_mean=99.3693
    #adjlist_ua = [[drug_usu, drug_utu], [dis_sgs, dis_ss]]

    # c3结果：AUC_mean=0.9683,AP_mean=0.9682
    #adjlist_ua = [[drug_ussu, drug_utgtu], [dis_ss, dis_sgtgs]]

    # AUC=1.0,ap=0.99999
    # adjlist_ua = [[drug_ussu, drug_utgtu], [dis_sus, dis_sgtgs]]

    #final结果：AUC_mean=0.9679,AP_mean=0.9697
    adjlist_ua = [[drug_usu, drug_utu], [dis_sus, dis_sgs]]

    adjM = sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types_NSAP.npy')
    train_val_test_pos_drug_dis = np.load(prefix + '/train_val_test_pos_drug_dis.npz')
    # train_val_test_neg_drug_dis= np.load(prefix + '/train_val_test_neg_drug_dis.npz')
    train_val_test_neg_drug_dis= np.load(prefix + '/dd_ns_train_val_test_neg_drug_dis.npz')
    return adjlist_ua ,adjM, type_mask, train_val_test_pos_drug_dis, train_val_test_neg_drug_dis

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# The configuration below is from the paper.
default_configure = {
    'lr': 0.001,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'in_size':128,
    'hidden_size': 128,
    'out_size':128,
    'dropout': 0.5,
    'weight_decay': 0.001,
    'epoch': 200,   # Number of epochs. Default is 100.
    'patience': 5,
    'batch_size':8,
    'repeat':3,  #'Repeat the training and testing for N times. Default is 1.'
}

sampling_configure = {
    'batch_size': 8
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args


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
        self.save_path=save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # dt = datetime.datetime.now()
        # self.save_path = './checkpoint/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
        #     dt.date(), dt.hour, dt.minute, dt.second)
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model, ckt_path=None):
        """Load the latest checkpoint."""
        #self.save_path='./checkpoint/checkpoint_LastNSAP_1.pt'
        if not ckt_path:
            model.load_state_dict(torch.load(self.save_path))
        else:
            model.load_state_dict(torch.load(ckt_path))


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0


def parse_minibatch(adjlists_ua, user_artist_batch, device):
    g_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    idx_node = [[], []]
    for mode, adjlists in enumerate(adjlists_ua):
        for adjlist in adjlists:
            # nodelist = [adjlist[row[mode]] for row in user_artist_batch]
            nodelist=[]
            for row in user_artist_batch:
                cur_node=np.array([row[mode]])
                cur_nodelist=np.nonzero(adjlist[cur_node].todense())[1]
                nodelist.append(np.concatenate((cur_node,cur_nodelist),axis=0))
            edges = []
            nodes = set()
            for row_parsed in nodelist:
                # if mode == 1:
                #     row_parsed += offset
                nodes.add(row_parsed[0])
                if len(row_parsed) > 1:
                    neighbors = np.asarray(row_parsed[1:])
                else:
                    neighbors = np.asarray([row_parsed[0]])

                for dst in neighbors:
                    nodes.add(dst)
                    edges.append((row_parsed[0], dst))

            mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
            edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
            g = dgl.DGLGraph()
            g=g.to(device)
            g.add_nodes(len(nodes))
            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            g = dgl.add_self_loop(g) #消除0入度结点所带来的影响
            g_lists[mode].append(g)
            idx_batch_mapped_lists[mode].append(
                np.array([mapping[row[mode]]
                          for row in user_artist_batch]))
            idx_node[mode].append(torch.LongTensor(list(sorted(nodes))).to(device))
    # print("g_list: ", len(g_lists), len(g_lists[0]), len(g_lists[1]))
    # print("idx_batch_map; ", len(idx_batch_mapped_lists), len(idx_batch_mapped_lists[0]), len(idx_batch_mapped_lists[1]))
    # print("idx_node: ", len(idx_node), len(idx_node[0]), len(idx_node[1]))
    # exit(0)
    return g_lists, idx_batch_mapped_lists, idx_node

