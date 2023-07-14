import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dgl.nn.pytorch import GATConv

class AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        """ 
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        Cross-Att: Query and Value should always come from the same source (Aiming to forcus on), Key comes from the other source
        Self-Att : Both three Query, Key, Value come form the same source (For refining purpose)
        """

        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        #表示按行计算:  dim = 0, 对每一列进行softmax
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.node_attn_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.node_attn_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attn_layers = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, g_lists, feat_lists, target_idx_lists):
        node_embeddings = []
        for node_attn_layer, g ,feature,target_idx in zip(self.node_attn_layers,g_lists,feat_lists,target_idx_lists):
            node_embeddings.append(node_attn_layer(g, feature).flatten(1)[target_idx])
        node_embeddings = torch.stack(node_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attn_layers(node_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.fc = nn.Linear(hidden_size * num_heads[-1], out_size)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)


    def forward(self, g_lists, feat_lists, target_idx_lists):
        for layer in self.layers:
            h = layer(g_lists, feat_lists, target_idx_lists)

        return self.fc(h)


class CAHAN_lp(nn.Module):
    def __init__(self, num_metapath_list, feats_dim_list, in_size, hidden_size, out_size, num_heads, dropout):
        super(CAHAN_lp, self).__init__()
        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_size, bias=True) for feats_dim in feats_dim_list])
        self.num_metapaths = num_metapath_list
        self.hidden_size = hidden_size
        #drug层
        self.layer1 = HAN(num_metapath_list[0], in_size, hidden_size, out_size, num_heads, dropout)
        #disease层
        self.layer2 = HAN(num_metapath_list[1], in_size, hidden_size, out_size, num_heads, dropout)
        # 嵌套交叉注意力
        self.att = AttentionBlock(hid_dim = out_size, n_heads = 1, dropout=dropout)

        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)


        # feature dropout after trainsformation
        if dropout > 0:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x

    def forward(self, inputs):
        g_lists, features_list, type_mask, target_idx_lists, idx_node_lists = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_size, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        feat_list = [[], []]
        for i, nlist in enumerate(idx_node_lists):
            for nodes in nlist:
                feat_list[i].append(transformed_features[nodes])

        drug_h = self.layer1(g_lists[0], feat_list[0], target_idx_lists[0])
        dis_h = self.layer2(g_lists[1], feat_list[1], target_idx_lists[1])

        drug_h = self.att(drug_h,dis_h,dis_h)
        dis_h = self.att(dis_h,drug_h,drug_h)
        
        return [drug_h, dis_h]