#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from gene_ii_co_oc import load_sp_mat
from models.AsymModule import AsymMatrix
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix


#Adding UHBR
def Split_HyperGraph_to_device(H, device, split_num=16):
    H_list = []
    length = H.shape[0] // split_num
    for i in range(split_num):
        if i == split_num - 1:
            H_list.append(H[length * i : H.shape[0]])
        else:
            H_list.append(H[length * i : length * (i + 1)])
    H_split = [SparseTensor.from_scipy(H_i).to(device) for H_i in H_list]
    return H_split
def normalize_Hyper(H):
    D_v = sp.diags(1 / (np.sqrt(H.sum(axis=1).A.ravel()) + 1e-8))
    D_e = sp.diags(1 / (np.sqrt(H.sum(axis=0).A.ravel()) + 1e-8))
    H_nomalized = D_v @ H @ D_e @ H.T @ D_v
    return H_nomalized


def mix_hypergraph(raw_graph, threshold=10):
    ui_graph, bi_graph, ub_graph = raw_graph

    uu_graph = ub_graph @ ub_graph.T
    for i in range(ub_graph.shape[0]):
        for r in range(uu_graph.indptr[i], uu_graph.indptr[i + 1]):
            uu_graph.data[r] = 1 if uu_graph.data[r] > threshold else 0

    bb_graph = ub_graph.T @ ub_graph
    for i in range(ub_graph.shape[1]):
        for r in range(bb_graph.indptr[i], bb_graph.indptr[i + 1]):
            bb_graph.data[r] = 1 if bb_graph.data[r] > threshold else 0
    # print(ui_graph.shape[1], bi_graph.shape[1])
    H = sp.vstack((ui_graph.T, bi_graph.T))
    non_atom_graph = sp.vstack((ub_graph, bb_graph))
    non_atom_graph = sp.hstack((non_atom_graph, sp.vstack((uu_graph, ub_graph.T))))
    H = sp.hstack((H, non_atom_graph))
    return H
def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    
    graph = coo_matrix(graph)
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class CrossCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]

        self.w1 = conf["w1"]
        self.w2 = conf["w2"]
        self.w3 = conf["w3"]
        self.w4 = conf["w4"]
        self.extra_layer = conf["extra_layer"]

        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        print(type(self.ui_graph))

        self.ubi_graph = self.ub_graph @ self.bi_graph

        self.ovl_ui = self.ubi_graph.tocsr().multiply(self.ui_graph.tocsr())
        self.ovl_ui = self.ovl_ui > 0
        self.non_ovl_ui = self.ui_graph - self.ovl_ui
        # w1: 0.8, w2: 0.2
        self.ui_graph = self.ovl_ui * self.w1 + self.non_ovl_ui * self.w2

        # generate the graph without any dropouts for testing
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()
        self.get_user_agg_graph_ori()

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()
        self.get_user_agg_graph()

        self.init_md_dropouts()

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

        # light-gcn weight
        temp = self.conf["UB_coefs"]
        self.UB_coefs = torch.tensor(temp).unsqueeze(0).unsqueeze(-1).to(self.device)
        temp = self.conf["BI_coefs"]
        self.BI_coefs = torch.tensor(temp).unsqueeze(0).unsqueeze(-1).to(self.device)
        temp = self.conf["UI_coefs"]
        self.UI_coefs = torch.tensor(temp).unsqueeze(0).unsqueeze(-1).to(self.device)
        del temp
        self.a_self_loop = self.conf["self_loop"]
        self.n_head = self.conf["nhead"]
        # ii-asym matrix
        self.sw = conf["sw"]
        self.nw = conf["nw"]
        self.ibi_edge_index = torch.tensor(np.load("datasets/{}/n_neigh_ibi.npy".format(conf["dataset"]), allow_pickle=True)).to(self.device)
        self.iui_edge_index = torch.tensor(np.load("datasets/{}/n_neigh_iui.npy".format(conf["dataset"]), allow_pickle=True)).to(self.device)
        self.iui_gat_conv = Amatrix(in_dim=64, out_dim=64, n_layer=1, dropout=0.1, heads=self.n_head, concat=False, self_loop=self.a_self_loop, extra_layer=self.extra_layer)
        self.ibi_gat_conv = Amatrix(in_dim=64, out_dim=64, n_layer=1, dropout=0.1, heads=self.n_head, concat=False, self_loop=self.a_self_loop, extra_layer=self.extra_layer)
        self.iui_asym = to_tensor(self.get_ii_asym(self.ui_graph.T)).to(self.device)
        print(type(self.iui_asym))
        print("done GEN iui asym")
        self.ibi_asym = to_tensor(self.get_ii_asym(self.ubi_graph.T)).to(self.device)
        print("done GEN ibi asym")
        print(type(self.ibi_asym))


        #UHBR
        self.num_users, self.num_bundles, self.num_items = (
            self.ub_graph.shape[0],
            print(self.ub_graph.shape[0])
            self.ub_graph.shape[1],
            print(self.ub_graph.shape[1]),
            self.ui_graph.shape[1],
            print(self.ui_graph.shape[1])
        )
        H = mix_hypergraph(raw_graph)
        self.atom_graph = Split_HyperGraph_to_device(normalize_Hyper(H), device)

        print("finish generating hypergraph")
        # embeddings
        self.users_feature_hg = nn.Parameter(
            torch.FloatTensor(self.num_users, self.embedding_size).normal_(0, 0.5 / self.embedding_size)
        )
        self.bundles_feature_hg = nn.Parameter(
            torch.FloatTensor(self.num_bundles, self.embedding_size).normal_(0, 0.5 / self.embedding_size)
        )
        self.user_bound = nn.Parameter(
            torch.FloatTensor(self.embedding_size, 1).normal_(0, 0.5 / self.embedding_size)
        )
        self.drop = nn.Dropout(0.2)
        
    def hyper_propagate(self):
        embed_0 = torch.cat([self.users_feature_hg, self.bundles_feature_hg], dim=0)
        embed_1 = torch.cat([G @ embed_0 for G in self.atom_graph], dim=0)
        all_embeds = embed_0 / 2 + self.drop(embed_1) / 3
        users_feature_hg, bundles_feature_hg = torch.split(
            all_embeds, [self.num_users, self.num_bundles], dim=0
        )

        return users_feature_hg, bundles_feature_hg
    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)

    def get_ii_asym(self, ix_mat):
        ii_co = ix_mat @ ix_mat.T
        i_count = ix_mat.sum(axis=1)
        print(i_count)
        i_count += (i_count == 0) # mask all zero with 1
        print(i_count)
        # return norm_ii
        # return ii_asym
        mask = ii_co > 4
        ii_co = ii_co.multiply(mask)
        ii_asym = ii_co / i_count
        return ii_asym
        
    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        bi_graph = self.bi_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        bi_propagate_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph], [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])
        self.bi_propagate_graph_ori = to_tensor(laplace_transform(bi_propagate_graph)).to(device)
        
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

                graph2 = bi_propagate_graph.tocoo()
                values2 = np_edge_dropout(graph2.data, modification_ratio)
                bi_propagate_graph = sp.coo_matrix((values2, (graph2.row, graph2.col)), shape=graph2.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)
        self.bi_propagate_graph = to_tensor(laplace_transform(bi_propagate_graph)).to(device)


    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_level_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.bundle_level_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)


    def get_user_agg_graph(self):
        ui_graph = self.ui_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.ui_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            ui_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        user_size = ui_graph.sum(axis=1) + 1e-8
        ui_graph = sp.diags(1/user_size.A.ravel()) @ ui_graph
        self.user_agg_graph = to_tensor(ui_graph).to(device)


    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)

    
    def get_user_agg_graph_ori(self):
        ui_graph = self.ui_graph
        user_size = ui_graph.sum(axis=1) + 1e-8
        ui_graph = sp.diags(1/user_size.A.ravel()) @ ui_graph
        self.user_agg_graph_ori = to_tensor(ui_graph).to(self.device)


    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test, coefs=None):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test: # !!! important
                features = mess_dropout(features)

            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        if coefs is not None:
            all_features = all_features * coefs
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    
    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature
    
    
    def get_IL_user_rep(self, IL_items_feature, test):
        if test:
            IL_users_feature = torch.matmul(self.user_agg_graph_ori, IL_items_feature)
        else:
            IL_users_feature = torch.matmul(self.user_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_users_feature = self.bundle_agg_dropout(IL_users_feature)

        return IL_users_feature


    def propagate(self, test=False):
        #  =============================  item level propagation  =============================
        #  ======== UI =================
        IL_items_feat = self.iui_gat_conv(self.items_feature, self.iui_edge_index) * self.nw + self.items_feature * self.sw
        # self.items_feature * self.iui_edge_index
        # N*N matmul#(n*64)

        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori, self.users_feature, IL_items_feat, self.item_level_dropout, test, self.UI_coefs)
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature, IL_items_feat, self.item_level_dropout, test, self.UI_coefs)
        # IL_items_feat = torch.spmm((self.iui_asym),IL_items_feature) * self.nw + IL_items_feature * self.sw
        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)

        # ========== BI ================
        IL_items_feat2 = self.ibi_gat_conv(self.items_feature, self.ibi_edge_index) * self.nw + self.items_feature * self.sw

        if test:
            BIL_bundles_feature, IL_items_feature2 = self.one_propagate(self.bi_propagate_graph_ori, self.bundles_feature, IL_items_feat2, self.item_level_dropout, test, self.BI_coefs)
        else:
            BIL_bundles_feature, IL_items_feature2 = self.one_propagate(self.bi_propagate_graph, self.bundles_feature, IL_items_feat2, self.item_level_dropout, test, self.BI_coefs)
        # IL_items_feat2 = torch.spmm((self.ibi_asym),IL_items_feature2) * self.nw + IL_items_feature2 * self.sw

        # agg item -> user
        BIL_users_feature = self.get_IL_user_rep(IL_items_feature2, test)

        # w3: 0.2, w4: 0.8
        fuse_bundles_feature = IL_bundles_feature * (1 - self.w3) + BIL_bundles_feature * self.w3
        fuse_users_feature = IL_users_feature * (1 - self.w4) + BIL_users_feature * self.w4

        #  ============================= bundle level propagation =============================
        if test:
            # BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test, self.UB_coefs)
            BL_users_feature, BL_bundles_feature = self.hyper_propagate()
        else:
            BL_users_feature, BL_bundles_feature = self.hyper_propagate()
            # BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test, self.UB_coefs)

        users_feature = [fuse_users_feature, BL_users_feature]
        bundles_feature = [fuse_bundles_feature, BL_bundles_feature]

        return users_feature, bundles_feature
    
    
    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss


    def cal_loss(self, users_feature, bundles_feature):
        # IL: item_level, BL: bundle_level
        # [bs, 1, emb_size]
        IL_users_feature, BL_users_feature = users_feature
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature = bundles_feature
        # [bs, 1+neg_num]
        pred = torch.sum(IL_users_feature * IL_bundles_feature, 2) + torch.sum(BL_users_feature * BL_bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)

        u_cross_view_cl = self.cal_c_loss(IL_users_feature, BL_users_feature)
        b_cross_view_cl = self.cal_c_loss(IL_bundles_feature, BL_bundles_feature)

        c_losses = [u_cross_view_cl, b_cross_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss


    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        users_feature, bundles_feature = self.propagate()

        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)

        return bpr_loss, c_loss


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        return scores
    

class Amatrix(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer=1, dropout=0.0, heads=2, concat=False, self_loop=True, extra_layer=False):
        super(Amatrix, self).__init__()
        self.num_layer = n_layer
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.self_loop = self_loop
        self.extra_layer = extra_layer
        self.convs = nn.ModuleList([AsymMatrix(in_channels=self.in_dim, 
                                              out_channels=self.out_dim, 
                                              dropout=self.dropout,
                                              heads=self.heads,
                                              concat=self.concat,
                                              add_self_loops=self.self_loop,
                                              extra_layer=self.extra_layer) 
                                              for _ in range(self.num_layer)])


    def forward(self, x, edge_index):
        feats = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            feats.append(x)
        feat = torch.stack(feats, dim=1)
        x = torch.mean(feat, dim=1)
        return x