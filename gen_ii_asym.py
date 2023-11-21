import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
from scipy.stats import norm
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph.to_dense()

def get_graph(path, x, y, sep):
    with open(os.path.join(path), 'r') as f:
        b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split(sep)), f.readlines()))

    indice = np.array(b_i_pairs, dtype=np.int32)
    values = np.ones(len(b_i_pairs), dtype=np.float32)
    b_i_graph = sp.coo_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=(x, y)).tocsr()
    return b_i_graph


def save_sp_mat(csr_mat, name):
    sp.save_npz(name, csr_mat)


def load_sp_mat(name):
    return sp.load_npz(name)

def filter(threshold, mat):
    mask = mat >= threshold
    mat = mat * mask
    return mat
def gen_ii_co_oc(ui):
    ii = ui @ ui.T
    return ii    
#calculate similarity matrix
def get_sim_mat(mat):
    num_items = mat.shape[0]
    print(num_items)
    similarity_matrix = torch.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            size_Iu = mat[i][i]  # Calculate |I_u|
            # Calculate similarity based on the equation, handling division by zero
            similarity_matrix[i][j] = mat[i][j] / size_Iu if size_Iu != 0 else 0.0
    return similarity_matrix
  
#get cosine similarity
def gen_ii_cosine(ui):
    ui = torch.nn.functional.normalize(gen_ii_co_oc(ui), p=2, dim=1)
    return ui @ ui.T


def get_sorensen_index(mat):
    num_items = mat.shape[0]
    sorensen_matrix = torch.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            size = mat[i][i] + mat[j][j]  # Calculate |I_u|
            # Calculate similarity based on the equation, handling division by zero
            sorensen_matrix[i][j] = 2 * mat[i][j] / (size) if size != 0 else 0.0
    return sorensen_matrix

def get_asymcos(ui):
    mat = gen_ii_co_oc(ui)
    cosine = gen_ii_cosine(ui)
    asymcos = cosine * get_sim_mat(mat) * get_sorensen_index(mat)
    return asymcos

def gen_ii_asym(ix_mat, threshold=0):
    '''
    mat: ui or bi
    '''
    ii_co = ix_mat @ ix_mat.T
    i_count = ix_mat.sum(axis=1)
    i_count += (i_count == 0) # mask all zero with 1
    # norm_ii = normalize(ii_asym, norm='l1', axis=1)
    
    mask = ii_co > threshold
    ii_co = ii_co.multiply(mask)
    ii_asym = ii_co / i_count
    # normalize by row -> asym matrix
    # return norm_ii
    return ii_asym
    # return ii_co

def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-d", "--dataset", default="Youshu", type=str, help="dataset to train")
    args = parser.parse_args()
    return args

def get_stat(path, sep):
    with open(path, 'r') as f:
        a, b, c = f.readline().split(sep)
    return int(a), int(b), int(c)

paras = get_cmd().__dict__
dataset_name = paras["dataset"]

sep = '\t'

users, bundles, items = get_stat(f'datasets/{dataset_name}/{dataset_name}_data_size.txt', sep=sep)
dir = f'datasets/{dataset_name}'
path = [dir + '/user_bundle_train.txt',
        dir + '/user_item.txt',
        dir + '/bundle_item.txt']

raw_graph = [get_graph(path[0], users, bundles, sep),
                get_graph(path[1], users, items, sep),
                get_graph(path[2], bundles, items, sep)]

ub, ui, bi = raw_graph
# u_i = to_tensor(ui.T)
# print(type(ui))
# sim = cosine_similarity(u_i,dense_output=True)
# print(sim)
# print(sim.shape)
# print(type(sim))
# tensor_similarity_matrix = torch.tensor(sim)
print(gen_ii_asym(ui.T))
print(type(gen_ii_asym(ui.T)))
# print(type(u_i))

# print(get_asymcos(u_i))
# if __name__ == '__main__':
    
#     paras = get_cmd().__dict__
#     dataset_name = paras["dataset"]

#     sep = '\t'

#     users, bundles, items = get_stat(f'datasets/{dataset_name}/{dataset_name}_data_size.txt', sep=sep)
#     dir = f'datasets/{dataset_name}'
#     path = [dir + '/user_bundle_train.txt',
#             dir + '/user_item.txt',
#             dir + '/bundle_item.txt']
    
#     raw_graph = [get_graph(path[0], users, bundles, sep),
#                  get_graph(path[1], users, items, sep),
#                  get_graph(path[2], bundles, items, sep)]

#     ub, ui, bi = raw_graph

#     pbar = tqdm(enumerate([ui.T, bi.T, ub.T, bi]), total = 4, desc="gene", ncols=100)
#     asym_mat = []
#     for i, mat in pbar:
#         asym_mat.append(gen_ii_asym(mat))

#     pbar = tqdm(enumerate(["/iui_cooc.npz", "/ibi_cooc.npz", "/bub_cooc.npz", "/bib_cooc.npz"]), total = 4, desc="save", ncols=100)
#     for i, data in pbar:
#         save_sp_mat(asym_mat[i], dir + data)