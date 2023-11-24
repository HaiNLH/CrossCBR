import numpy as np
import scipy.sparse as sp
import torch

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph

# Example COO matrix
data = [1, 0, 1, 1, 0]
row_indices = [0 , 0, 1, 2, 2]
col_indices = [1, 2, 0, 1, 2]

coo_graph = sp.coo_matrix((data, (row_indices, col_indices)), shape=(3, 3))
i_count = coo_graph.sum(axis=1)
print(i_count)
# Convert COO matrix to PyTorch sparse tensor
# tensor_graph = to_tensor(coo_graph).to_dense()
test_graph = coo_graph@coo_graph.T
print(coo_graph.todense())
print(test_graph.todense())

