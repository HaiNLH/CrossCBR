import torch

import torch.nn.functional as F
#user item matrix
ui = torch.tensor([[1, 1, 1, 0, 1], [0, 1, 1, 0, 0],[1,0,0,1, 0],[1,1,1,1, 0],[0,1,0,1, 0],[0,0,0,1, 0]]).float()
#get cooc ii matrix from user
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
def get_sorensen_index(mat):
    num_items = mat.shape[0]
    sorensen_matrix = torch.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            size = mat[i][i] + mat[j][j]  # Calculate |I_u|
            # Calculate similarity based on the equation, handling division by zero
            sorensen_matrix[i][j] = 2 * mat[i][j] / (size) if size != 0 else 0.0
    return sorensen_matrix
#get cosine similarity
def gen_ii_cosine(ui):
    ui = F.normalize(gen_ii_co_oc(ui), p=2, dim=1)
    return ui @ ui.T

def get_asymcos(ui):
    mat = gen_ii_co_oc(ui)
    cosine = gen_ii_cosine(ui)
    asymcos = cosine * get_sim_mat(mat) * get_sorensen_index(mat)
    return asymcos
print(gen_ii_cosine(ui.T).size())

print(get_asymcos(ui.T))

