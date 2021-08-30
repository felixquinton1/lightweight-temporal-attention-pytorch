import copy

import numpy as np
import torch
mat = torch.Tensor(np.load('/home/FQuinton/Bureau/data_pse/META/transition_matrix.npy')).int()

dic = {
    0: "Prairie",
    1: "Triticale",
    2: "Maïs",
    3: "Seigle",
    4: "Blé",
    5: "Colza",
    6: "Orge W",
    7: "Tournesol",
    8: "Vigne",
    9: "Soja",
    10: "Sorghum",
    11: "Luzerne",
    12: "Avoine W",
    13: "Légume fourr",
    14: "Céréales mixtes",
    15: "Fleurs fruits legumes",
    16: "Avoine S",
    17: "Pomme de terre",
    18: "Orge S",
    19: "Paturage boisé"
}

mat_norm = torch.zeros((20, 20, 20))
for i in range(20):
    for j in range(20):
        for k in range(20):
            if torch.sum(mat[i][j]) != 0:
                mat_norm[i][j][k] = mat[i][j][k]/torch.sum(mat[i][j])
            else :
                mat_norm[i][j][k] = 0
max_list = []
mat_copy = copy.deepcopy(mat_norm)
for i in range(500):
    val = torch.where(mat_copy == torch.amax(mat_copy))
    mat_copy[val[0][0]][val[1][0]][val[2][0]] = -1
    if(torch.sum(mat[val[0][0]][val[1][0]]).item() > 9 and round(mat_norm[val[0][0]][val[1][0]][val[2][0]].item(),2) > 0.1):
        max_list.append([round(mat_norm[val[0][0]][val[1][0]][val[2][0]].item(), 2),
                         torch.sum(mat[val[0][0]][val[1][0]]).item(),
                         mat[val[0][0]][val[1][0]][val[2][0]].item(),
                         dic[val[0][0].item()],
                         dic[val[1][0].item()],
                         dic[val[2][0].item()]])
max_list.sort(reverse = True)
m = np.array(max_list)
m = m[:, 2].astype(int)
n = np.sum(m)
pass