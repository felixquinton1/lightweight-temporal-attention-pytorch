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

mat_copy = copy.deepcopy(mat)
compteur = torch.zeros(20, dtype=int)
nb_rot = torch.zeros(20, dtype=int)
# for i in range(20):
#     mat_copy[i][i][i] = 0
somme = torch.sum(mat_copy, axis=(1, 2))
somme2 = torch.sum(mat, axis=(1, 2))
for i in range(20):
    while compteur[i]/somme[i] < 0.90:
        val = torch.where(mat_copy[i] == torch.amax(mat_copy[i]))
        compteur[i] += mat_copy[i][val[0][0]][val[1][0]]
        nb_rot[i] += 1
        mat_copy[i][val[0][0]][val[1][0]] = -1
nb_rot2 = torch.cat((nb_rot[1:8], nb_rot[9:19]))
print(nb_rot)
print(np.mean(nb_rot.tolist()))
print(np.std(nb_rot.tolist()))
tot = 0
for i in range(20):
    for j in range(20):
        for k in range(20):
            if mat[i][j][k] != 0:
                tot +=1
# print(tot)

# print('i')