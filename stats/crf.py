import os
import json
from tqdm import tqdm
import numpy as np
pid = os.listdir('/home/FQuinton/Bureau/data_embedding_2020/DATA/Fold1/2020/')
file = "/home/FQuinton/Bureau/data_pse/META/labels.json"
transition_matrix = np.load('/home/FQuinton/Bureau/data_pse/META/stats/transition_matrix.npy')
mat = np.zeros((20,20), dtype=int)
with open(file) as f:
    data = json.load(f)
    for id in tqdm(pid):
        classe_2018 = data['CODE9_2018'][id[:-4]]
        classe_2019 = data['CODE9_2019'][id[:-4]]
        classe_2020 = data['CODE9_2020'][id[:-4]]

        x = np.load('/home/FQuinton/Bureau/data_pred_labels_0_padding/2020/' + id)
        sequence = transition_matrix[classe_2018][classe_2019]
        for i in range(20):
            if sequence[i] == 0:
                sequence[i] = 1
        sequence_prob = sequence / np.sum(sequence)
        new_prob = x * sequence_prob
        pred = np.argmax(new_prob)
        mat[classe_2020][pred] += 1
np.save('/home/FQuinton/Bureau/data_pse/META/stats/confusion_matrix_crf.npy', mat)