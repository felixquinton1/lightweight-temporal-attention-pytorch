import os
import json
from tqdm import tqdm
import numpy as np
pid = os.listdir('/home/FQuinton/Bureau/labels_embeddings/data_pred_global/2020/')
file = "/home/FQuinton/Bureau/data_pse/META/labels.json"
transition_matrix1 = np.load('/home/FQuinton/Bureau/data_pse/META/transition_matrix_0.npy')
transition_matrix2 = np.load('/home/FQuinton/Bureau/data_pse/META/transition_matrix_1.npy')
transition_matrix3 = np.load('/home/FQuinton/Bureau/data_pse/META/transition_matrix_2.npy')
transition_matrix4 = np.load('/home/FQuinton/Bureau/data_pse/META/transition_matrix_3.npy')
transition_matrix5 = np.load('/home/FQuinton/Bureau/data_pse/META/transition_matrix_4.npy')
np.random.seed(1)
np.random.shuffle(pid)


mat = np.zeros((20,20), dtype=int)
with open(file) as f:
    data = json.load(f)
    p=0
    for id in tqdm(pid):
        classe_2018 = data['CODE9_2018'][id[:-4]]
        classe_2019 = data['CODE9_2019'][id[:-4]]
        classe_2020 = data['CODE9_2020'][id[:-4]]

        x = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_global/2020/' + id)
        if p<20721 :
            sequence = transition_matrix1[classe_2018][classe_2019] + 1
        elif p<41442:
            sequence = transition_matrix2[classe_2018][classe_2019] + 1
        elif p<62162:
            sequence = transition_matrix3[classe_2018][classe_2019] + 1
        elif p<82882:
            sequence = transition_matrix4[classe_2018][classe_2019] + 1
        else:
            sequence = transition_matrix5[classe_2018][classe_2019] + 1

        sequence_prob = sequence / np.sum(sequence)
        new_prob = x * sequence_prob
        pred = np.argmax(new_prob)
        mat[classe_2020][pred] += 1
np.save('/home/FQuinton/Bureau/data_pse/META/stats/confusion_matrix_crf2.npy', mat)