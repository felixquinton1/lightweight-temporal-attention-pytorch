import json
import numpy  as np
from tqdm import tqdm
import os
labels = "/home/FQuinton/Bureau/data_pse/META/labels.json"

pid = os.listdir('/home/FQuinton/Bureau/labels_embeddings/data_pred_global/2018/')
np.random.seed(1)
a = np.array(range(103602))

np.random.shuffle(a)
test_id = [[] for i in range(5)]
test_id[0] = a[20721:]
test_id[1] = np.concatenate((a[0:20721], a[41443:]))
test_id[2] = np.concatenate((a[0:41443], a[62163:]))
test_id[3] = np.concatenate((a[0:62163], a[82883:]))
test_id[4] = a[0:82883]


labels_int = {'CODE9_2018':{},
                'CODE9_2019':{},
                'CODE9_2020':{},
              }
m2018 = np.zeros(20, dtype = int)
m2019 = np.zeros(20, dtype = int)
m2020 = np.zeros(20, dtype = int)
test = np.zeros(20, dtype = int)

with open(labels) as f:
    data = json.load(f)
    # for key, value in data['CODE9_2018'].items():
    #     m2018[value] += 1
    # for key, value in data['CODE9_2019'].items():
    #     m2019[value] += 1
    # for key, value in data['CODE9_2020'].items():
    #     m2020[value] += 1


    for i in tqdm(range(5)):
        mtot = np.zeros((20, 20, 20), dtype=int)
        test = np.zeros(20, dtype=int)
        for j in test_id[i]:
            # for key, value in data['CODE9_2018'].items():
            a = data['CODE9_2018'][pid[j][:-4]]
            # if key in data['CODE9_2019'] and key in data['CODE9_2020']:
            test[a] += 1
            b = data['CODE9_2019'][pid[j][:-4]]
            test[b] += 1
            c = data['CODE9_2020'][pid[j][:-4]]
            test[c] += 1
            mtot[a][b][c] += 1
        np.save('/home/FQuinton/Bureau/data_pse/META/transition_matrix_' + str(i) + '.npy', mtot)
print(test)
print(np.sum(test))

