import json
import numpy  as np
labels = "/home/FQuinton/Bureau/data_pse/META/labels.json"
np.random.seed(1)
a = np.array(range(103602))
np.random.shuffle(a)

labels_int = {'CODE9_2018':{},
                'CODE9_2019':{},
                'CODE9_2020':{},
              }
m2018 = np.zeros(20, dtype = int)
m2019 = np.zeros(20, dtype = int)
m2020 = np.zeros(20, dtype = int)
test = np.zeros(20, dtype = int)
mtot = np.zeros((20,20,20), dtype= int)
with open(labels) as f:
    data = json.load(f)
    # for key, value in data['CODE9_2018'].items():
    #     m2018[value] += 1
    # for key, value in data['CODE9_2019'].items():
    #     m2019[value] += 1
    # for key, value in data['CODE9_2020'].items():
    #     m2020[value] += 1


    for key, value in data['CODE9_2018'].items():
        a = value
        if key in data['CODE9_2019'] and key in data['CODE9_2020']:
            test[a] += 1
            b = data['CODE9_2019'][key]
            test[b] += 1
            c = data['CODE9_2020'][key]
            test[c] += 1
            mtot[a][b][c] += 1
print(test)
np.save('/home/FQuinton/Bureau/data_pse/META/transition_matrix.npy', mtot)