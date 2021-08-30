import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
pid = os.listdir('/home/FQuinton/Bureau/labels_embeddings/test/2020/')
file = "/home/FQuinton/Bureau/data_pse/META/labels.json"
years = ['2018','2019','2020']
# years = ['2020']
# l = 2072040
# l = 103602
m = 101
l = 6216120
a = np.arange(m, dtype=int)
b = np.zeros(m)
c = np.zeros(m)
acc = np.zeros(m)
conf = np.zeros(m)
bm = np.zeros(m)
with open(file) as f:
    data = json.load(f)
    for year in years:
        for id in tqdm(pid):
            x = np.load('/home/FQuinton/Bureau/labels_embeddings/test/' + year + '/' + id)
            # x = x * 100
            val = data['CODE9_' + year][id[:-4]]
            for i,v in enumerate(x):
                if(i == val):
                    acc[int(v*100)] += 1
                bm[int(v*100)] += 1
                conf[int(v*100)] += v
                b[int(v*100)] += 1
            c[int(x[val]*100)] += 1

acc = acc/bm
conf = conf/bm

ece = 0
for i in range(m):
    ece += (bm[i]/l) * np.absolute(acc[i] - conf[i])


# x = x.astype(int)
            # val = data['CODE9_' + year][id[:-4]]
            # for i in x:
            #     b[i] += 1
            # c[x[val]] += 1
plt.scatter(a, c/b)
print(ece*100)
# print(np.mean(np.absolute(a - c/b*100)))
# print(np.std(np.absolute(a - c/b*100)))
plt.plot([0,100], [0,100], linestyle='dashed', color='#bfb5aa', lw=2)
plt.bar(a, a, align='center', alpha=0.5, label='écart', color='#FF9966')
plt.bar(a, c/b*100, align='center', alpha=0.5, label='sortie', color='#21a7dd')
# plt.text(60, 4.1, 'erreur moyenne:' + erreur, fontsize=15,  color='red')

plt.legend()

plt.xlabel('Probabilité')
plt.ylabel('Taux de bonne classification')

plt.show()
