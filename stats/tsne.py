import os

import numpy as np
from sklearn.manifold import TSNE
import json
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(11.7,8.27)})

np.random.seed(1)
indices = np.array(range(103602))
np.random.shuffle(indices)
test_list = indices[:20721]
years = ['2018','2019','2020']
# pid = os.listdir('/home/FQuinton/Bureau/data_embedding_labels_0_padding/DATA/Fold1/2018/')
pid = os.listdir('/home/FQuinton/Bureau/data_embedding_2020/DATA/Fold1/2018/')
test_pid = [pid[i] for i in test_list]



file = "/home/FQuinton/Bureau/data_pse/META/labels.json"
L = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# L = [0,1,2,3,4,5,6,7,8,9]
# L = [10,11,12,13,14,15,16,17,18,19]
# L = [0,19]
palette = sns.color_palette("gist_rainbow", n_colors=len(L))
s = {}
l = [0 for i in range(20)]
X = []
Y = []
style = []
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
with open(file) as f:
    data = json.load(f)

    for year in years:
        for file in test_pid:
            if data['CODE9_' + year][file[:-4]] in L:
                classe = data['CODE9_' + year][file[:-4]]
                if l[classe] < 200 :
                    x = np.load('/home/FQuinton/Bureau/data_embedding_2020/DATA/Fold1/' + year + '/{}.npy'.format(file[:-4]))
                    X.append(x)
                    Y.append(dic[classe])
                    style.append(year)
                    l[classe] += 1
        l = [0 for i in range(20)]

X = np.array(X)
X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded.shape

sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=Y, style=style, legend='full')
# sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=Y, style=style, palette=palette)
# Put the legend out of the figure
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

