import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

#Permet d'obtenir un diagramme de hinton

classes = ["Meadow", "Triticale", "Maize", "Rye", "Wheat", "Rape", "Barley Winter", "Sunflower", "Vineyard",
         "Soybean", "Sorghum", "Alfalfa", "Oat Winter", "Leguminous", "Mix cereals", "Flo. Fru. Veg.",
         "Oat Summer", "Potato", "Barley Summer", "Wood Pasture"]

mat = pkl.load(open('/home/FQuinton/Bureau/lightweight-temporal-attention-pytorch/models_saved/labels_0_padding/overall/2020_conf_mat.pkl','rb'))

max_weight = 2 ** np.ceil(np.log2(np.abs(mat).max()))


fig, ax = plt.subplots()
ax.patch.set_facecolor('gray')
# We want to show all ticks...
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
# ... and label them with the respective list entries
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for (x, y), w in np.ndenumerate(mat):
    color = 'white' if w > 0 else 'black'
    size = np.power((abs(w) / max_weight), 1 / 4)
    rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                         facecolor=color, edgecolor=color)
    ax.add_patch(rect)
ax.autoscale_view()
ax.invert_yaxis()
fig.tight_layout()
plt.show()

