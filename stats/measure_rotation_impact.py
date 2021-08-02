
import numpy as np
import json
import os
from tqdm import tqdm
np.random.seed(1)
dic = {
    0: "Prairie",
    1: "Triticale",
    2: "Mais",
    3: "Seigle",
    4: "Ble",
    5: "Colza",
    6: "Orge W",
    7: "Tournesol",
    8: "Vigne",
    9: "Soja",
    10: "Sorghum",
    11: "Luzerne",
    12: "Avoine W",
    13: "Legume_fourr",
    14: "Cereales_mixtes",
    15: "Fleurs_fruits_legumes",
    16: "Avoine_S",
    17: "Pomme_de_terre",
    18: "Orge_S",
    19: "Paturage_boise"
}
pid = os.listdir('/home/FQuinton/Bureau/labels_embeddings/data_pred_global/2018/')
file = "/home/FQuinton/Bureau/data_pse/META/labels.json"
years = ['2018', '2019', '2020']
js = {}

with open(file) as f:
    data = json.load(f)
    for id in tqdm(pid):
        x = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_labels_0_padding/2020/' + id)
        y = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_global/2020/' + id)
        z = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_labels_only/2020/' + id)
        classe_2018 = data['CODE9_2018'][id[:-4]]
        classe_2019 = data['CODE9_2019'][id[:-4]]
        classe_2020 = data['CODE9_2020'][id[:-4]]
        pred_classe_x = dic[np.argmax(x)]
        pred_classe_y = dic[np.argmax(y)]
        pred_classe_z = dic[np.argmax(z)]
        pred_x = x[classe_2020]
        pred_y = y[classe_2020]
        js[id[:-4]] = {'classe_2018': dic[classe_2018],
                  'classe_2019' : dic[classe_2019],
                  'classe_2020' : dic[classe_2020],
                  'pred_lab' : pred_x,
                  'pred_lab_max': pred_classe_x,
                  'pred_global_max': pred_classe_y,
                  'pred_struct_max': pred_classe_z,
                  'pred_global': pred_y}


with open('/home/FQuinton/Bureau/data_pse/META/stats/compare_pred_value.json',
          'w') as file:
    file.write(json.dumps(js, indent=4))