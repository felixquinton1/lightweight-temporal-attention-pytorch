
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
        x_2020 = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_labels_0_padding/2020/' + id)
        x_2019 = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_labels_0_padding/2019/' + id)
        x_2018 = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_labels_0_padding/2018/' + id)
        y_2020 = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_global/2020/' + id)
        y_2019 = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_global/2020/' + id)
        y_2018 = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_global/2020/' + id)
        z = np.load('/home/FQuinton/Bureau/labels_embeddings/data_pred_labels_only/2020/' + id)

        classe_2018 = data['CODE9_2018'][id[:-4]]
        classe_2019 = data['CODE9_2019'][id[:-4]]
        classe_2020 = data['CODE9_2020'][id[:-4]]

        pred_classe_x_2018 = dic[np.argmax(x_2018)]
        pred_classe_x_2019 = dic[np.argmax(x_2019)]
        pred_classe_x_2020 = dic[np.argmax(x_2020)]

        pred_classe_y_2018 = dic[np.argmax(y_2018)]
        pred_classe_y_2019 = dic[np.argmax(y_2019)]
        pred_classe_y_2020 = dic[np.argmax(y_2020)]

        pred_classe_z = dic[np.argmax(z)]
        pred_x_2020 = x_2020[classe_2020]
        pred_y_2020 = y_2020[classe_2020]
        js[id[:-4]] = {'classe_2018': dic[classe_2018],
                  'classe_2019' : dic[classe_2019],
                  'classe_2020' : dic[classe_2020],
                  'pred_lab_max_2018': pred_classe_x_2018,
                  'pred_lab_max_2019': pred_classe_x_2019,
                  'pred_lab_max_2020': pred_classe_x_2020,
                  'pred_lab_2020': pred_x_2020,
                  'pred_global_max_2018': pred_classe_y_2018,
                  'pred_global_max_2019': pred_classe_y_2019,
                  'pred_global_max_2020': pred_classe_y_2020,
                  'pred_global_2020': pred_y_2020,
                  'pred_struct_max': pred_classe_z
                       }


with open('/home/FQuinton/Bureau/data_pse/META/stats/compare_pred_value2.json',
          'w') as file:
    file.write(json.dumps(js, indent=4))