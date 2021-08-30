import json
import numpy as np
file = '/home/FQuinton/Bureau/data_pse/META/stats/compare_pred_value.json'

dic = {
    "Prairie": 0,
    "Triticale": 1,
    "Mais": 2,
    "Seigle": 3,
    "Ble": 4,
    "Colza": 5,
    "Orge W": 6,
    "Tournesol": 7,
    "Vigne": 8,
    "Soja": 9,
    "Sorghum": 10,
    "Luzerne": 11,
    "Avoine W": 12,
    "Legume_fourr": 13,
    "Cereales_mixtes": 14,
    "Fleurs_fruits_legumes": 15,
    "Avoine_S": 16,
    "Pomme_de_terre": 17,
    "Orge_S": 18,
    "Paturage_boise": 19
}
with open(file) as f:
    data = json.load(f)
    moyenne = np.zeros(20)
    moyenne_glob = np.zeros(20)
    moyenne_lab = np.zeros(20)
    compteur = np.zeros(20, dtype=int)
    compteur2 = np.zeros(20, dtype=int)
    compteur_no_rot = 0
    moyenne_no_rot = 0

    compteur_rot = 0
    moyenne_rot = 0
    m = 0
    t = 0


    med = [[] for i in range(20)]
    med_all = [[] for i in range(20)]
    moyenne_dif = 0
    mean = np.zeros(20)
    for key,value in data.items():
        compteur2[dic[value['classe_2020']]] += 1

        # if value['pred_struct_max'] == value['classe_2020']:
        #     if value['pred_global_max'] != value['classe_2020'] or value['pred_lab_max'] != value['classe_2020'] :
        compteur[dic[value['classe_2020']]] += 1
        moyenne[dic[value['classe_2020']]] += value['pred_lab'] - value['pred_global']
        moyenne_glob[dic[value['classe_2020']]] += value['pred_global']
        moyenne_lab[dic[value['classe_2020']]] += value['pred_lab']
        m += value['pred_lab'] - value['pred_global']
        t += 1
        med[dic[value['classe_2020']]].append(value['pred_lab'] - value['pred_global'])
        med_all[dic[value['classe_2020']]].append(value['pred_global'])
        if value['classe_2018'] == value['classe_2019'] and value['classe_2018'] == value['classe_2020']:
            compteur_no_rot += 1
            moyenne_no_rot += value['pred_lab'] - value['pred_global']
        else:
            compteur_rot += 1
            moyenne_rot += value['pred_lab'] - value['pred_global']
    for i in range(20):
        med[i] = np.median(med[i])
        med_all[i] = np.median(med_all[i])
    moyenne = moyenne/compteur
    moyenne_glob = moyenne_glob/compteur
    moyenne_lab = moyenne_lab/compteur
    moyenne_no_rot = moyenne_no_rot/compteur_no_rot
    moyenne_rot = moyenne_rot/compteur_rot
    m = m/t

# print('med')
# print(med)
# print('med all')
# print(med_all)
# print("ratio")
# print(np.array(med)/(1 - np.array(med_all)))
#
# print(np.nanmean(np.array(med)/(1 - np.array(med_all))))


print('tot: ')
print(moyenne)
# print(np.nanmean(moyenne))
#
# print('moyenne labels')
# print(moyenne_lab)
#
print('moyenne global')
print(moyenne_glob)
#
# print('ratio')
print(moyenne/(1-moyenne_glob))
print(np.nanmean(moyenne/(1-moyenne_glob)))
#
# print(compteur/compteur2)

#
# print('no_rot')
# print(moyenne_no_rot)
# print(compteur_no_rot)
#
# print('rot')
# print(moyenne_rot)
# print(compteur_rot)

# print('m')
# print(m)
# print(t)