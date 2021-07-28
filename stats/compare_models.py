import json

file = '/home/FQuinton/Bureau/data_pse/META/stats/compare_pred_value.json'
# with open(file) as f:
#     data = json.load(f)
#     rot = 0
#     no_rot = 0
#
#     rot_glob = 0
#     no_rot_glob = 0
#
#     rot_false = 0
#     no_rot_false = 0
#
#     tot = 0
#     tot_lab_true = 0
#     tot_global_true = 0
#     tot_false = 0
#     for key,value in data.items():
#         tot += 1
#         if value['classe_2020'] == value['pred_lab']:
#             tot_lab_true += 1
#             if value['classe_2018'] == value['classe_2019'] and value['classe_2018'] == value['classe_2020']:
#                 no_rot += 1
#             else:
#                 rot += 1
#         elif value['classe_2020'] == value['pred_global']:
#             tot_global_true += 1
#             if value['classe_2018'] == value['classe_2019'] and value['classe_2018'] == value['classe_2020']:
#                 no_rot_glob += 1
#             else:
#                 rot_glob += 1
#         else:
#             tot_false +=1
#             if value['classe_2018'] == value['classe_2019'] and value['classe_2018'] == value['classe_2020']:
#                 no_rot_false += 1
#             else:
#                 rot_false += 1
#
#
#     print('Tot: ' + str(tot))
#     print('label True: ')
#     print('tot_lab_true: ' + str(tot_lab_true))
#     print('Rot: ' + str(rot))
#     print('No rot: ' + str(no_rot))
#
#     print('global True: ')
#     print('tot_lab_true: ' + str(tot_global_true))
#     print('Rot: ' + str(rot_glob))
#     print('No rot: ' + str(no_rot_glob))
#
#     print('Both false: ' + str(tot_false))
#     print('Rot: ' + str(rot_false))
#     print('No rot: ' + str(no_rot_false))
#


with open(file) as f:
    data = json.load(f)
    moyenne = 0
    compteur = 0

    compteur_no_rot = 0
    moyenne_no_rot = 0

    compteur_rot = 0
    moyenne_rot = 0

    for key,value in data.items():
        if value['pred_global_max'] != value['classe_2020']:
            compteur += 1
            moyenne += value['pred_lab'] - value['pred_global']
            if value['classe_2018'] == value['classe_2019'] and value['classe_2018'] == value['classe_2020']:
                compteur_no_rot += 1
                moyenne_no_rot += value['pred_lab'] - value['pred_global']
            else:
                compteur_rot += 1
                moyenne_rot += value['pred_lab'] - value['pred_global']
    moyenne = moyenne/compteur
    moyenne_no_rot = moyenne_no_rot/compteur_no_rot
    moyenne_rot = moyenne_rot/compteur_rot
print('tot: ')
print(moyenne)
print(compteur)

print('no_rot')
print(moyenne_no_rot)
print(compteur_no_rot)

print('rot')
print(moyenne_rot)
print(compteur_rot)