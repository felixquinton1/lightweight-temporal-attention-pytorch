#Permet d'obtenir des statistiques sur la taille et le nombre de parcelles.

import json
labels = "/home/FQuinton/Bureau/data_pse/META/labels.json"
sizes = "/home/FQuinton/Bureau/data_pse/META/sizes.json"
nb_apparition_2018 = {'nb_parcelles': 0}
nb_apparition_2019 = {}
nb_apparition_2020 = {}
nb_apparition_tot = {}
small_parcels = {}
with open(labels) as f:
    data = json.load(f)
    for key, value in data['CODE9_2018'].items():
        nb_apparition_2018['nb_parcelles'] += 1
        if value in nb_apparition_tot:
            nb_apparition_tot[value] += 1
        else:
            nb_apparition_tot[value] = 0
        if value in nb_apparition_2018:
            nb_apparition_2018[value] += 1
        else:
            nb_apparition_2018[value] = 0

    for key, value in data['CODE9_2019'].items():
        if value in nb_apparition_tot:
            nb_apparition_tot[value] += 1
        else:
            nb_apparition_tot[value] = 0
        if value in nb_apparition_2019:
            nb_apparition_2019[value] += 1
        else:
            nb_apparition_2019[value] = 0

    for key, value in data['CODE9_2020'].items():
        if value in nb_apparition_tot:
            nb_apparition_tot[value] += 1
        else:
            nb_apparition_tot[value] = 0
        if value in nb_apparition_2020:
            nb_apparition_2020[value] += 1
        else:
            nb_apparition_2020[value] = 0

with open('/home/FQuinton/Bureau/data_pse/META/stats/nb_apparition_2018.json',
          'w') as file:
    file.write(json.dumps(nb_apparition_2018, indent=4))
with open('/home/FQuinton/Bureau/data_pse/META/stats/nb_apparition_2019.json',
          'w') as file:
    file.write(json.dumps(nb_apparition_2019, indent=4))
with open('/home/FQuinton/Bureau/data_pse/META/stats/nb_apparition_2020.json',
          'w') as file:
    file.write(json.dumps(nb_apparition_2020, indent=4))

with open('/home/FQuinton/Bureau/data_pse/META/nb_apparition_tot.json',
          'w') as file:
    file.write(json.dumps(nb_apparition_tot, indent=4))

with open(sizes) as f:
    data = json.load(f)
    for key, value in data.items():
        if value < 20:
            small_parcels[key] = value

with open('/home/FQuinton/Bureau/data_pse/META/stats/small_parcels.json',
          'w') as file:
    file.write(json.dumps(small_parcels, indent=4))
