import json
import os

#Retire les parcelles correspondants à certains critères

occurences = "C:/Users/felix/OneDrive/Bureau/test/out/dataset_preparation/META/stats/nb_apparition_2018.json"
small_parcels = "C:/Users/felix/OneDrive/Bureau/test/out/dataset_preparation/META/stats/small_parcels.json"
labels = "C:/Users/felix/OneDrive/Bureau/test/out/dataset_preparation/META/labels.json"
labels_str = "/home/FQuinton/Bureau/data_pse/META/labels_str.json"
path = "/home/FQuinton/Bureau/data_pse/DATA/"
move_to = "/home/FQuinton/Bureau/unused_data/"
occ_min = 100
parcelles_min = 50
removed_labels = "NO_EX_SUR"

with open(labels_str) as f:
    data = json.load(f)
    for key, value in data['CODE9_2018'].items():
        if value == removed_labels:
            if os.path.exists(path + "2018/" + key + ".npy"):
                os.rename(path + "2018/" + key + ".npy", move_to + "2018/" + key + ".npy")
            if os.path.exists(path + "2019/" + key + ".npy"):
                os.rename(path + "2019/" + key + ".npy", move_to + "2019/" + key + ".npy")
            if os.path.exists(path + "2020/" + key + ".npy"):
                os.rename(path + "2020/" + key + ".npy", move_to + "2020/" + key + ".npy")
    for key, value in data['CODE9_2019'].items():
        if value == removed_labels:
            if os.path.exists(path + "2018/" + key + ".npy"):
                os.rename(path + "2018/" + key + ".npy", move_to + "2018/" + key + ".npy")
            if os.path.exists(path + "2019/" + key + ".npy"):
                os.rename(path + "2019/" + key + ".npy", move_to + "2019/" + key + ".npy")
            if os.path.exists(path + "2020/" + key + ".npy"):
                os.rename(path + "2020/" + key + ".npy", move_to + "2020/" + key + ".npy")
    for key, value in data['CODE9_2020'].items():
        if value == removed_labels:
            if os.path.exists(path + "2018/" + key + ".npy"):
                os.rename(path + "2018/" + key + ".npy", move_to + "2018/" + key + ".npy")
            if os.path.exists(path + "2019/" + key + ".npy"):
                os.rename(path + "2019/" + key + ".npy", move_to + "2019/" + key + ".npy")
            if os.path.exists(path + "2020/" + key + ".npy"):
                os.rename(path + "2020/" + key + ".npy", move_to + "2020/" + key + ".npy")

