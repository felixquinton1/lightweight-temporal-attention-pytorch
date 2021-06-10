import json
import os
occurences = "C:/Users/felix/OneDrive/Bureau/test/out/dataset_preparation/META/stats/nb_apparition_2018.json"
small_parcels = "C:/Users/felix/OneDrive/Bureau/test/out/dataset_preparation/META/stats/small_parcels.json"
labels = "C:/Users/felix/OneDrive/Bureau/test/out/dataset_preparation/META/labels.json"
labels_str = "/home/FQuinton/Bureau/data_pse/META/labels_str.json"
path = "/home/FQuinton/Bureau/data_pse/DATA/"
move_to = "/home/FQuinton/Bureau/unused_data/"
occ_min = 100
parcelles_min = 50
removed_labels = "NO_EX_SUR"
# with open(occurences) as f:
#     data = json.load(f)
#     with open(labels) as g:
#         data2 = json.load(g)
#         for key, value in data.items():
#             if value < occ_min:
#                 for key2, value2 in data2["CODE9_2018"].items():
#                     if value2 == key:
#                         if os.path.exists(path + "2018/" + key2 + ".npy"):
#                             os.remove(path + "2018/" + key2 + ".npy")
#                         if os.path.exists(path + "2019/" + key2 + ".npy"):
#                             os.remove(path + "2019/" + key2 + ".npy")
#                         if os.path.exists(path + "2020/" + key2 + ".npy"):
#                             os.remove(path + "2020/" + key2 + ".npy")
#
#                 for key2, value2 in data2["CODE9_2019"].items():
#                     if value2 == key:
#                         if os.path.exists(path + "2018/" + key2 + ".npy"):
#                             os.remove(path + "2018/" + key2 + ".npy")
#                         if os.path.exists(path + "2019/" + key2 + ".npy"):
#                             os.remove(path + "2019/" + key2 + ".npy")
#                         if os.path.exists(path + "2020/" + key2 + ".npy"):
#                             os.remove(path + "2020/" + key2 + ".npy")
#
#                 for key2, value2 in data2["CODE9_2020"].items():
#                     if value2 == key:
#                         if os.path.exists(path + "2018/" + key2 + ".npy"):
#                             os.remove(path + "2018/" + key2 + ".npy")
#                         if os.path.exists(path + "2019/" + key2 + ".npy"):
#                             os.remove(path + "2019/" + key2 + ".npy")
#                         if os.path.exists(path + "2020/" + key2 + ".npy"):
#                             os.remove(path + "2020/" + key2 + ".npy")

# with open(small_parcels) as f:
#     data = json.load(f)
#     for key, value in data.items():
#         if value < parcelles_min:
#             if os.path.exists(path + "2018/" + key + ".npy"):
#                 os.remove(path + "2018/" + key + ".npy")
#             if os.path.exists(path + "2019/" + key + ".npy"):
#                 os.remove(path + "2019/" + key + ".npy")
#             if os.path.exists(path + "2020/" + key + ".npy"):
#                 os.remove(path + "2020/" + key + ".npy")

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

