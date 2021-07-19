import json
import os
import numpy as np

input_path = "/home/FQuinton/Bureau/data_embedding/META"
output_path = "/home/FQuinton/Bureau/data_embedding/DATA/2020"
json_by_parcel = {}
json_2018_2019 = {}

for filename in os.listdir(input_path):
    with open(os.path.join(input_path, filename)) as f:
        data = json.load(f)
        for key, value in data.items():
            if key[-4:] == '2020':
                json_by_parcel[key[:-5]] = value
            elif key[:-5] in json_2018_2019.keys():
                json_2018_2019[key[:-5]] = np.mean(np.array([json_2018_2019[key[:-5]], value]), axis=0)
            else :
                json_2018_2019[key[:-5]] = value

for key, value in json_by_parcel.items():
    json_by_parcel[key] += json_2018_2019[key].tolist()
    np.save(os.path.join(output_path, str(key)), value)