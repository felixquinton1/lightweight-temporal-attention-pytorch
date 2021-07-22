import json
import os
import numpy as np

input_path = "/home/FQuinton/Bureau/data_embedding2/DATA/Fold5"
annee = '2020'
output_path = "/home/FQuinton/Bureau/data_embedding2/DATA/Fold5/all"


for filename in os.listdir(os.path.join(input_path,annee)):
    a = np.load(os.path.join(input_path, annee, filename))
    b = np.load(os.path.join(input_path, '2018', filename))
    c = np.load(os.path.join(input_path, '2019', filename))

    d = np.concatenate((a, ((b + c )/ 2)))
    np.save(os.path.join(output_path, str(filename)), d)
    pass