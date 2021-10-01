import json
import os
#Change le format des dates.

dates = {}
dir = "/mnt/71A36E2C77574D51/preprocess/tif_2018/"
dir2 = "/mnt/71A36E2C77574D51/preprocess/tif_2019/"
dir3 = "/mnt/71A36E2C77574D51/preprocess/tif_2020/"
files = os.listdir(dir)
files2 = os.listdir(dir2)
files3 = os.listdir(dir3)
p = 0
for f in files:
    dates[p] = f[:-4]
    p += 1
for f in files2:
    dates[p] = f[:-4]
    p += 1

for f in files3:
    dates[p] = f[:-4]
    p += 1

with open('/home/FQuinton/Bureau/data_pse/META/dates.json',
          'w') as file:
    file.write(json.dumps(dates, indent=4))