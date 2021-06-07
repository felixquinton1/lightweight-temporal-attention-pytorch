import numpy as np
import os
import pickle as pkl
dir = '/home/FQuinton/Bureau/donnees_pse/DATA/2018/'
files = os.listdir(dir)
avg = []
l_pixels = 0
for f in files:
    dates = np.load('/home/FQuinton/Bureau/donnees_pse/DATA/2018/' + f)
    for date in dates:
        for idx, band in enumerate(date):
            for pixel in band:
                l_pixels += 1
                avg[idx] += pixel
for band in avg:
    band = band / (l_pixels)

