import numpy as np
import os
import pickle as pkl
from tqdm import tqdm

dir = ['/home/FQuinton/Bureau/donnees_pse/DATA/2018/', '/home/FQuinton/Bureau/donnees_pse/DATA/2019/',
       '/home/FQuinton/Bureau/donnees_pse/DATA/2020/']

avg_tot = [0,0,0,0,0,0,0,0,0,0]
avg_tot_square = [0,0,0,0,0,0,0,0,0,0]
l_pixels_tot = 0

for i in range(len(dir)):
    files = os.listdir(dir[i])
    avg = [0,0,0,0,0,0,0,0,0,0]
    avg_square = [0,0,0,0,0,0,0,0,0,0]
    l_pixels = 0
    for f in tqdm(files):
        data = np.load(dir[i] + f)
        sum_file = np.sum(data, axis=(0, 2))
        avg = np.add(avg,sum_file)
        avg_tot = np.add(avg_tot, sum_file)

        square_sum_file = np.sum(np.square(data), axis=(0, 2))
        avg_square = np.add(avg_square, square_sum_file)
        avg_tot_square = np.add(avg_tot_square, square_sum_file)
        # for date in dates:
        #     for idx, band in enumerate(date):
        #         s = np.sum(band)
        #         l = len(band)
        #         avg[idx] = s
        #         avg_tot[idx] = s
        #         l_pixels += l
        #         l_pixels_tot += l
                # for pixel in band:
                #     l_pixels += 1
                #     l_pixels_tot += 1
                #     avg[idx] += pixel
                #     avg_square[idx] += pixel**2

    for idx, band in enumerate(avg):
        avg_tot[idx] += band
        band = band / l_pixels

    for idx, band in enumerate(avg_square):
        avg_tot_square[idx] += band
        band = band / l_pixels

    std = [0,0,0,0,0,0,0,0,0,0]
    for idx in range(len(avg)):
        std[idx] = np.sqrt(avg_square[idx] - avg[idx]**2)
    norm = (avg, std)
    if i == 0:
        pkl.dump(norm, open('/home/FQuinton/Bureau/donnees_pse/META/normvals_2018.pkl', 'wb'))

    elif i == 1:
        pkl.dump(norm, open('/home/FQuinton/Bureau/donnees_pse/META/normvals_2019.pkl', 'wb'))

    else:
        pkl.dump(norm, open('/home/FQuinton/Bureau/donnees_pse/META/normvals_2020.pkl', 'wb'))

for band in avg_tot:
    band = band / l_pixels_tot

for band in avg_tot_square:
    band = band / l_pixels_tot

std_tot = [0,0,0,0,0,0,0,0,0,0]
for idx in range(len(avg_tot)):
    std_tot[idx] = np.sqrt(avg_tot_square[idx] - avg_tot[idx]**2)
norm = (avg_tot, std_tot)
pkl.dump(norm, open('/home/FQuinton/Bureau/donnees_pse/META/normvals_tot.pkl', 'wb'))

