import numpy as np
import os
import pickle as pkl
from tqdm import tqdm

dir = ['/home/FQuinton/Bureau/data_pse/DATA/2018/', '/home/FQuinton/Bureau/data_pse/DATA/2019/',
       '/home/FQuinton/Bureau/data_pse/DATA/2020/']

avg_tot = np.zeros(10)
avg_tot_square = np.zeros(10)
l_pixels_tot = 0

for i in range(len(dir)):
    files = os.listdir(dir[i])
    avg = np.zeros(10)
    avg_square = np.zeros(10)
    l_pixels = 0
    for f in tqdm(files):
        data = np.load(dir[i] + f)
        data = data.astype('float64')
        avg = np.add(avg, np.sum(data, axis=(0, 2)))
        avg_square = np.add(avg_square, np.sum(np.square(data), axis=(0, 2)))

        nb_pixels = np.shape(data)[0] * np.shape(data)[2]
        l_pixels += nb_pixels
        l_pixels_tot += nb_pixels

    avg_tot = np.add(avg_tot, avg)
    avg = np.divide(avg, l_pixels)
    avg_square = np.divide(avg_square, l_pixels)
    # for band in avg:
    #     band = band / l_pixels
    #
    # avg_tot_square = np.add(avg_tot_square, avg_square)
    # for band in avg_square:
    #     band = band / l_pixels

    std = np.sqrt(np.add(avg_square, -np.square(avg)))
    norm = (avg.tolist(), std.tolist())
    if i == 0:
        pkl.dump(norm, open('/home/FQuinton/Bureau/data_pse/META/normvals_2018.pkl', 'wb'))

    elif i == 1:
        pkl.dump(norm, open('/home/FQuinton/Bureau/data_pse/META/normvals_2019.pkl', 'wb'))

    else:
        pkl.dump(norm, open('/home/FQuinton/Bureau/data_pse/META/normvals_2020.pkl', 'wb'))

avg_tot = np.divide(avg, l_pixels)
avg_tot_square = np.divide(avg_square, l_pixels)
# for band in avg_tot:
#     band = band / l_pixels_tot
#
# for band in avg_tot_square:
#     band = band / l_pixels_tot

std_tot = np.sqrt(np.add(avg_tot_square, -np.square(avg_tot)))
norm = (avg_tot.tolist(), std_tot.tolist())
pkl.dump(norm, open('/home/FQuinton/Bureau/data_pse/META/normvals_tot.pkl', 'wb'))

