#import the numpy and gdal libraries
import numpy as np
from osgeo import gdal
import json
import os
import pickle as pkl
dir = '/mnt/71A36E2C77574D51/preprocess/tif_2018/'
files = os.listdir(dir)
band = {}
m = [0,0,0,0,0,0,0,0,0,0]
s = [0,0,0,0,0,0,0,0,0,0]
for f in files:
    ds = gdal.Open('/mnt/71A36E2C77574D51/preprocess/tif_2018/' + f)
    norm = ()

    for i in range(1, ds.RasterCount + 1):
        l = ds.GetRasterBand(i).ReadAsArray()
        m[i-1] += np.mean(l)
        s[i-1] += np.std(l)
m_tot = m
s_tot = s
for i in range(len(m)):
    m[i] = m[i]/36
    s[i] = s[i]/36
norm = (m, s)
pkl.dump(norm, open('/home/FQuinton/Bureau/donnees_pse/META/normvals_2018.pkl', 'wb'))

m = [0,0,0,0,0,0,0,0,0,0]
s = [0,0,0,0,0,0,0,0,0,0]
for f in files:
    ds = gdal.Open('/mnt/71A36E2C77574D51/preprocess/tif_2019/' + f)
    norm = ()

    for i in range(1, ds.RasterCount + 1):
        l = ds.GetRasterBand(i).ReadAsArray()
        m[i-1] += np.mean(l)
        s[i-1] += np.std(l)

for i in range(len(m)):
    m_tot[i] += m[i]
    m[i] = m[i]/27
    s_tot[i] += s[i]
    s[i] = s[i]/27
norm = (m, s)
pkl.dump(norm, open('/home/FQuinton/Bureau/donnees_pse/META/normvals_2019.pkl', 'wb'))

m = [0,0,0,0,0,0,0,0,0,0]
s = [0,0,0,0,0,0,0,0,0,0]
for f in files:
    ds = gdal.Open('/mnt/71A36E2C77574D51/preprocess/tif_2020/' + f)
    norm = ()

    for i in range(1, ds.RasterCount + 1):
        l = ds.GetRasterBand(i).ReadAsArray()
        m[i-1] += np.mean(l)
        s[i-1] += np.std(l)

for i in range(len(m)):
    m_tot[i] += m[i]
    m_tot[i] = m_tot[i]/92
    m[i] = m[i]/29
    s_tot[i] += s[i]
    s_tot[i] += s_tot[i]/92
    s[i] = s[i]/29
norm = (m, s)
pkl.dump(norm, open('/home/FQuinton/Bureau/donnees_pse/META/normvals_2020.pkl', 'wb'))
norm = (m_tot, s_tot)
pkl.dump(norm, open('/home/FQuinton/Bureau/donnees_pse/META/normvals_tot.pkl', 'wb'))


# layers = []
#
# #open raster
# ds = gdal.Open('/mnt/71A36E2C77574D51/preprocess/tif_2018/20180811.tif')
#
# #loop thru bands of raster and append each band of data to 'layers'
# #note that 'ReadAsArray()' returns a numpy array
# for i in range(1, ds.RasterCount+1):
#     print(np.mean(ds.GetRasterBand(i).ReadAsArray()))
#
# with open('/home/FQuinton/Bureau/donnees_pse/META/normvals.pkl', 'rb') as f:
#     data = pkl.load(f)
#     print(data)