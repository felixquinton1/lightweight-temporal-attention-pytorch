import argparse
import logging

import os
from os import listdir, remove, makedirs
from os.path import isdir, isfile, join, exists
import rasterio.shutil
from rasterio.warp import calculate_default_transform, reproject, Resampling

from zipfile import ZipFile
if __name__ == "__main__":
    dst_crs = 'EPSG:2154'
    for tif_name in os.listdir('/mnt/71A36E2C77574D51/donnees/out/preproj/'):
        print(tif_name)
        with rasterio.open('/mnt/71A36E2C77574D51/donnees/out/preproj/' + tif_name) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            with rasterio.open('/mnt/71A36E2C77574D51/donnees/out/proj/' + tif_name, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)