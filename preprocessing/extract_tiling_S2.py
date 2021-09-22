#!/usr/bin/env python
# coding: utf-8

#
# Copyright (c) 2020 IGN France.
#
# This file is part of lpis_prepair
# (see https://gitlab.com/capmon/lpis_prepair).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import argparse
import logging

import os
from os import listdir, remove, makedirs
from os.path import isdir, isfile, join, exists
import rasterio.shutil
from rasterio.warp import calculate_default_transform, reproject, Resampling

from zipfile import ZipFile

import pathlib

from tqdm import tqdm


def list_theia_dir(in_dir):
    """
    renvoie une liste des archives S2 theia présentes dans un dossier

    la liste renvoie une liste de chemin de fichiers zip. On considere qu'un fichier
    zip est une archive THEIA quand son nom commence par "SENTINEL"

    :param in_dir: dossier contenant des archives sentinel 2 THEIA
    :return: liste de chemins de fichier zip de type archive theia
    """
    subdir = [f for f in listdir(in_dir) if isdir(os.path.join(in_dir, f))]
    # print(f"subdir {subdir}")
    theia_dir = [os.path.abspath(os.path.join(in_dir, f)) for f in subdir if f[:8] == "SENTINEL"]

    dir_files = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
    zip_files = [f for f in dir_files if f[-4:] == ".zip"]
    theia_zip = [os.path.abspath(os.path.join(in_dir, f)) for f in zip_files if f[:8] == "SENTINEL"]

    return theia_dir + theia_zip


def get_rel_orbit(abs_orbit, sat_id):
    if sat_id == "SENTINEL2B":
        return ((abs_orbit - 27) % 143) + 1
    elif sat_id == "SENTINEL2A":
        return ((abs_orbit + 2) % 143) + 1
    else:
        return None


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def theia_get_band(s2_tile_path, band_id):
    """

    :param s2_tile_path:
    :param band_id:
    :return:
    """
    tile_name = os.path.split(s2_tile_path)[1]
    is_zip = False
    if tile_name[-4:] == ".zip":
        is_zip = True
        with ZipFile(s2_tile_path) as zf:
            dirs = list(set([os.path.dirname(x) for x in zf.namelist()]))
            dirs.sort(key=len)
            tile_name = dirs[0]

    band_dict = {
        "2": "_FRE_B2",
        "3": "_FRE_B3",
        "4": "_FRE_B4",
        "5": "_FRE_B5",
        "6": "_FRE_B6",
        "7": "_FRE_B7",
        "8": "_FRE_B8",
        "8A": "_FRE_B8A",
        "11": "_FRE_B11",
        "12": "_FRE_B12"}

    band_suffix = band_dict[band_id] + ".tif"
    if is_zip:
        # out_path = "zip+file://{0}".format(os.path.join(s2_tile_path+"!",tile_name, tile_name + band_suffix))
        out_path = "zip+file://{0}".format(s2_tile_path + "!" + "/" + tile_name + "/" + tile_name + band_suffix)
    else:
        out_path = os.path.join(s2_tile_path, tile_name + band_suffix)
    return out_path


def theia_get_masks(s2_tile_path):
    """

    :param s2_tile_path:
    :param band_id:
    :return:
    """
    tile_name = os.path.split(s2_tile_path)[1]
    is_zip = False
    if tile_name[-4:] == ".zip":
        is_zip = True
        with ZipFile(s2_tile_path) as zf:
            dirs = list(set([os.path.dirname(x) for x in zf.namelist()]))
            dirs.sort(key=len)
            tile_name = dirs[0]

    masks = []
    for mask in ["CLM", "EDG", "SAT", "MG2"]:
        for res in ["R1", "R2"]:
            if is_zip:
                # curr_mask = "zip+file://{0}".format(os.path.join(s2_tile_path+"!", tile_name, "MASKS", tile_name+f"_{mask}_{res}.tif"))
                curr_mask = "zip+file://{0}".format(
                    s2_tile_path + "!" + "/" + tile_name + "/MASKS/" + tile_name + f"_{mask}_{res}.tif")
            else:
                curr_mask = os.path.join(s2_tile_path, "MASKS", tile_name + f"_{mask}_{res}.tif")
            masks.append(curr_mask)

    return masks


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="extract s2 tiles and delete unused file")
    # parser.add_argument(
    #     "-i", "--in_s2_dir", dest="in_s2_dir", help="input dir containing s2 tile form theia", required=True)
    # parser.add_argument(
    #     "-o", "--out_s2_dir", dest="out_s2_dir", help="output dir containing s2 tile with tiling", required=True)
    # parser.add_argument(
    #     "-b", "--out_bands", dest="out_bands", help="s2 bands to extract, all by default", nargs='+', required=False)
    # args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    out_dir = "/mnt/71A36E2C77574D51/preprocess2/"
    # out_dir = "C:/Users/felix/OneDrive/Bureau/test/out/"
    #out_dir = args.out_s2_dir
    if not exists(out_dir):
        makedirs(out_dir)

    # if not args.out_bands:
    # bands_10m = ["2", "3", "4", "8"]
    # bands_20m = ["5", "6", "7", "8A", "11", "12"]
    bands_10m = ["2", "3", "4"]
    bands_20m = []
    out_bands = bands_10m + bands_20m
    # else:
    #     out_bands = args.out_bands

    out_dir = os.path.abspath(out_dir)
    #all_theia_archives = list_theia_dir(args.in_s2_dir)
    all_theia_archives = list_theia_dir("/mnt/71A36E2C77574D51/felix/S2/T31TFM_A/")
    # all_theia_archives = list_theia_dir("C:/Users/felix/OneDrive/Bureau/test/in/")
    logging.info("find  {0} archives".format(len(all_theia_archives)))
    for theia_archive in tqdm(all_theia_archives):
        out_tile_dir = os.path.join(out_dir, os.path.basename(theia_archive))
        if out_tile_dir[-4:] == ".zip":
            out_tile_dir = out_tile_dir[:-4]
        logging.info("tile archive : {0} to {1}".format(theia_archive, out_tile_dir))
        tif_name = str(out_tile_dir[-32: -24])
        if os.path.exists(out_tile_dir):
            continue

        in_band_1 = theia_get_band(theia_archive, bands_10m[0])
        with rasterio.open(in_band_1) as band_1:
            kwds = band_1.profile
        scale = 2
        kwds.update(count=len(out_bands), dtype='int16')
        with rasterio.open('/mnt/71A36E2C77574D51/preprocess2/tif/' + tif_name + '.tif', 'w', **kwds) as dest:
        # with rasterio.open('C:/Users/felix/OneDrive/Bureau/test/out/preproj/' + tif_name + '.tif', 'w', **kwds) as dest:
            for idx, band in enumerate(out_bands):
                in_band_path = theia_get_band(theia_archive, band)
                out_band_path = os.path.join(out_tile_dir, os.path.basename(in_band_path))
                with rasterio.open(in_band_path) as in_band:
                    if band in bands_10m:
                        dest.write(in_band.read(1), idx + 1)
                    if band in bands_20m:
                        height = in_band.height * scale
                        width = in_band.width * scale
                        data = in_band.read(
                            out_shape=(in_band.count, int(height), int(width)),
                            resampling=Resampling.bilinear)

                        transform = in_band.transform * in_band.transform.scale(
                            (in_band.width / data.shape[-1]),
                            (in_band.height / data.shape[-2]))
                        profile = in_band.profile
                        profile.update(transform=transform, driver='GTiff', height=height, width=width, crs=in_band.crs)
                        with rasterio.open('/mnt/71A36E2C77574D51/preprocess2/10m/' + str(idx) + '.tif',
                                           'w', **profile) as step:
                        # with rasterio.open('C:/Users/felix/OneDrive/Bureau/test/out/10m/' + str(idx) + '.tif',
                        #                    'w', **profile) as step:
                            step.write(data)
                        with rasterio.open('/mnt/71A36E2C77574D51/preprocess2/10m/' + str(idx) + '.tif')\
                                as resampled_band:
                        # with rasterio.open('C:/Users/felix/OneDrive/Bureau/test/out/10m/' + str(idx) + '.tif') \
                        #         as resampled_band:
                            dest.write(resampled_band.read(1), idx + 1)

        # rasterio.shutil.copy(
        #     in_band, out_band_path, driver='GTiff', tiled='YES', blockxsize=512, blockysize=512,
        #     compress="DEFLATE", predictor=2)

        # mask_dir = os.path.join(out_tile_dir, "MASKS")
        # pathlib.Path(mask_dir).mkdir(parents=True, exist_ok=True)
        # mask_list = theia_get_masks(theia_archive)
        # for mask_path in mask_list:
        #     out_mask_path = 'C:/Users/felix/OneDrive/Bureau/test/out/' + tif_name + '_mask.tif'
        #     # out_mask_path = os.path.join(mask_dir, os.path.basename(mask_path))
        #     with rasterio.open(mask_path) as in_mask:
        #
        #         rasterio.shutil.copy(
        #             in_mask, out_mask_path, driver='GTiff', tiled='YES', blockxsize=512, blockysize=512,
        #             compress="DEFLATE", predictor=2)
