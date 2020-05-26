# -*- coding: utf-8 -*-
"""
Created on Apr 02 09:09:31 2020

@author: ncoz
"""

import rasterio
from rasterio.merge import merge
from composite_s2gm_parallel import list_paths_s2
import numpy as np
import os


def main(start_date, end_date, bbox, save_dir):
    # Search for files
    df_files = list_paths_s2(start_date, end_date)

    # Create output folder
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    bounds = tuple(bbox)
    for i, src_pth in df_files["image_20m"].items():
        print(src_pth)
        with rasterio.open(src_pth) as ds:
            meta = ds.profile
            mosaic, out_trans = merge([ds], bounds=bounds)

        if mosaic.max() > 0:
            # Copy RGB values from the result
            visible_bands = mosaic[0:3, :, :].copy()
            # Swap R and B channels
            visible_bands[0, :, :] = mosaic[2, :, :]
            visible_bands[2, :, :] = mosaic[0, :, :]

            # Threshold 0.2 used by Sentinel to create TCI previews
            thrs = visible_bands > 0.2
            visible_bands[thrs] = 0.2

            rgb = 255 * (visible_bands / 0.2)
            rgb = rgb.astype(np.uint8)

            meta["driver"] = "PNG"
            meta["count"] = 3
            meta["width"] = rgb.shape[2]
            meta["height"] = rgb.shape[1]
            meta["dtype"] = "uint8"

            meta.pop('tiled', None)
            meta.pop('compress', None)
            meta.pop('interleave', None)

            _, filnam = os.path.split(src_pth)
            filnam = filnam[:-4] + "_Mura.png"
            png_filename = os.path.join(save_dir, filnam)
            with rasterio.open(png_filename, 'w', **meta) as dst:
                dst.write(rgb)
            print("  DONE!")
        else:
            print("  SKIPPED!")


if __name__ == "__main__":
    # Set time frame (currently Jan 2019)
    st_date = (2017, 4, 28)
    en_date = (2017, 4, 29)
    # Set geo extents
    geo_ext = [597540, 154360, 610660, 165000]  # ref Mura 2017
    # Save directory
    sav_pth = ".\\s2_Mura_2017-04"

    main(st_date, en_date, geo_ext, sav_pth)
