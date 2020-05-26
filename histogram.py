# -*- coding: utf-8 -*-
"""Creates histogram of a s2-composite mask.

@author: ncoz

@copyright: ZRC SAZU (Novi trg 2, 1000 Ljubljana, Slovenia)

@history:
    Created on Tue Apr 14 08:26:07 2020
"""
import rasterio
from rasterio.plot import show_hist
from os.path import split
import matplotlib.pyplot as plt
import numpy as np


def png_hist(img_pth, no):
    fold_loc, title = split(img_pth)
    save_name = title[:-4]
    img_type = save_name.split("_")
    img_type = img_type[-1]

    # Determine bins
    bins = np.arange(no+2)-0.5
    # Open raster
    with rasterio.open(img_pth) as src:
        fig, (ax1) = plt.subplots(1, 1, figsize=(7, 7))

        show_hist(src, bins=bins, lw=0.0, title=title, ax=ax1)

        plt.xticks(np.arange(0, no+2, 1))
        plt.legend([img_type])
        save_pth = fold_loc + "\\" + save_name + "_hist.png"
        plt.savefig(save_pth)
        # plt.show()

    return save_pth


if __name__ == "__main__":
    # images = [
    #     ".\\test_Mura-2017-03_v3\\mura_17-03_keep100_20m_sobs.tif",
    #     ".\\test_Mura-2017-03_v3\\mura_17-03_keep100_20m_nok.tif",
    #     ".\\test_Mura-2017-03_v3\\mura_17-03_lt41_20m_sobs.tif",
    #     ".\\test_Mura-2017-03_v3\\mura_17-03_lt41_20m_nok.tif",
    #     ".\\test_Mura-2017-03_v3\\mura_17-03_lt34_20m_sobs.tif",
    #     ".\\test_Mura-2017-03_v3\\mura_17-03_lt34_20m_nok.tif",
    #     ".\\test_Mura-2017-04_v3\\mura_17-04_keep100_20m_sobs.tif",
    #     ".\\test_Mura-2017-04_v3\\mura_17-04_keep100_20m_nok.tif",
    #     ".\\test_Mura-2017-04_v3\\mura_17-04_lt41_20m_sobs.tif",
    #     ".\\test_Mura-2017-04_v3\\mura_17-04_lt41_20m_nok.tif",
    #     ".\\test_Mura-2017-04_v3\\mura_17-04_lt34_20m_sobs.tif",
    #     ".\\test_Mura-2017-04_v3\\mura_17-04_lt34_20m_nok.tif",
    #     ".\\test_Mura-2017-07_v3\\mura_17-07_keep100_20m_sobs.tif",
    #     ".\\test_Mura-2017-07_v3\\mura_17-07_keep100_20m_nok.tif",
    #     ".\\test_Mura-2017-07_v3\\mura_17-07_lt41_20m_sobs.tif",
    #     ".\\test_Mura-2017-07_v3\\mura_17-07_lt41_20m_nok.tif",
    #     ".\\test_Mura-2017-07_v3\\mura_17-07_lt34_20m_sobs.tif",
    #     ".\\test_Mura-2017-07_v3\\mura_17-07_lt34_20m_nok.tif"
    # ]
    dr = "c:"
    mf = "\\Users\\ncoz\\ZRSVN Travinje\\data\\kompoziti_mura-2017"
    images = [
        "\\test_Mura-2017-05_v3\\mura_17-05_lt41_20m_sobs.tif",
        "\\test_Mura-2017-06_v3\\mura_17-06_lt41_20m_sobs.tif",
        "\\test_Mura-2017-08_v3\\mura_17-08_lt41_20m_sobs.tif",
        "\\test_Mura-2017-09_v3\\mura_17-09_lt41_20m_sobs.tif",
        "\\test_Mura-2017-10_v3\\mura_17-10_lt41_20m_sobs.tif",
        "\\test_Mura-2017-05_v3\\mura_17-05_lt41_20m_nok.tif",
        "\\test_Mura-2017-06_v3\\mura_17-06_lt41_20m_nok.tif",
        "\\test_Mura-2017-08_v3\\mura_17-08_lt41_20m_nok.tif",
        "\\test_Mura-2017-09_v3\\mura_17-09_lt41_20m_nok.tif",
        "\\test_Mura-2017-10_v3\\mura_17-10_lt41_20m_nok.tif"
    ]
    nrs = [7, 6, 12, 10, 12] * 2  # Number of images used in compositing
    for image, nr in zip([images[0]], [nrs[0]]):
        # pth = dr + mf + image
        # print (pth)
        # print(nr)
        png_hist(dr + mf + image, nr)
