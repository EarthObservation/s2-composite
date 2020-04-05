# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:09:31 2020

@author: ncoz

This code will be used for compositing based on max NDVI
"""

import numpy as np
from datetime import date, timedelta
import glob
import os
import time
import rasterio
from rasterio.windows import Window
import math
import pandas as pd
from affine import Affine
# import logging
import concurrent.futures


def round_multiple(nr, x_left, pix):
    """Rounds to the nearest multiple of the raster. Used when evaluating the
    geographical

    Parameters
    ----------
    nr : float
        Number that needs to be rounded.
    x_left : float
        Upper-left x or y coordinate of the raster.
    pix : float
        Pixel size in the selected direction.

    Returns
    -------
    float
        Number that was rounded to the nearest multiple of the x_left.

    """
    return pix * round((nr - x_left) / pix) + x_left


def is_list_uniform(check_list, msg):
    """Checks if all items in the list are the same, and raises an exception
    if not;
    i.e. item[0] == item[1] == item[2] == etc.

    Parameters
    ----------
    check_list : list
        List that will be evaluated.
    msg : string
        Error message that will be displayed.

    Raises
    ------
    Exception
        Custom message for the exception.

    Returns
    -------
    None.

    """
    if not check_list:
        raise ValueError('Empty list being passed to is_list_uniform..')
    check = check_list.count(check_list[0]) == len(check_list)
    if not check:
        raise Exception(msg)


def output_image_extent(src_fps, bbox):
    """Determines the maximum extents of the output image, taking into account
    extents of all considered files and a user specified bounding box.

    Parameters
    ----------
    src_fps : DataFrame
        List of file paths to input TIFs.
    bbox : list
        Extents in order [xL, yD, xR, yU].

    Raises
    ------
    Exception
        Error is raised if bounding box doesn't overlap one of the images.
        Future plan: exclude that image from file list??

    Returns
    -------
    output : dictionary
        Contains extents of and other metadata required for final image.

    """
    str_time = time.time()

    skip_im = []  # Initiate
    # Open TIF files as DataSets and evaluate properties
    tif_ext, pix_res, src_all, bnd_num = ([] for _ in range(4))

    for idx, fp in src_fps.items():
        print('Opening raster {}'.format(fp[109:-20]))
        with rasterio.open(fp) as src:
            src_all.append(src)
            # Read raster properties
            tif_ext.append([i for i in src.bounds])  # left, bottom, right, top
            pix_res.append(src.res)
            bnd_num.append(src.count)

    # Check if all images have the same pixel size
    is_list_uniform(pix_res, 'Pixel size of input images not matching.')

    # Check if all images have the same number of bands
    is_list_uniform(bnd_num, 'Number of bands in input images not matching.')

    # Pixel size
    pix_x, pix_y = pix_res[0]

    # Check if image falls out of BBOX and determine extents
    tif_ext_filtered = []
    xmin_box = np.nan
    ymin_box = np.nan
    xmax_box = np.nan
    ymax_box = np.nan
    if bbox:
        for idx, _ in src_fps.items():
            # Get extents of current image
            xmin_img, ymin_img, xmax_img, ymax_img = [a for a in tif_ext[idx]]

            # Round bbox to nearest multiple of raster
            xmin_box = round_multiple(bbox[0], xmin_img, pix_x)
            ymin_box = round_multiple(bbox[1], ymax_img, pix_y)
            xmax_box = round_multiple(bbox[2], xmin_img, pix_x)
            ymax_box = round_multiple(bbox[3], ymax_img, pix_y)

            # Check if bbox falls out of image extents (True if it falls out)
            chk_bbox = (xmin_img > xmax_box or ymin_img > ymax_box or
                        xmax_img < xmin_box or ymax_img < ymin_box)
            if chk_bbox:
                skip_im.append(idx)
            else:
                tif_ext_filtered.append(tif_ext[idx])
    else:
        tif_ext_filtered = tif_ext

    # FIND MAX EXTENTS AND DETERMINE SIZE OF OUTPUT IMAGE
    xmin_out = min((li[0] for li in tif_ext_filtered))
    ymin_out = min((li[1] for li in tif_ext_filtered))
    xmax_out = max((li[2] for li in tif_ext_filtered))
    ymax_out = max((li[3] for li in tif_ext_filtered))
    if bbox:
        # Compare with image extents
        xmin_out = max(xmin_out, xmin_box)
        ymin_out = max(ymin_out, ymin_box)
        xmax_out = min(xmax_out, xmax_box)
        ymax_out = min(ymax_out, ymax_box)

    # Calculate size of output array
    tif_wide = int(math.ceil(xmax_out - xmin_out) / abs(pix_x))
    tif_high = int(math.ceil(ymax_out - ymin_out) / abs(pix_y))
    nr_bands = bnd_num[0]
    nr_image = len(src_fps)

    output = {'skip': skip_im,
              'width': tif_wide,
              'height': tif_high,
              'bandsCount': nr_bands,
              'imgCount': nr_image,
              'bounds': [xmin_out, ymin_out, xmax_out, ymax_out],
              'pixels': (pix_x, pix_y)
              }

    end_time = time.time() - str_time
    print(f'--- Time to evaluate geo. extents: {end_time} seconds ---')
    return output


def pixel_offset(image_meta, x_coord, y_coord):
    """Finds the position of the selected pixel in the current image.

    Parameters
    ----------
    image_meta : dictionary
        Must include keys:
            "bounds" which is a bounding box object from TIF meta data,
            "pixels" which is a pixel resolution tuple from TIF meta data.
    x_coord : float
        X coordinate of the pixel center.
    y_coord : float
        Y coordinate of the pixel center.

    Returns
    -------
    rd_win : TYPE
        Offset window for reading a single pixel from .

    """
    # Output image extents
    xlf_out, _, _, yup_out = image_meta['bounds']

    # Read pixel resolution
    pix_x, pix_y = image_meta['pixels']

    # X-direction
    x_off = x_coord - xlf_out
    x_off = math.floor(x_off/pix_x)

    # Y-direction
    y_off = yup_out - y_coord
    y_off = math.floor(y_off/pix_y)

    # Prepare Window for reading raster from TIF
    rd_win = Window(x_off, y_off, 1, 1)

    return rd_win


def collect_meta(image_paths):
    img_meta = []
    for tif in image_paths:
        with rasterio.open(tif) as ds:
            meta_one = {'bounds': ds.bounds, 'pixels': ds.res}

        # Add row to input_images_meta Data Frame
        img_meta.append(meta_one)

    return img_meta


def isSnow(one_pixel):
    # with rasterio.open(tif_path) as ds:
    #     one_pixel = ds.read(window=win).flatten()

    tcb = (0.3029 * one_pixel[0]  # b2
           + 0.2786 * one_pixel[1]  # b3
           + 0.4733 * one_pixel[2]  # b4
           + 0.5599 * one_pixel[7]  # b8A
           + 0.5080 * one_pixel[8]  # b11
           + 0.1872 * one_pixel[9]  # b12
           )
    ndsi = (one_pixel[1] - one_pixel[8]) / (one_pixel[1] + one_pixel[8])

    s2gm_snow = ndsi > 0.6 and tcb > 0.36

    return s2gm_snow


def medoid_s2gm(calc_pix, dist='euclid'):
    s = pd.Series(index=calc_pix.index, dtype='float')
    for ii, rowi in calc_pix.iterrows():
        e_dist = 0
        for ij, rowj in calc_pix.drop([ii]).iterrows():
            # Calculate Euclidian distance
            if dist == 'euclid':
                e_dist += np.sqrt(((rowj - rowi) ** 2).sum())
            elif dist == 'norm_diff':
                # e_bnd[bd] = (xB - xA)**2
                # abs((xB - xA) / (xB + xA))
                e_dist += abs((rowj - rowi) / (rowj + rowi)).sum()
            else:
                raise Exception('Error: Unknown distance for MEDOID.')
        # Update column
        s.loc[ii] = e_dist

    return s


def stc_indexes(calc_pix):
    """Calculates indexes required for the STC method.

    Args:
        calc_pix (DataFrame): Data frame with bands as columns and observations
        as rows.

    Returns (DataFrame): Data Frame with all indexes.
    """
    df = pd.DataFrame()

    # mNDWI
    df['mNDWI'] = (
        (calc_pix.b3 - calc_pix.b11)
        / (calc_pix.b3 + calc_pix.b11))

    # NDVI
    df['NDVI'] = (
        (calc_pix.b8 - calc_pix.b4)
        / (calc_pix.b8 + calc_pix.b4)
        )

    # TCB
    df['TCB'] = (0.3029 * calc_pix.b2
                 + 0.2786 * calc_pix.b3
                 + 0.4733 * calc_pix.b4
                 + 0.5599 * calc_pix.b8A
                 + 0.5080 * calc_pix.b11
                 + 0.1872 * calc_pix.b12
                 )

    return df


def stc_s2gm(stc, calc_pix, snow_df):
    """Returns index of best pixel from STC method.

    The decision logic and thresholds based on S2GM algorithm by Kirches and
    Brockmann (2019)

    Args:
        stc (DataFrame): Data Frame containing indexes for STC method
        calc_pix (DataFrame): Data Frame containing pixels (for cloud test)
        snow_df (DataFrame): Data Frame with snow test results

    Returns (integer): index of selected best pixel

    """
    # Index used for snow and cloud tests
    idx_tcb_min = stc.TCB.idxmin()

    # Set criteria for STC
    criteria1 = (stc.mNDWI.mean() < -0.55
                 and stc.NDVI.max() - stc.NDVI.mean() < 0.05)
    if criteria1:
        return stc.NDVI.idxmax()

    criteria2 = (stc.NDVI.mean() < -0.3
                 and stc.mNDWI.mean() - stc.NDVI.min() < 0.05)
    if criteria2:
        return stc.mNDWI.idxmax()

    criteria3 = stc.NDVI.mean() > 0.6 and stc.TCB.mean() < 0.45
    if criteria3:
        return stc.NDVI.idxmax()

    # If pixel with min(TCB) fails cloud test
    criteria4 = not stc_cloud_test(calc_pix.loc[idx_tcb_min])
    if criteria4:
        return idx_tcb_min

    # If pixel with min(TCB) fails snow test
    criteria5 = not snow_df[idx_tcb_min]
    if criteria5:
        if stc.TCB.min() > 1:
            return None
        else:
            return idx_tcb_min

    criteria6 = stc.NDVI.mean() < -0.2
    if criteria6:
        return stc.mNDWI.idxmax()

    criteria7 = stc.TCB.mean() > 0.45
    if criteria7:
        return stc.NDVI.idxmin()

    # If non of the above return max(NDVI)
    return stc.NDVI.idxmax()


def stc_cloud_test(s_pix):
    """Returns true if clouds are detected.

    Based on the isCLoudOrSnow function from S2GM. The functions takes in one
    pixel and determines whether it was correctly classified as cloud. It checks
    for three criteria, first if it isn't snow, second if it is cloud with high
    probability and finally if it is cloud with low probability.

    Args:
        s_pix (Series): Pandas series containing info for a single pixel.

    Returns (bool): True if clouds, else False

    """
    r_b3b11 = s_pix.b3 / s_pix.b11    # Ratio b3/b11
    r_b11b3 = s_pix.b11 / s_pix.b3    # Ratio b3/b11
    rgb_mean = (s_pix.b2 + s_pix.b3 + s_pix.b4) / 3    # Brightness
    # Normalised difference between b8 and b11
    nd_b8b11 = (s_pix.b8 - s_pix.b11) / (s_pix.b8 + s_pix.b11)
    # tcHaze
    tc_haze = (-0.8239 * s_pix.b2
               + 0.0849 * s_pix.b3
               + 0.4396 * s_pix.b4
               - 0.0580 * s_pix.b8A
               + 0.2013 * s_pix.b11
               - 0.2773 * s_pix.b12)
    # TCB
    tcb = (0.3029 * s_pix.b2
           + 0.2786 * s_pix.b3
           + 0.4733 * s_pix.b4
           + 0.5599 * s_pix.b8A
           + 0.5080 * s_pix.b11
           + 0.1872 * s_pix.b12
           )
    # Normalised difference snow index
    ndsi = (s_pix.b3 - s_pix.b11) / (s_pix.b3 + s_pix.b11)

    # Test if it is not snow
    not_snow = ndsi > 0.7 and not(r_b3b11 > 1 and tcb < 0.36)
    if not_snow:
        return True

    # Test if it is high probability cloud
    hpc1 = ((r_b3b11 > 1 and rgb_mean > 0.3)
            and (tc_haze < -0.1 or (tc_haze > -0.08 and nd_b8b11 < 0.4))
            )
    hpc2 = (tc_haze < -0.2)
    hpc3 = (r_b3b11 > 1 and rgb_mean < 0.3)
    hpc4 = (tc_haze < -0.055 and rgb_mean > 0.12)
    hpc5 = (not(r_b3b11 > 1 and rgb_mean < 0.3)
            and (tc_haze < -0.09 and rgb_mean > 0.12)
            )
    hp_cloud = hpc1 or hpc2 or hpc3 and hpc4 or hpc5
    if not not_snow and hp_cloud:
        return True

    # Test if it is low probability cloud
    lpc1 = ((r_b11b3 > 1 and rgb_mean < 0.2)
            and (tc_haze < -0.1 or (tc_haze < -0.08 and nd_b8b11 < 0.4))
            )
    lpc2 = hpc2
    lpc3a = r_b3b11 > 1
    lpc3b = rgb_mean < 0.2
    lpc3c = hpc4
    lpc3d = not(r_b3b11 > 1 and rgb_mean < 0.2) and (tc_haze < -0.02)
    lpc3 = lpc3a and lpc3b and lpc3c or lpc3d

    lp_cloud = lpc1 or lpc2 or lpc3
    if not not_snow and not hp_cloud and lp_cloud:
        return True

    return False


def select_pixel(calc_pix, nok_one, snow_df, medoid_distance):
    if nok_one == 0:
        sel_pix_idx = None
    elif nok_one == 1:
        sel_pix_idx = calc_pix.index[0]
    elif nok_one < 4:
        # Calculate indexes required for STC method
        stc = stc_indexes(calc_pix)
        # Select index with STC method
        sel_pix_idx = stc_s2gm(stc, calc_pix, snow_df)
    else:
        # Calculate sum of distances using the selected method
        med = medoid_s2gm(calc_pix, dist=medoid_distance)
        # Select pixel with min MEDOID value (shortest distance to others)
        sel_pix_idx = med.idxmin()

    return sel_pix_idx


def list_paths_s2(st_date, en_date):
    # Unpack input
    y_st, m_st, d_st = st_date
    y_en, m_en, d_en = en_date

    # Dirs with source files
    mdir_10m = "Q:\\Sentinel-2_atm_10m_mosaicked_d96"
    mdir_20m = "Q:\\Sentinel-2_atm_20m_mosaicked_d96"

    # Suffixes
    sfx_img = "*_p2atm_d96tm.tif"
    sfx_msk = "*_p2atm_mask_d96tm.tif"

    sdate = date(y_st, m_st, d_st)  # start date
    edate = date(y_en, m_en, d_en)  # end date
    delta = edate - sdate

    file_list = {"date": [],
                 "image_10m": [],
                 "image_20m": [],
                 "mask_20m": []
                 }
    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        s_day = day.strftime("%Y%m%d")  # make date into string
        s_year = day.strftime("%Y")

        q10m = os.path.join(mdir_10m, s_year, s_day + "*")
        q20m = os.path.join(mdir_20m, s_year, s_day + "*")

        dirs10m = glob.glob(q10m)
        dirs20m = glob.glob(q20m)

        if dirs10m is None:
            dirs10m = []
        if dirs20m is None:
            dirs20m = []

        dirs10m.sort()
        dirs20m.sort()

        if len(dirs10m) == len(dirs20m) and len(dirs10m) == 0:
            continue
        # elif len(dirs10m) == len(dirs20m) and len(dirs10m) == 1:
        #     chk1 = os.path.split(dirs10m[0])[1][:-4]
        #     chk2 = os.path.split(dirs20m[0])[1][:-4]
        #     if not(chk1 == chk2):
        #         raise Exception(f"Different 10m and 20m filename for date {s_day}")
        elif len(dirs10m) == len(dirs20m) and len(dirs10m) > 0:
            chk1 = [os.path.split(k)[1][:-4] for k in dirs10m]
            chk2 = [os.path.split(k)[1][:-4] for k in dirs20m]
            if not (chk1 == chk2):
                raise Exception(f"Different 10m and 20m filenames for date {s_day}")
        elif len(dirs10m) > len(dirs20m):
            raise Exception(f"Filename {s_day} missing in 20m resolution")
        else:
            raise Exception(f"File {s_day} missing in 10m resolution")

        for dir10, dir20 in zip(dirs10m, dirs20m):
            q_i10 = os.path.join(dir10, sfx_img)
            q_i20 = os.path.join(dir20, sfx_img)
            q_m20 = os.path.join(dir20, sfx_msk)

            pth_i10 = glob.glob(q_i10)[0]
            pth_i20 = glob.glob(q_i20)[0]
            pth_m20 = glob.glob(q_m20)[0]

            file_list["date"].append(s_day)
            file_list["image_10m"].append(pth_i10)
            file_list["image_20m"].append(pth_i20)
            file_list["mask_20m"].append(pth_m20)

    df_paths = pd.DataFrame(file_list)

    return df_paths


def open_rio_list(df_inputs):
    src_files = []
    for item in df_inputs:
        src_files.append(rasterio.open(item))

    return src_files


def close_rio_list(ds_list):
    for file in ds_list:
        file.close()


def get_mask_value(pix_mask, criteria="less_than", threshold=35):
    """Selects if pixel is snow/bad/valid based on the mask.

    The functions first checks if pixel is classified as snow (value 33).
    If not the functions then decides if pixel is valid or bad, using one of the
    two methods. "less_than" will identify bad pixels if mask is less than
    threshold value, while "all_bad" only selects valid pixels that have the
    same mask value as threshold.

    Args:
        pix_mask (int): mask class value
        criteria (str): "less_than" or "all_bad"
        threshold (int): threshold value for determining a valid pixel

    Returns:
        (str) One of the three possible outcomes: "snow", "bad", or "valid".
    """
    if criteria == "all_bad":
        if pix_mask == 33:
            return "snow"
        elif pix_mask != threshold:
            return "bad"
        else:
            return "valid"
    elif criteria == "less_than":
        if pix_mask == 33:
            return "snow"
        elif pix_mask < threshold:
            return "bad"
        else:
            return "valid"
    else:
        raise ValueError(f"Unrecognized value \"{criteria}\" in criteria.")


def process_one_line(y, other):
    """"""
    main_ex, df_inputs, resolution, mask_crt, mask_thr, medoid_distance = other

    out_extents = main_ex["bounds"]
    wid_x = main_ex['width']
    pixels = main_ex['pixels']
    nr_bnd = main_ex['bandsCount']

    df_inputs["src_files_20m"] = open_rio_list(df_inputs["image_20m"])
    df_inputs["src_masks_20m"] = open_rio_list(df_inputs["mask_20m"])
    if resolution == "10m":
        df_inputs["src_files_10m"] = open_rio_list(df_inputs["image_10m"])

    nobs = np.zeros((1, wid_x), dtype=np.int8)
    nok = nobs.copy()
    composite = np.zeros((nr_bnd, wid_x), dtype=np.float32)
    # TEMPORARY MESSAGE
    print(f"Started processing line {y+1}...")
    for x in range(wid_x):
        # Coordinates of the selected pixel (pixel center)
        x_coord = out_extents[0] + (x + 0.5) * pixels[0]
        y_coord = out_extents[3] - (y + 0.5) * pixels[1]

        # Initiate table for the calculation of the best pixel
        p_col = ['b2', 'b3', 'b4', 'b5', 'b6',
                 'b7', 'b8', 'b8A', 'b11', 'b12']
        calc_pix = pd.DataFrame(columns=p_col, dtype='float')
        snow_df = pd.Series(dtype='bool')

        # initiate variables
        nobs_one = 0
        nok_one = 0

        # LOOP ALL INPUT IMAGES TO POPULATE THE CALCULATION TABLE
        for ind, row in df_inputs.iterrows():
            # 1\ CHECK IF PIXEL EXISTS (in the current image)
            # ==============================================================
            if resolution == "10m":
                r_meta = row["meta_10m"]
            else:
                r_meta = row["meta_20m"]

            chk_bbx = (
                    r_meta['bounds'].right
                    >= x_coord
                    >= r_meta['bounds'].left
            )
            chk_bby = (
                    r_meta['bounds'].top
                    >= y_coord
                    >= r_meta['bounds'].bottom
            )

            if not (chk_bbx and chk_bby):
                continue
            else:
                nobs_one += 1

            # 2\ CHECK IF PIXEL IS GOOD (always use 20m mask)
            # ==============================================================
            # Get location of this pixel in mask
            win_20m = pixel_offset(row["meta_20m"], x_coord, y_coord)
            if resolution == "10m":
                win_10m = pixel_offset(row["meta_10m"], x_coord, y_coord)
            else:
                win_10m = None
            # Read mask value
            opm = row["src_masks_20m"].read(window=win_20m)
            # Determine pixel type (snow/valid/bad)
            pix_class = get_mask_value(opm[0, 0, 0], mask_crt, mask_thr)

            if pix_class == "snow":
                # Read the pixel so it can be checked for snow
                pix_snow = row["src_files_20m"].read(window=win_20m).flatten()
                if resolution == "10m":
                    pix_10m = row["src_files_10m"].read(window=win_10m).flatten()
                    pix_snow[0:3] = pix_10m[0:3]
                    pix_snow[6] = pix_10m[3]

                # Skip if pixel is zero (class. problem at swath border)
                pixel_check = (
                        any(np.isnan(pix_snow))
                        or any(np.isinf(pix_snow))
                        or any(pix_snow == 0)
                )
                if pixel_check:
                    continue
                else:
                    # Check for snow
                    is_snow = isSnow(pix_snow)
                    if is_snow:
                        nok_one += 1
                        one_pixel = pix_snow
                    else:
                        continue
            elif pix_class == "bad":  # not valid pixels
                continue
            else:
                # Pixel is valid
                one_pixel = row["src_files_20m"].read(window=win_20m).flatten()
                if resolution == "10m":
                    pix_10m = row["src_files_10m"].read(window=win_10m).flatten()
                    one_pixel[0:3] = pix_10m[0:3]
                    one_pixel[6] = pix_10m[3]
                # Skip if pixel (any band) is zero/nan/inf
                # (classification problem at swath border, mask shows good
                # pixel when it is in fact bad)
                pixel_check = (
                        any(np.isnan(one_pixel))
                        or any(np.isinf(one_pixel))
                        or any(one_pixel == 0)
                )
                if pixel_check:
                    continue
                else:
                    is_snow = False
                    nok_one += 1
            # ==============================================================

            # \5 Populate Pandas Data Frame
            # ==============================================================
            calc_pix.loc[ind] = one_pixel
            # Add is_snow test result
            snow_df.loc[ind] = is_snow

        # \6 Select pixel (different methods depending on nok)
        # ----------------------
        sel_pix_idx = select_pixel(calc_pix, nok_one,
                                   snow_df, medoid_distance)

        # SELECT BEST PIXEL
        if sel_pix_idx:
            # sel_pix = calc_pix.loc[sel_pix_idx].to_numpy(dtype='float32')
            sel_pix = np.array(calc_pix.loc[sel_pix_idx])
        else:
            sel_pix = np.full((10,), np.nan, dtype='float32')

        # BUILD ROW OF OUTPUT MATRIX
        nobs[:, x] = nobs_one
        nok[:, x] = nok_one
        if resolution == "10m":
            composite[:, x] = np.append(sel_pix[0:3], sel_pix[6])
        else:
            composite[:, x] = sel_pix

    close_rio_list(df_inputs["src_files_20m"])
    close_rio_list(df_inputs["src_masks_20m"])
    if resolution == "10m":
        close_rio_list(df_inputs["src_files_10m"])

    return y, composite, nobs, nok


def do_something(idx, other):
    seconds, nok_value = other
    print(f'Process for {idx} started... execution time {seconds}s')
    time.sleep(seconds)
    nobs_value = idx[0] + idx[1]
    rslt = np.full((10, 1, 1), nobs_value)
    return idx, rslt, nobs_value, nok_value


def main():
    # ========= TEMPORARY INPUT ============================================
    # bbox = [348900, 16400, 631600, 201580]  # SLO w/o outermost pixels
    # bbox = [500000, 110000, 530000, 130000]  # Celjska kotlina
    # bbox = [500000, 110000, 502000, 112000]  # 100x100 in 20m
    bbox = [500000, 110000, 504000, 114000]  # 200x200 in 20m
    # bbox = [500000, 110000, 500100, 110100]  # XS

    # Set composite time frame (currently Jan 2019)
    start_date = (2019, 3, 1)
    end_date = (2019, 3, 31)

    # Resolution
    resolution = "10m"  # "10m" or "20m"

    # Medoid method
    medoid_distance = "euclid"  # Either "euclid" or "norm_diff"

    # Mask/filtering
    mask_crt = "less_than"  # Either "less_than" or "all_bad"
    mask_thr = 33

    # Save paths/dir/names...
    save_dir = ".\\test_s2gm"
    save_nam = "test17_lt33" + "_" + resolution
    # ======================================================================

    # ##### #
    # START #
    # ##### #
    time_a = time.time()

    # =========================== #
    # GET ALL REQUIRED PARAMETERS #
    # =========================== #

    # Create pandas table with info for all input images
    # --------------------------------------------------
    df_inputs = list_paths_s2(start_date, end_date)
    if df_inputs.empty:
        raise Exception("Paths to source files not found!")

    # Get extents of the output image
    # -------------------------------
    if resolution == "10m":
        main_extents = output_image_extent(df_inputs['image_10m'], bbox)
    elif resolution == "20m":
        main_extents = output_image_extent(df_inputs['image_20m'], bbox)
    else:
        raise Exception(f"Value  {resolution}  is not valid resolution.")

    # Obtain properties of output array (same for all bands/images)
    out_extents = main_extents['bounds']
    out_w = main_extents['width']
    out_h = main_extents['height']
    nr_bands = main_extents['bandsCount']

    # Initiate arrays (good obs, valid obs, output image)
    print("Preparing data for processing.")
    nobs = np.zeros((out_h, out_w), dtype=np.int8)
    nok = nobs.copy()
    composite = np.zeros((nr_bands, out_h, out_w), dtype=np.float32)

    # Metadata for both 10m and 20m (only 20m mask is used)
    df_inputs['meta_10m'] = pd.Series(
        collect_meta(list(df_inputs['image_10m']))
    )
    df_inputs['meta_20m'] = pd.Series(
        collect_meta(list(df_inputs['image_20m']))
    )

    # FILTER LIST OF IMAGES (if any fall out of bounds)
    if main_extents['skip']:
        df_inputs = df_inputs.drop(index=main_extents['skip'])
        df_inputs.reset_index(drop=True, inplace=True)

    # =========================== #
    # MAIN LOOP #
    # =========================== #
    # try:
    # Open all files and append to data frame
    # The files stay open during the loop; CLOSE THEM AT THE END!
    # df_inputs["src_files_20m"] = open_rio_list(df_inputs["image_20m"])
    # df_inputs["src_masks_20m"] = open_rio_list(df_inputs["mask_20m"])
    # if resolution == "10m":
    #     df_inputs["src_files_10m"] = open_rio_list(df_inputs["image_10m"])

    # seconds = 0.1
    # nok_value = 42
    # other = (seconds, nok_value)
    other = (main_extents, df_inputs,
             resolution, mask_crt,
             mask_thr, medoid_distance
             )

    # PROCESS LINES IN PARALLEL
    print("Start processing in parallel...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(process_one_line, line, other) for line in range(out_h)]

        for f in concurrent.futures.as_completed(results):
            idy, r_comp, r_nobs, r_nok = f.result()
            nobs[idy, :] = r_nobs
            nok[idy, :] = r_nok
            composite[:, idy, :] = r_comp
            print(f" >> Done value: {idy+1}")

    # # LOOP OVER ALL PIXELS
    # for y in range(out_h):
    #     pix_rslt = process_one_line(y, other)
    #
    #     idy, r_comp, r_nobs, r_nok = pix_rslt
    #     nobs[idy, :] = r_nobs
    #     nok[idy, :] = r_nok
    #     composite[:, idy, :] = r_comp
    #     print(f" >> Done line: {idy+1}")

    # Message when the final pixel was calculated
    print("     DONE!")

    # Close all data sets
    # close_rio_list(df_inputs["src_files_20m"])
    # close_rio_list(df_inputs["src_masks_20m"])
    # if resolution == "10m":
    #     close_rio_list(df_inputs["src_files_10m"])

    # #### #
    # STOP #
    # #### #
    time_b = time.time()
    print(f"-- Total processing time: {time_b-time_a} --")
    # except Exception as e:
    #     print("\n")
    #     print(e)
    # finally:
    # Save composite
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    out_nam = save_nam + "_composite.tif"
    out_pth = os.path.join(save_dir, out_nam)

    if resolution == "10m":
        meta_pth = df_inputs.loc[0, "image_10m"]
    else:
        meta_pth = df_inputs.loc[0, "image_20m"]

    with rasterio.open(meta_pth) as sample:
        meta_out = sample.profile.copy()

    x_lf_out = out_extents[0]
    y_up_out = out_extents[3]
    af_out = meta_out.get('transform')
    out_trans = Affine(af_out.a, 0.0, x_lf_out, 0.0, af_out.e, y_up_out)

    meta_out.update(
        height=composite.shape[1], width=composite.shape[2],
        transform=out_trans, bigtiff="yes"
        )

    with rasterio.open(out_pth, "w", **meta_out) as dest:
        dest.write(composite)

    # Save nok mask
    out_nam = save_nam + "_nok.tif"
    out_pth = os.path.join(save_dir, out_nam)
    nok_meta = meta_out.copy()
    nok_meta.update(
        count=1,
        dtype="int8"
        )

    with rasterio.open(out_pth, "w", **nok_meta) as dest:
        dest.write(np.expand_dims(nok, axis=0))

    # Save nobs mask
    out_nam = save_nam + "_nobs.tif"
    out_pth = os.path.join(save_dir, out_nam)
    with rasterio.open(out_pth, "w", **nok_meta) as dest:
        dest.write(np.expand_dims(nobs, axis=0))


if __name__ == "__main__":
    main()
