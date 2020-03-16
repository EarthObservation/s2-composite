# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:09:31 2020

@author: ncoz

This code will be used for compositing based on max NDVI
"""

import numpy as np
import glob
import os
import time
import rasterio
from rasterio.windows import Window
import math
import pandas as pd
from affine import Affine
import logging
# import pickle


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
        Number that was rounded to the nearest multiple of the pix_lefte.

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
        print('Opening raster {}'.format(fp[109:-24]))
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
            # print(tif_ext[idx])
            xmin_img, ymin_img, xmax_img, ymax_img = [a for a in tif_ext[idx]]

            # Round bbox to nearest multiple of raster
            xmin_box = round_multiple(bbox[0], xmin_img, pix_x)
            ymin_box = round_multiple(bbox[1], ymax_img, pix_y)
            xmax_box = round_multiple(bbox[2], xmin_img, pix_x)
            ymax_box = round_multiple(bbox[3], ymax_img, pix_y)

            # Check if bbox falls out of image extents (True if it falls out)
            chk_bbox = (xmin_img > xmax_box or ymin_img > ymax_box or
                        xmax_img < xmin_box or ymax_img < ymin_box)
            # print(chk_bbox)
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


def isCloud(s_pix):
    # TODO: Unfinished, at the moment it is ignored, but may be used in future

    r_b3b11 = s_pix.b3 / s_pix.b11    # Ratio b3/b11
    # r_b11b3 = s_pix.b11 / s_pix.b03    # Ratio b3/b11
    rgb_mean = (s_pix.b2 + s_pix.b3 + s_pix.b4) / 3    # Brightness
    # Normalised difference between b8 and b11
    nd_b8b11 = (s_pix.b8 - s_pix.b1) / (s_pix.b8 + s_pix.b11)
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

    # Test for if it is not snow
    test_snow = ndsi > 0.7 and not (r_b3b11 > 1 and tcb > 0.36)
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
    test_hpc = hpc1 or hpc2 or hpc3 and hpc4 or hpc5
    # Test if it is low probability cloud
    test_lpc = None

    if test_snow:
        cloud_test = True
    elif test_hpc:
        cloud_test = True
    elif test_lpc:
        cloud_test = True
    else:
        cloud_test = False

    return cloud_test


def medoid(calc_pix, dist='euclid'):
    s = pd.Series(index=calc_pix.index, dtype='float')
    for ii, rowi in calc_pix.iterrows():
        e_dist = 0
        for ij, rowj in calc_pix.drop([ii]).iterrows():
            # Calculate euclidian distance
            if dist == 'euclid':
                e_dist += np.sqrt(((rowj - rowi) ** 2).sum())
            elif dist == 'norm_diff':
                # e_bnd[bd] = (xB - xA)**2
                # abs((xB - xA) / (xB + xA))
                e_dist += abs((rowj - rowi) / (rowj + rowi)).sum()
            else:
                print('Error: Unknown distance for MEDOID.')
                raise Exception
        # Update column
        s.loc[ii] = e_dist

    return s


def stc_s2(calc_pix):
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


def select_pixel(calc_pix, nok_one, snow_df, medoid_distance):
    if nok_one == 0:
        # print('No valid pixels')
        sel_pix_idx = None
    elif nok_one == 1:
        # print('Only one valid pixel')
        sel_pix_idx = calc_pix.index[0]
    elif nok_one < 4:
        # print(f'Found {nok_one} valid pixels --> STC')
        stc = stc_s2(calc_pix)

        # Set criteria for STC
        criteria1 = (stc.mNDWI.mean() < -0.55
                     and stc.NDVI.max() - stc.NDVI.mean() < 0.05)
        criteria2 = (stc.NDVI.mean() < -0.3
                     and stc.mNDWI.mean() - stc.NDVI.min() < 0.05)
        criteria3 = stc.NDVI.mean() > 0.6 and stc.TCB.mean() < 0.45
        idx_tcb_min = stc.TCB.idxmin()
        # criteria4 = not isCloud(calc_pix.loc[idx_tcb_min])
        criteria4 = False
        criteria5a = not snow_df[idx_tcb_min] and stc.TCB.min() < 1
        criteria5b = not snow_df[idx_tcb_min] and stc.TCB.min() > 1
        criteria6 = stc.NDVI.mean() < -0.2
        criteria7 = stc.TCB.mean() > 0.45

        if criteria1:
            sel_pix_idx = stc.NDVI.idxmax()
        elif criteria2:
            sel_pix_idx = stc.mNDWI.idxmax()
        elif criteria3:
            sel_pix_idx = stc.NDVI.idxmax()
        elif criteria4:
            sel_pix_idx = idx_tcb_min
        elif criteria5a:
            sel_pix_idx = idx_tcb_min
        elif criteria5b:
            sel_pix_idx = None  # TODO: No valid pixels, set nan
        elif criteria6:
            sel_pix_idx = stc.mNDWI.idxmax()
        elif criteria7:
            sel_pix_idx = stc.NDVI.idxmin()
        else:
            sel_pix_idx = stc.NDVI.idxmax()
    else:
        # print(f'Found {nok_one} valid pixels --> MEDOID')
        # Calculate sum of distances using the selected method
        med = medoid(calc_pix, dist=medoid_distance)

        # Select pixel with min MEDOID value (shortest distance to others)
        sel_pix_idx = med.idxmin()

    return sel_pix_idx


def list_paths_s2(main_dir, year, months):
    # main_dir = 'Q:\\Sentinel-2_atm_10m_mosaicked_d96'
    # # 'Q:\\Sentinel-2_atm_20m_mosaicked_d96'

    year = str(year)

    q = os.path.join(main_dir, year) + "\\" + year + "*"
    tif_dirs = glob.glob(q)

    selected = []
    for m in months:
        by_m = year + "\\" + year + f"{m:02d}"
        selected = selected + [s for s in tif_dirs if by_m in s]

    tif_list = []
    tif_masks = []
    for d in selected:
        qq = os.path.join(d, "*_p2atm_d96tm.tif")
        pp = os.path.join(d, "*_p2atm_mask_d96tm.tif")

        tif_f = glob.glob(qq)
        tif_m = glob.glob(pp)

        tif_list.append(tif_f[0])
        tif_masks.append(tif_m[0])

    return tif_list, tif_masks


def open_rio_list(df_inputs):
    src_files = []
    for item in df_inputs:
        src_files.append(rasterio.open(item))

    return src_files


def close_rio_list(ds_list):
    # TODO: TESTIRAJ!!!!!
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


def main():
    # ========= TEMPORARY INPUT ============================================
    # bbox = [348890, 100000, 631610, 140000]
    bbox = [500000, 100000, 600000, 140000]
    # bbox = [599000, 100000, 600000, 101000]

    # Set composite time frame (currently Jan 2019)
    y = 2019
    m = [1]
    mdir_10m = 'Q:\\Sentinel-2_atm_10m_mosaicked_d96'
    mdir_20m = 'Q:\\Sentinel-2_atm_20m_mosaicked_d96'
    tif_list10, tif_masks10 = list_paths_s2(mdir_10m, y, m)
    tif_list20, tif_masks20 = list_paths_s2(mdir_20m, y, m)

    # Medoid method
    medoid_distance = "euclid"  # Either "euclid" or "norm_diff"

    # Mask/filtering
    mask_crt = "less_than"  # Either "less_than" or "all_bad"
    mask_thr = 35

    # Save paths/dir/names...
    save_dir = ".\\test_s2gm"
    save_nam = "test05"
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
    inputs = {"image_10m": tif_list10,
              "mask_10m": tif_masks10,
              "image_20m": tif_list20,
              "mask_20m": tif_masks20
              }
    df_inputs = pd.DataFrame(inputs)

    # TODO: Execute if list is not empty, if empty raise exception
    # Get extents of the output image
    # -------------------------------
    main_extents = output_image_extent(df_inputs['image_20m'], bbox)

    # Obtain properties of output array (same for all bands/images)
    out_extents = main_extents['bounds']
    out_w = main_extents['width']
    out_h = main_extents['height']
    nr_bands = main_extents['bandsCount']

    # Initiate arrays for storing number of available & good observations
    nobs = np.zeros((out_h, out_w), dtype=np.int8)
    nok = nobs.copy()

    # Initiate main array
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
    try:
        # Open all files and append to data frame
        # The files stay open during the loop; CLOSE THEM AT THE END!
        df_inputs['src_files_20m'] = open_rio_list(df_inputs['image_20m'])
        df_inputs['src_masks_20m'] = open_rio_list(df_inputs['mask_20m'])

        # LOOP OVER ALL PIXELS
        # for (y, x), _ in np.ndenumerate(nobs):
        for (y, x) in [(776, 816)]:
            t_st = time.time()
            # print(f"(x, y) = ({x}, {y})")
            if x == 0 and y == 0:
                print(f"   Processing line {y+1} of {out_h}...", end="")
            elif x == 0:
                print("     DONE!")
                print(f"   Processing line {y + 1} of {out_h}...", end="")

            # Coordinates of the selected pixel (pixel center)
            x_coord = out_extents[0] + (x + 0.5) * main_extents['pixels'][0]
            y_coord = out_extents[3] - (y + 0.5) * main_extents['pixels'][1]

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
                # src_id = df_inputs.index.get_loc(ind)
                # 1\ CHECK IF PIXEL EXISTS (in the current image)
                # ==============================================================
                # TODO: select correct meta (10m or 20m)
                chk_bbx = (
                        row['meta_20m']['bounds'].right
                        >= x_coord
                        >= row['meta_20m']['bounds'].left
                )
                chk_bby = (
                    row['meta_20m']['bounds'].top
                    >= y_coord
                    >= row['meta_20m']['bounds'].bottom
                )

                if not (chk_bbx and chk_bby):
                    # print("skip: out of bounds")
                    continue
                else:
                    # Add one to valid observations
                    nobs_one += 1

                # 2\ CHECK IF PIXEL IS GOOD (always use 20m mask)
                # ==============================================================
                # Get location of this pixel in mask
                win = pixel_offset(row['meta_20m'], x_coord, y_coord)
                # Read mask value
                opm = row['src_masks_20m'].read(window=win)
                # Determine pixel type (snow/valid/bad)
                crt = get_mask_value(opm[0, 0, 0], mask_crt, mask_thr)

                if crt == "snow":
                    # Read the pixel so it can be checked for snow
                    pix_20m = row["src_files_20m"].read(window=win).flatten()
                    # Skip if pixel is zero (class. problem at swath border)
                    pixel_check = (
                        any(np.isnan(pix_20m))
                        or any(np.isinf(pix_20m))
                        or any(pix_20m == 0)
                    )
                    if pixel_check:
                        continue

                    # Check for snow
                    is_snow = isSnow(pix_20m)
                    if is_snow:
                        nok_one += 1
                        one_pixel = pix_20m
                    else:
                        continue
                elif crt == "bad":  # not valid pixels
                    continue
                else:
                    # Pixel is valid
                    one_pixel = row["src_files_20m"].read(window=win).flatten()
                    # Skip if pixel (any band) is zero/nan/inf
                    # (classification problem at swath border, mask shows good
                    # pixel whe it is in fact bad)
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

                # TODO:
                # 4\ IF THIS IS 10m image, then read pixel from 20m for missing bands
                #
                # 10m bands: 1, 2, 3 are the same, 4 is put into 7 slot
                #
                # to read from 20m: 4, 5, 6 and 8, 9, 10
                #
                # ==============================================================

                # \5 Populate Pandas Data Frame
                # ==============================================================
                # \5.1 Read from 10m or 20m
                calc_pix.loc[ind, 'b2'] = one_pixel[0]
                calc_pix.loc[ind, 'b3'] = one_pixel[1]
                calc_pix.loc[ind, 'b4'] = one_pixel[2]
                calc_pix.loc[ind, 'b8'] = one_pixel[6]
                # \5.2 Always read from 20m
                calc_pix.loc[ind, 'b5'] = one_pixel[3]
                calc_pix.loc[ind, 'b6'] = one_pixel[4]
                calc_pix.loc[ind, 'b7'] = one_pixel[5]
                calc_pix.loc[ind, 'b8A'] = one_pixel[7]
                calc_pix.loc[ind, 'b11'] = one_pixel[8]
                calc_pix.loc[ind, 'b12'] = one_pixel[9]
                # \5.3 Add isSnow test result
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
                sel_pix = np.full((nr_bands,), np.nan, dtype='float32')

            # # Populate the result matrices
            # sel_pix = np.reshape(sel_pix, (sel_pix.size, 1, 1))
            composite[:, y, x] = sel_pix
            nobs[y, x] = nobs_one
            nok[y, x] = nok_one

            # print(f" --> {time.time()-t_st} sec.")

        # Message when the final pixel was calculated
        print("     DONE!")

        # Close all data sets
        close_rio_list(df_inputs["src_files_20m"])
        close_rio_list(df_inputs["src_masks_20m"])

        # #### #
        # STOP #
        # #### #
        time_b = time.time()
        print(f"-- Total processing time: {time_b-time_a} --")
    finally:
        # Save composite
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        out_nam = save_nam + "_composite.tif"
        out_pth = os.path.join(save_dir, out_nam)

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
