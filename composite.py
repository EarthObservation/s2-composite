# -*- coding: utf-8 -*-
"""

@author: ncoz

@copyright: ZRC SAZU (Novi trg 2, 1000 Ljubljana, Slovenia)

@history:
    Created on Fri Feb  7 10:18:07 2020
"""

import rasterio
# from rasterio.mask import mask
from rasterio.windows import Window
from shutil import rmtree
import os
import time
import math
# import geopandas as gpd
# from shapely.geometry import box
import numpy as np

import dask.array as da
import pickle
from affine import Affine
import cv2


def round_multiple(nr, xL, pix):
    """Rounds to the nearest multiple of the raster. Used when evaluating the
    geographical

    Parameters
    ----------
    nr : float
        Number that needs to be rounded.
    xL : float
        Upper-left x or y coordinate of the raster.
    pix : float
        Pixel size in the selected direction.

    Returns
    -------
    float
        Number that was rounded to the nearest multiple of the pixle.

    """
    return pix * round((nr - xL) / pix) + xL


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
    check = check_list.count(check_list[0]) == len(check_list)
    if not check:
        raise Exception(msg)


def get_mask_idx(fp, offset, cMask="all_bad", dilate=1):
    """Will find mask from Sentinel-2 results (renaming the input path),
    based on the selected method, it will determine bad pixels and return
    indexes and number where bad pixels occur.

    Parameters
    ----------
    fp : TYPE
        DESCRIPTION.
    offset : TYPE
        DESCRIPTION.
    cMask : str, optional
        DESCRIPTION. The default is "all_bad".
    dilate : int, optional
        Determines kernel size for dilation with a square kernel of size
        (2n+1). If set to 0, no dilation will be preformed. If negative the
        mask will be insted eroded on the same priniciple. The default is 1

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    idxBad : TYPE
        DESCRIPTION.
    nBad : TYPE
        DESCRIPTION.

    """
    # INSERT MASK INTO THE FILENAME
    new = fp.split("_")
    new.insert(-1, 'mask')
    fp_m = "_".join(new)

    # Read mask array
    if os.path.isfile(fp_m):
        src = rasterio.open(fp_m)
        mask_array = src.read(1, window=offset)
        src.close()

    # Dilate to increase good areas by 1 pixel (ignore for background)
    if dilate == 0:
        dilated = mask_array
    elif dilate < 0:
        # Erode if negative
        krn = 2 * abs(dilate) + 1
        kernel = np.ones((krn, krn), np.uint8)
        dilated = cv2.erode(mask_array, kernel, iterations=1)
    else:
        krn = 2 * dilate + 1
        kernel = np.ones((krn, krn), np.uint8)
        dilated = cv2.dilate(mask_array, kernel, iterations=1)

    # Get index of pixels
    if cMask == "background":
        threshold = 10
    elif cMask == "clouds/snow":
        threshold = 33
    elif cMask == "clouds/snow/haze":
        threshold = 34
    elif cMask == "clouds/snow/haze/shadow":
        threshold = 40
    elif cMask == "all_bad_keep50":
        dilated[dilated == 50] = 100
        threshold = 100
    elif cMask == "all_bad":
        threshold = 100
    else:
        raise Exception("{} - no such keyword for mask in"
                        " get_mask_idx.".format(cMask)
                        )

    # Get pixels that will be masked
    if cMask[:7] == "all_bad":
        idxBad = np.where(dilated != threshold)
        nBad = idxBad[0].size
    else:
        idxBad = np.where(dilated <= threshold)
        nBad = idxBad[0].size

    return idxBad, nBad


def output_image_extent(src_fps, bbox):
    """Determines the maximum extents of the output image, taking into account
    extents of all considered files and a user specified boundig box.

    Parameters
    ----------
    src_fps : list
        List of file paths to input TIFs.
    bbox : list
        Extents in order [xL, yD, xR, yU].

    Raises
    ------
    Exception
        Error is raised if bounding box doesn't overlap one of the images.
        Fututre plan: exclude that image from file list??

    Returns
    -------
    output : dictionary
        Contains extents of and other metadata required for final image.

    """
    str_time = time.time()

    # Open TIF files as DataSets and evaluate properties
    tif_ext, pix_res, src_all, bnd_num = ([] for _ in range(4))

    for fp in src_fps:
        print('Opening raster {}'.format(fp[109:-24]))
        src = rasterio.open(fp)
        src_all.append(src)

        # Read raster properties
        tif_ext.append([i for i in src.bounds])  # left, bottom, right, top
        pix_res.append(src.res)
        bnd_num.append(src.count)

        src.close()

    # Check if all images have the same pixel size
    is_list_uniform(pix_res, 'Pixel size of input images not matching.')

    # Check if all images have the same number of bands
    is_list_uniform(bnd_num, 'Number of bands in input images not matching.')

    # Pixle size
    pix_x, pix_y = pix_res[0]

    # FIND MAX EXTENTS AND DETERMINE SIZE OF OUTPUT IMAGE
    xL_out = min((list[0] for list in tif_ext))
    yD_out = min((list[1] for list in tif_ext))
    xR_out = max((list[2] for list in tif_ext))
    yU_out = max((list[3] for list in tif_ext))

    if bbox:
        # Round bbox to nearest multiple of raster
        xL_bbox = round_multiple(bbox[0], xL_out, pix_x)
        yD_bbox = round_multiple(bbox[1], yU_out, pix_y)
        xR_bbox = round_multiple(bbox[2], xL_out, pix_x)
        yU_bbox = round_multiple(bbox[3], yU_out, pix_y)

        # Break if bbox falls out of image extents
        chk_bbox = (xL_out > xR_bbox or yD_out > yU_bbox or
                    xR_out < xL_bbox or yU_out < yD_bbox)
        if chk_bbox:
            raise Exception('BBOX out of image extents')

        # Compare with image extents
        xL_out = max(xL_out, xL_bbox)
        yD_out = max(yD_out, yD_bbox)
        xR_out = min(xR_out, xR_bbox)
        yU_out = min(yU_out, yU_bbox)

    # Calculate size of output array
    tif_wide = int(math.ceil(xR_out - xL_out) / abs(pix_x))
    tif_high = int(math.ceil(yU_out - yD_out) / abs(pix_y))
    nr_bands = bnd_num[0]
    nr_image = len(src_fps)

    output = {'width': tif_wide,
              'height': tif_high,
              'bandsCount': nr_bands,
              'imgCount': nr_image,
              'bounds': [xL_out, yD_out, xR_out, yU_out],
              'pixels': (pix_x, pix_y)
              }

    end_time = time.time() - str_time
    print('--- Time to evaluate geo. extents: %s seconds ---' % (end_time))
    return output


def image_offset(out_ext, src_ds):
    # Output image extents
    xL_out, yD_out, xR_out, yU_out = out_ext

    # Load metadata from source dataset
    x_col = src_ds.meta['width']
    y_row = src_ds.meta['height']

    # Extents of image we are reading
    xL, yD, xR, yU = [xy for xy in src_ds.bounds]
    # Read pixel resolution
    pix_x, pix_y = src_ds.res

    # X-direction
    if (xL >= xL_out):
        # Offset
        win_x = 0
        off_x = int((xL - xL_out) / abs(pix_x))

        # Width
        bbx = int((xR_out - xL) / pix_x)
        len_x = min(x_col, bbx)

    else:
        # Offset
        win_x = int((xL_out - xL) / pix_x)
        off_x = 0

        # Width
        bb1 = int((xR - xL_out) / pix_x)  # src < out
        bb2 = int((xR_out - xL_out) / pix_x)  # src > out
        len_x = min(bb1, bb2)

    # Y-direction
    if (yU <= yU_out):
        # Offset
        win_y = 0
        off_y = int((yU_out - yU) / abs(pix_y))

        # Height
        bby = int((yU - yD_out) / pix_y)
        len_y = min(y_row, bby)

    else:
        # Offset
        win_y = int((yU - yU_out) / pix_y)
        off_y = 0

        # Height
        bb1 = int((yU_out - yD) / pix_y)
        bb2 = int((yU_out - yD_out) / pix_y)
        len_y = min(bb1, bb2)

    # Prepare Window for reading raster from TIF
    rd_win = Window(win_x, win_y, len_x, len_y)
    # Prepare intervals for slicing
    slicex = [off_x, (off_x + len_x)]
    slicey = [off_y, (off_y + len_y)]

    # Calculate index for inserting image into output array (and Window)
    return rd_win, slicex, slicey


# ============================================================================
def composite(src_fps, save_loc, save_nam,
              method="median", comp_mask="all_bad", bbox=None):

    # Prepare save location
    save_dir = os.path.join(save_loc, save_nam)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Get extents
    main_extents = output_image_extent(src_fps, bbox)

    # Obtain propertis of output array (same for all bands/images)
    out_extents = main_extents['bounds']
    out_w = main_extents['width']
    out_h = main_extents['height']
    nr_bands = main_extents['bandsCount']

    # Initiate arrays for storing noumber of available & good observations
    nobs = np.zeros((out_h, out_w), dtype=np.int8)
    nok = nobs.copy()

    # Create temp dir if it doesn't exist
    sav_dir = '.\\tmp'
    if not os.path.exists(sav_dir):
        os.mkdir(sav_dir)

    # MAIN LOOP FOR COMPOSITING
    tTim_A = time.time()
    tmp_sav_pth = []
    for band in range(nr_bands):
        print("#\n# Creating composite for Band {}".format(band+1))
        comp_stack = []
        # Loop all images
        for i, fp in enumerate(src_fps):
            str_time = time.time()

            # Open data set
            src = rasterio.open(fp)

            # Save copy of profile for writing tiff at the end
            if band == 0 and i == 0:
                out_meta = src.profile.copy()

            print("#   Processing Image {}.".format(i+1))

            # Skip Reading the image if bbox is out of bounds
            xL, yD, xR, yU = [xy for xy in src.bounds]
            xL_out, yD_out, xR_out, yU_out = out_extents
            chk_bbox = (xL > xR_out or yD > yU_out or
                        xR < xL_out or yU < yD_out)
            if chk_bbox:
                print('#   Image {} not included (out of bounds).'.format(i))
                break

            # Calculate offset for reading and slicing
            win, sl_x, sl_y = image_offset(out_extents, src)

            # ------------------------------
            # Read image and store to pickle
            # ------------------------------
            # Set offset Window for reading of TIF subset
            offset = win

            # Initiate array for output
            comp_band = np.full((out_h, out_w), np.nan, dtype=np.float32)

            # Read image and save to pickle
            print("#     Reading the image.")
            if band == 0:
                tmp_read = src.read(window=offset)
                for nc in range(1, nr_bands):
                    img_nam = ('img' + str(i+1).zfill(2) + "_b"
                               + str(nc+1).zfill(2) + '.p')
                    img_pth = os.path.join(sav_dir, img_nam)
                    pickle.dump(tmp_read[nc], open(img_pth, "wb"))
                tmp_read = tmp_read[0]
            else:
                img_nam = ('img' + str(i+1).zfill(2) + "_b"
                           + str(band+1).zfill(2) + '.p')
                img_pth = os.path.join(sav_dir, img_nam)
                tmp_read = pickle.load(open(img_pth, "rb"))

            # Read the image into the array
            comp_band[sl_y[0]:sl_y[1], sl_x[0]:sl_x[1]] = tmp_read
            tmp_read = None
            src.close()

            # ------------------------------
            # determine bad pixels from mask
            # ------------------------------
            print("#     Determining bad pixels.")
            if band == 0:

                # Get index of mask
                idx_bad = get_mask_idx(fp, offset, comp_mask, dilate=-1)

                # Get index of background
                idx_bck = get_mask_idx(fp, offset, "background")

                # Update nok and nobs
                nobs[sl_y[0]:sl_y[1], sl_x[0]:sl_x[1]] += 1
                nok[sl_y[0]:sl_y[1], sl_x[0]:sl_x[1]] += 1

                nok[idx_bad[0][0]+sl_y[0], idx_bad[0][1]+sl_x[0]] += -1
                nobs[idx_bck[0][0]+sl_y[0], idx_bck[0][1]+sl_x[0]] += -1

                # Save index to pickle for later use
                idx_nam = 'idxBad_' + str(i+1).zfill(2) + '.p'
                idx_pth = os.path.join(sav_dir, idx_nam)
                pickle.dump(idx_bad, open(idx_pth, "wb"))
                idx_bck = None

            else:
                # Read from Pickle
                idx_nam = 'idxBad_' + str(i+1).zfill(2) + '.p'
                idx_pth = os.path.join(sav_dir, idx_nam)
                idx_bad = pickle.load(open(idx_pth, "rb"))

            # Apply mask to image
            if idx_bad[1] > 0:
                comp_band[idx_bad[0][0]+sl_y[0],
                          idx_bad[0][1]+sl_x[0]] = np.nan
                idx_bad = None

            # Stack comp_band array into Dask Array
            comp_stack.append(da.from_array(comp_band, chunks=(1024, 1024)))

            # Close the array to save memory
            comp_band = None

            end_time = time.time()
            print('#   --- Time: %s seconds ---' % (end_time-str_time))

        # Stack all images into 1 array
        stacked = da.stack(comp_stack, axis=0)

        # Calculate composite for selected method with dask
        print("# Compositing Band {}".format(band+1))
        str_time = time.time()
        if method == 'mean':
            comp_out = da.nanmean(stacked, axis=0, keepdims=True).compute()
        elif method == 'median':
            comp_out = da.nanmedian(stacked, axis=0, keepdims=True).compute()
        elif method == 'max':
            comp_out = da.nanmax(stacked, axis=0, keepdims=True).compute()
        elif method == 'min':
            comp_out = da.nanmin(stacked, axis=0, keepdims=True).compute()
        else:
            raise Exception('{} is not a valid compositing '
                            'method!'.format(method))
        end_time = time.time()
        print('# --- Time: %s seconds ---' % (end_time-str_time))

        # After one band is resolved, save to temp file and release memory by
        # deleting the array
        if nr_bands > 1:

            print('# Saving temporary composite file for this band.')

            # Create file name and save using pickle
            sav_fil = 'b_' + str(band+1).zfill(2) + '.p'
            sav_pth = os.path.join(sav_dir, sav_fil)
            pickle.dump(comp_out, open(sav_pth, "wb"))

            # Add to savePth list with filenames
            tmp_sav_pth.append(sav_pth)

            #  Clean up workspace
            comp_out = None

        tTim_B = time.time()
        print('--- Total time: %s seconds --- \n' % (tTim_B - tTim_A))

    # ----------------------------------------------------------------------------
    # OUT OF THE COMPOSITE LOOP RESTORE SAVED FILES AND BUIL TIF
    # ----------------------------------------------------------------------------
    if nr_bands > 1:

        print("# Restoring saved bands.")
        str_time = time.time()

        # Initiate output array
        comp_out = np.full((nr_bands, out_h, out_w), np.nan, dtype=np.float32)

        for bnd, pth in enumerate(tmp_sav_pth):
            comp_out[bnd, :, :] = pickle.load(open(pth, "rb"))

        # Remove temporary folder
        rmtree(sav_dir, ignore_errors=True)
        end_time = time.time()
        print('--- Time: %s seconds ---' % (end_time-str_time))

    # ----------------------------------------------------------------------------
    # SAVE RESULTS TO TIF
    # ----------------------------------------------------------------------------
    print("# Saving composite image to TIFF.")
    str_time = time.time()

    # Save composite
    out_nam = save_nam + "_composite.tif"
    out_pth = os.path.join(save_dir, out_nam)

    out_px = out_meta["transform"][0]
    out_py = out_meta["transform"][4]
    out_trans = Affine(out_px, 0.0, xL_out, 0.0, out_py, yU_out)

    out_meta.update(
        height=comp_out.shape[1], width=comp_out.shape[2],
        transform=out_trans, bigtiff="yes"
        )

    with rasterio.open(out_pth, "w", **out_meta) as dest:
        dest.write(comp_out)

    # Save nok mask
    out_nam = save_nam + "_nok.tif"
    out_pth = os.path.join(save_dir, out_nam)
    nok_meta = out_meta.copy()
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

    end_time = time.time()
    print('--- Time: %s seconds ---' % (end_time-str_time))

    tTim_B = time.time()
    print('\n--- Total time: %s seconds --- \n' % (tTim_B - tTim_A))


# ========================================================================
# TEMPORARY INPUT
# ========================================================================
# Select from: mean, median, max, min
method = 'median'
comp_mask = 'all_bad'

save_loc = "."
save_nam = "20202102_test_02"

bbox = None
#        # xL    # yD    # xR    # yU
bbox = [348890, 100000, 631610, 140000]
# bbox_C = [481380, 16390, 631610, 201590]  # E079 IMAGE
# bbox_D = [348890, 16390, 448580, 201590]  # W022 IMAGE
# out_extents = [348890, 16390, 631610, 201590]  # C122 IMAGE

file1 = ("E:\\sentinel_composite\\dev_src\\"
         "20190401T100031_S2A_MSIL2A_20190401T105727_C122_10m\\"
         "20190401T100031_S2A_MSIL2A_20190401T105727_C122"
         "_10m__ms_p2atm_d96tm.tif")

file2 = ('E:\\sentinel_composite\\dev_src\\'
         '20190406T100029_S2B_MSIL2A_20190406T125021_C122_10m\\'
         '20190406T100029_S2B_MSIL2A_20190406T125021_C122_10m'
         '__ms_p2atm_d96tm.tif')

file3 = ('E:\\sentinel_composite\\dev_src\\'
         '20190403T095029_S2B_MSIL2A_20190403T115455_E079_10m/'
         '20190403T095029_S2B_MSIL2A_20190403T115455_E079_10m'
         '__ms_p2atm_d96tm.tif')

file4 = ('E:\\sentinel_composite\\dev_src\\'
         '20190404T101031_S2A_MSIL2A_20190404T185546_W022_10m\\'
         '20190404T101031_S2A_MSIL2A_20190404T185546_W022_10m'
         '__ms_p2atm_d96tm.tif')

file5 = ('E:\\sentinel_composite\\dev_src\\'
         '20190408T095031_S2A_MSIL2A_20190408T102633_E079_10m\\'
         '20190408T095031_S2A_MSIL2A_20190408T102633_E079_10m'
         '__ms_p2atm_d96tm.tif')

file6 = ('E:\\sentinel_composite\\dev_src\\'
         '20190409T101029_S2B_MSIL2A_20190409T130030_W022_10m\\'
         '20190409T101029_S2B_MSIL2A_20190409T130030_W022_10m'
         '__ms_p2atm_d96tm.tif')

# List of paths to input files
src_fps = [file1, file2, file3, file4, file5, file6]
# ========================================================================
# RUN
composite(src_fps, save_loc, save_nam, method, comp_mask, bbox)
