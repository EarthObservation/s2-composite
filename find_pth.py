from datetime import date, timedelta
import os
import glob
import pandas as pd

# Dirs with source files
mdir_10m = "Q:\\Sentinel-2_atm_10m_mosaicked_d96"
mdir_20m = "Q:\\Sentinel-2_atm_20m_mosaicked_d96"

# Suffixes
sfx_img = "*_p2atm_d96tm.tif"
sfx_msk = "*_p2atm_d96tm.tif"

sdate = date(2019, 3, 1)   # start date
edate = date(2019, 3, 30)   # end date
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
        if not(chk1 == chk2):
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

df = pd.DataFrame(file_list)

print(df)
