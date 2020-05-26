import rasterio
from rasterio.windows import Window
import time


pth = ("q:\\Sentinel-2_atm_20m_mosaicked_d96\\2019\\"
       "20190106T100409_S2B_MSIL2A_20190106T134029_C122_20m\\"
       "20190106T100409_S2B_MSIL2A_20190106T134029_C122_20m__ms_p2atm_d96tm.tif"
       )

st1 = time.time()
with rasterio.open(pth) as ds:
    st2 = time.time()
    print("\nREAD LINE")
    print(f"Time to open: {st2-st1} seconds")
    lines = [1, round(ds.height/2), ds.height-1]
    for line in lines:
        st = time.time()
        win = Window(0, line, ds.width, 1)
        read_line = ds.read(window=win)
        en = time.time() - st
        print(f"   Time to read line {line}: {en:.2e} seconds")

    print("\nREAD PIXEL")
    pixels = [1, round(ds.height / 2), ds.height - 1]
    read_pix = []
    for pix in pixels:
        st = time.time()
        win = Window(pix, pix, 1, 1)
        read_pix.append(ds.read(window=win))
        dt = time.time()
        en = dt - st
        print(f"   Time to read pixel {pix}: {en:.2e} seconds")

    # read entire column
    print("\nREAD COLUMN")
    columns = [1, round(ds.width/2), ds.width-1]
    for col in columns:
        st = time.time()
        win = Window(col, 0, 1, ds.height)
        read_col = ds.read(window=win)
        en = time.time() - st
        print(f"   Time to read column {col}: {en} seconds")

    # read entire band
    print("\nREAD BANDS")
    bnds = [1, round(ds.count), ds.count]
    for bnd in bnds:
        st = time.time()
        read_band = ds.read(bnd)
        en = time.time() - st
        print(f"   Time to read band {bnd}: {en} seconds")
