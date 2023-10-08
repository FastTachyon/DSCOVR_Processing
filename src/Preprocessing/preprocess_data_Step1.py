import numpy as np
import pandas as pd
import glob
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import datetime
from scipy import signal

from tqdm import tqdm
import os

datafiles = glob.glob("data/*.zip")
headers = ["timestamp", "Bx", "By", "Bz"] + [f"E_bin_{i:02d}" for i in range(0,50)]

for datafile in tqdm(datafiles, unit="year"):

    df = pd.read_csv(datafile, compression='zip', names=headers, sep=',', quotechar='"', parse_dates=[0], na_values='0',)
    theyear = int(os.path.split(datafile)[-1].split("_")[-2])

    print(f"Loaded {len(df)} rows from {datafile} (Year {theyear})")

    # Add magnitude of magnetic field
    df["Bmag"] = np.sqrt(df.Bx**2 + df.By**2 + df.Bz**2)


    ## Calculate the average energy and the energy variance
    # ref. https://stackoverflow.com/a/50786849

    print("Computing average energy and energy variance")
    A =  df.iloc[:, 4:54].fillna(0).values
    n_rows = len(A)
    bins = np.arange(A.shape[1]+1)
    mids = 0.5*(bins[1:] + bins[:-1])

    sums = np.empty(A.shape[0])
    avgs = np.empty(A.shape[0])
    vars = np.empty(A.shape[0])
    for i, n in tqdm(enumerate(A), unit="row", total=n_rows):
        try:
            sums[i] = n.sum()
            avgs[i] = np.average(mids, weights=n)
            vars[i] = np.average((mids - avgs[i])**2, weights=n)
        except ZeroDivisionError:
            sums[i] = np.nan
            avgs[i] = np.nan
            vars[i] = np.nan

    df["Esum"] = sums
    df["Eavg"] = avgs
    df["Evar"] = vars
    # This is 1 if spectrum is unavaibable for current time
    df["NaSpectra"] = (avgs!=avgs).astype(int)

    ## Clean up
    # df.drop(columns = [c for c in df.columns if "spectrum_bin" in c], inplace=True)
    df.set_index("timestamp", inplace=True)
    df_range = (df.index[0], df.index[-1])
    print(df.head(15))

    ## Filtering
    # (fc 1/.60 since we want to  fc = 1H and sata is sampled every minute)
    b, a = signal.butter(4, 1./60)
    for col in df.columns:
       df[col] = df[col].interpolate()
       df[col] = signal.filtfilt(b, a, df[col], padlen=150)

    df = df[~df.index.duplicated()]

    ## Add Omniweb data (https://isgi.unistra.fr/data_download.php)
    odf = pd.read_table("data/omniweb.gsfc.nasa.gov_staging_omni2_a5LREcGZHO.lst", delim_whitespace=True, names=["Y", "D", "H", "Kp10","RSP", "DST", "AP", "SOLARF10", "AE", "AL", "AU", "PCN", "LYMANALPHASOLAR"])
    odf["datetime_str"] = odf.Y.map('{:04d}'.format) + odf.D.map('{:03d}'.format) + odf.H.map('{:02d}'.format)
    odf["timestamp"] = odf.apply(lambda x: datetime.datetime.strptime(x["datetime_str"], "%Y%j%H"), axis=1)
    odf.drop(["datetime_str", "Y", "D", "H"], axis=1, inplace=True)
    odf.set_index("timestamp", inplace=True)

    # Keeping only data of interest
    odf = odf[np.logical_and(odf.index>=df_range[0], odf.index<=df_range[1])]

    ## Merging the data
    print("Merging the data")
    tdf = pd.merge(df, odf, how='outer', left_index=True, right_index=True).drop_duplicates()

    freq = "1H"
    tdf_i = tdf.resample(freq).interpolate(method='linear')

    # Plotting
    print("Plotting...")
    plotdir = f"plots/{theyear}"
    os.makedirs(plotdir, exist_ok=True)
    for themonth in tqdm(range(1,13), unit="plot"):
        yearend = theyear
        if themonth == 12:
            monthend = 1
            yearend = theyear+1
        else:
            monthend = themonth+1
            yearend = theyear

        fig, ax = plt.subplots(figsize=(16,9))
        tdf_i.plot(y=["Kp10", "Eavg", "AP", "DST"], xlim=(datetime.datetime(theyear,themonth,1), datetime.datetime(yearend,monthend,1,)), ax=ax)
        plt.savefig(f"{plotdir}/{theyear}-{themonth:02d}.png")

    # Exporting
    print("Exporting...")
    outdir = f"out/"
    tdf_i.to_csv(os.path.join(outdir, f"data_{theyear}_{freq}.csv.zip"), compression="infer")