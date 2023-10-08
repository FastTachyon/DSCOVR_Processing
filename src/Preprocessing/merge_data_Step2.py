import pandas as pd
import glob
from tqdm import tqdm

datafiles = glob.glob("out/data_*.zip")
df = pd.read_csv(datafiles[0], compression='zip', index_col=0)

for datafile in tqdm(datafiles[1:], unit="year"):
    df = pd.concat((df, pd.read_csv(datafile, compression='zip', index_col=0)))

print("First few rows")
print(df.head())

print("Last few rows")
print(df.tail())

df.to_csv(f"out/all_data_v1.1.csv.zip", compression="infer")