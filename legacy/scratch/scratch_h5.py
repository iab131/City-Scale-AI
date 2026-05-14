import h5py
import numpy as np

try:
    with h5py.File("data/METR-LA.h5", "r") as f:
        print(list(f.keys()))
        if 'df' in f:
            df = f['df']
            print(list(df.keys()))
            if 'block0_values' in df:
                print("Values shape:", df['block0_values'].shape)
except Exception as e:
    print("Error:", e)
