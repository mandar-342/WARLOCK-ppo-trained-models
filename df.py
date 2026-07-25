import pandas as pd
import numpy as np

df = pd.read_parquet("data/features/train.parquet")

print("=" * 100)

for col in df.columns:
    if df[col].isna().any():
        idx = np.where(df[col].isna())[0]

        print(f"\nCOLUMN : {col}")
        print(f"NaN Count : {len(idx)}")
        print(f"First Index : {idx[0]}")
        print(df.loc[idx[0]-2:idx[0]+2, [col]])