import pandas as pd
import os

def do_normalization():
    raw_df = pd.read_csv("./preprocessed.csv")
    raw_df_copy = raw_df.copy()
    # use z-score normalization for PCA, save as reduction.csv
    for col in raw_df.columns:
        if col == "Status":
            continue
        std = raw_df[col].std()
        mean = raw_df[col].mean()
        raw_df[col] = (raw_df[col] - mean)/ std
    raw_df.to_csv("./preprocessing/reduction.csv", index=False)
    os.remove("./preprocessed.csv")

# use min-max normalization for feature selection, save as non_reduction.csv
# for col in raw_df_copy.columns:
#     if col == "Status":
#         continue
#     df = raw_df_copy[col]
#     raw_df_copy[col] = (df-df.min())/(df.max()-df.min())
# raw_df_copy.to_csv("./non_reduction.csv", index=False)
    
