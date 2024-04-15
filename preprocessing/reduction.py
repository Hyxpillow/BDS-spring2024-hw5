import pandas as pd
from sklearn.decomposition import PCA
import numpy as np


def do_reduction():
    raw_df = pd.read_csv("./preprocessing/reduction.csv")
    tmp = raw_df['Status']
    raw_df = raw_df.drop(['Status'], axis=1)
    # Firstly, do the following and get all eigenvalues
    # pca = PCA()
    # pca.fit(raw_df)
    # print(pca.explained_variance_)

    # [3.75618438e+00 2.13554222e+00 1.90353074e+00 1.74258657e+00
    # 1.45309662e+00 1.34125762e+00 1.23976281e+00 1.14441611e+00
    # 1.02922215e+00 9.94301289e-01 9.32137969e-01 9.02665083e-01
    # 7.93158812e-01 7.31150496e-01 4.75769237e-01 2.16003665e-01
    # 1.66991147e-01 4.22230933e-02 3.08512173e-30 1.93172076e-30
    # 1.77342814e-31]

    # pick PC1 ~ PC12
    pca = PCA(n_components=12)
    pca.fit(raw_df)
    f = pd.DataFrame(pca.transform(raw_df))
    f['Status'] = tmp
    f.to_csv("./preprocessing/reduction.csv", index=False)