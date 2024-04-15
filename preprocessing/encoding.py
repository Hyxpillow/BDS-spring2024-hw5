import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def do_encoding():
    raw_df = pd.read_csv("./Breast_Cancer_dataset.csv")
    numeric_cols = ['Age','Tumor Size','Regional Node Examined','Reginol Node Positive','Survival Months']
    categorical_cols = ['Race','Marital Status','N Stage','6th Stage','differentiate','Grade','A Stage','T Stage ','Estrogen Status','Progesterone Status']
    # for col in raw_df.columns:  # To see the unique set of each categorical column, for encoding.
    #     print(col, raw_df[col].unique())
    raw_df['T Stage '] = raw_df['T Stage '].map({'T1': 0,'T2': 1, "T3": 2, "T4": 3})
    raw_df['N Stage'] = raw_df['N Stage'].map({'N1': 0,'N2': 1, "N3": 2})
    raw_df['A Stage'] = raw_df['A Stage'].map({'Regional': 0,'Distant': 1})
    raw_df['6th Stage'] = raw_df['6th Stage'].map(
        {'IIA' : 0,'IIB': 1, "IIIA": 2, "IIIB": 3, "IIIC": 4})
    raw_df['Estrogen Status'] = raw_df['Estrogen Status'].map({'Positive': 0,'Negative': 1})
    raw_df['Grade'] = raw_df['Grade'].map({
        '1': 0,
        '2': 1,
        '3': 2,
        ' anaplastic; Grade IV': 3,
        })
    raw_df['differentiate'] = raw_df['differentiate'].map(
        {'Undifferentiated' : 0,
        'Poorly differentiated': 1, 
        "Moderately differentiated": 2, 
        "Well differentiated": 3,})
    raw_df['Progesterone Status'] = raw_df['Progesterone Status'].map(
        {'Positive' : 1,'Negative': 0})
    raw_df['Status'] = raw_df['Status'].map({'Alive' : 1,'Dead': 0})
    # Now, since there is no strong relationship between "Race" and "Marital Status, use one hot encoding"

    onehot_categories = ["Race", "Marital Status"]
    encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore').fit(raw_df[onehot_categories])
    encoded_cols = list(encoder.get_feature_names_out(onehot_categories))
    raw_df[encoded_cols] = encoder.transform(raw_df[onehot_categories])
    raw_df = raw_df.drop(onehot_categories, axis=1)
    # Before encoding
    # 68    White   Married T1  N1  IIA Poorly  differentiated  3   Regional    4   Positive    Positive    24  1   60  Alive
    # After encodeing 
    # 68    0.0	0.0	1.0	0.0	1.0	0.0	0.0	0.0	 1	1 	1	2	3	1	4	1	1	24	1	60	1
    raw_df.to_csv("./preprocessed.csv", index=False)