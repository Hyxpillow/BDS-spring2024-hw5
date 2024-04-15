from numpy import sort
import pandas as pd

def look_for_outliers():
    raw_df = pd.read_csv("./preprocessed.csv")
    # Look for outliers by eyes. 
    
    # for col in raw_df.columns:  
    #     print(col, sort(raw_df[col].unique()))
    
    # Based on the printed result, there is no clear outlier which should be removed.