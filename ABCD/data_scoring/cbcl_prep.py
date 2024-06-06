import pandas as pd

def exclude(df, miss_val_colum):
    return df[df[miss_val_colum].notna()]
    