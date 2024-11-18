import skrub 
from pathlib import Path 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import skrub


from helper import rr, var_dict, prep


def find_mean_raw_vs_t_cols(df, cols_to_add=None):
    """
    Identify columns to remove based on specific naming conventions across all columns in a DataFrame.
    
    Parameters:
    - df: DataFrame with columns to be checked
    
    Returns:
    - List of column names to remove
    """
    cols2rem = set()
    column_names = list(df.columns)
    
    # Pre-filter columns based on suffixes to reduce comparisons
    x_columns = [col for col in column_names if col.endswith('x')]
    t_columns = [col for col in column_names if col.endswith('t')]
    raw_columns = [col for col in column_names if col.endswith('raw')]
    
    # Check x and t suffix criteria - only add 'x' if paired with the same base name 't'
    for x_col in x_columns:
        base_name_x = x_col[:-2]  # Remove '_x' suffix
        for t_col in t_columns:
            base_name_t = t_col[:-2]  # Remove '_t' suffix
            if base_name_x == base_name_t:
                cols2rem.add(x_col)
                break  # No need to check further once we find a matching 't'

    # Check raw criteria - add 'raw' columns and their base columns if base exists
    for raw_col in raw_columns:
        base_name = raw_col[:-4]
        if base_name in column_names:
            cols2rem.add(raw_col)
            #cols2rem.add(base_name)
    
    # Check snaxrsp criteria - add columns with 'snaxrsp' in the name
    if cols_to_add is not None: 
        cols2rem.update(cols_to_add)
    

    return list(cols2rem)


### add a way to select specifi columns where to conversion to NaN does not hapen (e.g. in snap , relationship has value 88 for professional, but elsewhere it is a missing value)
def pre_audit(df, missing_val_codes, cols_known_to_remove = None, thr_drop_missing = 50):
    df = prep.set_baseline_dtypes(df) # set dtypes 
    df_clean= df.copy()
    print(missing_val_codes)
    df_clean = df_clean.replace(missing_val_codes, np.nan) #replace all values that represent missing values with nan
    df_clean  = df_clean.dropna(axis=1, how="all") # drop empty columns 
    df_clean = df_clean.loc[:, df_clean.nunique(dropna=True) >1] # drop columns that contain only one constant value 

    cols2rem = find_mean_raw_vs_t_cols(df, cols_known_to_remove) #find mean and raw columns where t column exist
    print("Removing known and raw columns..  N =  :" , len(cols2rem))

    try:
        df_clean = df.drop(columns = cols2rem) # drop all unwanted columns
    except KeyError as e:
        print('ERRROR')
        print(e)
        
    missing_perc = df_clean.isna().mean(axis = 0) * 100 # find columns with too many missing values
    cols_drop_miss = df_clean.columns[np.where(missing_perc > thr_drop_missing)]
    df_clean = df.drop(columns = cols_drop_miss)
    print("Removing above threshold empty columns.. N =  :" , len(cols_drop_miss))
    
    print('Old shape: ', df.shape)
    print('New shape: ', df_clean.shape)
    verifiy_pre_audit(df_clean)
    print("\n")
    
    return df_clean

def verifiy_pre_audit(df_clean):
    x_columns = [col for col in df_clean.columns if col.endswith('x')]
    print('Remaining, ends with x : ', x_columns)
    t_columns = [col for col in df_clean.columns if col.endswith('t')]
    print('Remaining, ends with t : ',t_columns)
    raw_columns = [col for col in df_clean.columns if col.endswith('raw')]
    print('Remaining, ends with raw : ',raw_columns)



def get_missing_str(missing_val_list):
    missing_val_codes= missing_val_list.copy()
    for val in missing_val_list:
        if not isinstance(val, str):
            missing_val_codes.append(str(val))
    return missing_val_codes
    
    
def rem_top_assoss(df, user_input = False):
    report = skrub.TableReport(df)
    top_assoss = report._summary_without_plots['top_associations'] 
    print("The following high associations were found..")
    
    for assos in top_assoss:
        print(assos['left_column_name'], assos['right_column_name'], assos['cramer_v'])
    if user_input:
        cols = input("Which columns would ou like to remove? Please enter the column names separated by a comma. ")
        split = cols.split(',')
        cols_clean = [item.strip() for item in split ]

        try: 
            assert np.isin(cols_clean, df.columns).any()
        except AssertionError as e:
            print("\n")
            print("An error has occured..")

            for col in cols_clean:
                if col not in df.columns:
                    print(col , ' not in dataframe.')
            print("Returning unmodified dataframe.")
            return df
        print("Columns selected to remove:")
        print(cols_clean)



        return cols_clean

    return


            
        