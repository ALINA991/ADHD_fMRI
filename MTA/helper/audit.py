import skrub 
from pathlib import Path 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import skrub
import matplotlib.ticker as ticker

from helper import rr, var_dict, prep


def find_mean_raw_vs_t_cols(df, cols_to_add=None, cols_known_to_keep = []):
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
    cols2rem.difference_update(cols_known_to_keep)
    

    return list(cols2rem)

def set_dtypes_and_nan(df, df_info, missing_val_codes, dtype_dict, cols_known_to_keep=[]):
    # Check for description line and drop it
    if df['src_subject_id'].iloc[0] == 'Subject ID how it\'s defined in lab/project':
        df = df.drop(0, axis=0)

    # Identify subject-unspecific columns
    subject_unspecific_cols = [col for col in df.columns if col not in df_info['ElementName'].tolist()]

    if 'interview_date' in df.columns:
        subject_unspecific_cols.append('interview_date')

    # Ensure known-to-keep columns are not removed
    if cols_known_to_keep:
        subject_unspecific_cols = [col for col in subject_unspecific_cols if col not in cols_known_to_keep]

    print("Removing subject-unspecific columns ..  N = ", len(subject_unspecific_cols))
    print(subject_unspecific_cols)
    df_clean = df.drop(columns=subject_unspecific_cols, errors='ignore')  # `errors='ignore'` ensures no error if columns are missing
    df_clean = df_clean.replace(np.nan, -999)

    print(df_clean.shape)

    # Set data types
    print("Setting dtypes...")
    for col in df_clean.columns:
        if col in dtype_dict and col in df_info['ElementName'].tolist():
            try:
                dtype = dtype_dict[df_info.loc[df_info['ElementName'] == col, 'DataType'].values[0]]
                df_clean[col] = df_clean[col].astype(dtype)
            except (IndexError, ValueError, TypeError) as e:
                print(f"Error converting column {col}: {e}")
        elif col in cols_known_to_keep:
            print(f"Skipping type conversion for column {col} as it's not in info dictionary.")

    # Replace missing value codes back to NaN
    df_clean = df_clean.replace([-999, "-999"], np.nan).copy()

    return df_clean
    



def remove_nan_cols(df, cols_known_to_keep ):
    empty_cols = df.columns[ (df.isna().all(axis = 0)) & (~df.columns.isin(cols_known_to_keep)) ]
    print(empty_cols)
    print("Removing empty columns ..  N = ", len(empty_cols))
    print(empty_cols)
    df = df.drop(columns = empty_cols) # drop empty columns 
    print(df.shape)
    return df 

def remove_const_cols(df_clean, cols_known_to_keep):
    constant_cols = df_clean.columns[ (df_clean.nunique(dropna=True) <= 1) & (~df_clean.columns.isin(cols_known_to_keep))]
    print("Removing constant columns .. N = ", len(constant_cols))
    print(constant_cols)

    df_clean = df_clean.drop(columns= constant_cols)
    print(df_clean.shape)
    return df_clean
    
def remove_raw_and_know_cols(df_clean, cols_known_to_remove, cols_known_to_keep):
    cols2rem = find_mean_raw_vs_t_cols(df_clean, cols_known_to_remove, cols_known_to_keep) #find mean and raw columns where t column exist
    print("Removing known and raw columns..  N =  :" , len(cols2rem))
    print(cols2rem)
    

    try:
        df_clean = df_clean.drop(columns = cols2rem) # drop all unwanted columns
        print(df_clean.shape)
    except KeyError as e:
        print('    ERROR')
        print(e)
        print("Removing non-existing columns from list .. ")
        cols2rem = [col for col in cols2rem if col in df_clean.columns]
        df_clean = df_clean.drop(columns = cols2rem) # drop all unwanted columns
        print("Success. Removing known and raw columns..  N =  :" , len(cols2rem))
        print(cols2rem)
        print(df_clean.shape)
        
    return df_clean
    
def remove_thr_empty_cols(df_clean, thr_drop_missing, cols_known_to_keep):
    
    missing_perc = df_clean.isna().mean(axis = 0) * 100 # find columns with too many missing values
    cols_drop_miss = set(df_clean.columns[np.where(missing_perc > thr_drop_missing)])
    cols_drop_miss -= set(cols_known_to_keep)
    print("Removing above threshold empty columns.. N =  :" , len(cols_drop_miss))
    print(cols_drop_miss)
    df_clean = df_clean.drop(columns = cols_drop_miss)
    print(df_clean.shape)
    return df_clean
    
    
    
######## add function for remving cold without settig dtyes 
### maybe option to chhoose which ones to remove 
### a also to print chich ones would be removed 

def remove_cols(df, cols_known_to_remove = [], cols_known_to_keep= [], thr_drop_missing = 50):
    df_clean = remove_nan_cols(df, cols_known_to_keep)
    df_clean = remove_const_cols(df_clean, cols_known_to_keep)
    df_clean = remove_raw_and_know_cols(df_clean, cols_known_to_remove, cols_known_to_keep)
    df_clean = remove_thr_empty_cols(df_clean, thr_drop_missing, cols_known_to_keep)
    return df_clean


    




### add a way to select specifi columns where to conversion to NaN does not hapen (e.g. in snap , relationship has value 88 for professional, but elsewhere it is a missing value)
def pre_audit(df, df_info, missing_val_codes, dtype_dict,  cols_known_to_remove = None, cols_known_to_keep = [], thr_drop_missing = 20):
    #df_clean = prep.set_baseline_dtypes(df) # set dtypes fro baseline vars
    print('original shape : ', df.shape)
    df_clean= df.replace(missing_val_codes, np.nan).copy()
    df_clean = set_dtypes_and_nan(df_clean, df_info, missing_val_codes, dtype_dict, cols_known_to_keep)    # set dtypes for all, replcace missing vas with NaN
    df_clean = remove_cols(df_clean, cols_known_to_remove, cols_known_to_keep, thr_drop_missing)

    # df_clean = remove_nan_cols(df_clean, cols_known_to_keep)
    # df_clean = remove_const_cols(df_clean, cols_known_to_keep)
    # df_clean = remove_raw_and_know_cols(df_clean, cols_known_to_remove, cols_known_to_keep)
    # df_clean = remove_thr_empty_cols(df_clean, thr_drop_missing, cols_known_to_keep)
    # constant_cols = df_clean.columns[ (df_clean.nunique(dropna=True) <= 1) & (~df_clean.columns.isin(cols_known_to_keep))]
    # print("Removing constant columns .. N = ", len(constant_cols))

    # df_clean = df_clean.drop(columns= constant_cols)
    

    
    # cols2rem = find_mean_raw_vs_t_cols(df_clean, cols_known_to_remove) #find mean and raw columns where t column exist
    # print("Removing known and raw columns..  N =  :" , len(cols2rem))
    


    # try:
    #     df_clean = df_clean.drop(columns = cols2rem) # drop all unwanted columns
    #     print(df_clean.shape)
    # except KeyError as e:
    #     print('    ERROR')
    #     print(e)
    #     print("Removing non-existing columns from list .. ")
    #     cols2rem = [col for col in cols2rem if col in df_clean.columns]
    #     df_clean = df_clean.drop(columns = cols2rem) # drop all unwanted columns
    #     print("Success. Removing known and raw columns..  N =  :" , len(cols2rem))
    #     print(df_clean.shape)
        

    # missing_perc = df_clean.isna().mean(axis = 0) * 100 # find columns with too many missing values
    # cols_drop_miss = df_clean.columns[np.where(missing_perc > thr_drop_missing)]
    # print("Removing above threshold empty columns.. N =  :" , len(cols_drop_miss))
    # df_clean = df_clean.drop(columns = cols_drop_miss)
    # print(df_clean.shape)

    
    print('Old shape: ', df.shape)
    print('New shape: ', df_clean.shape)
    
    verifiy_pre_audit(df_clean, missing_val_codes)
    print(df_clean.dtypes)
    print("\n")
    
    return df_clean


def audit():
    # pre audit 

        # verify activate status of patients 
    
    # map relationship 
    
    #get timepoint 
    
    # extract data per rater 
    
    # handle duplicate raters
        # drop 
        
        # first occurance 
        
        # average 
        
    # merge
    
        #raters
        
        # qsts 
    
    
    pass

def verifiy_pre_audit(df_clean, missing_val_codes):
    x_columns = [col for col in df_clean.columns if col.endswith('x')]
    print('Remaining, ends with x : ', x_columns)
    t_columns = [col for col in df_clean.columns if col.endswith('t')]
    print('Remaining, ends with t : ',t_columns)
    raw_columns = [col for col in df_clean.columns if col.endswith('raw')]
    print('Remaining, ends with raw : ',raw_columns)
    
    if df_clean.shape[1] <= 3:
        print('    WARNING! This dataframe is likely unusable.')
        print('    Columns still present after pre-audit: ', df_clean.columns)
        
    exists = (df_clean.isin(missing_val_codes)).any().any()
    if exists: 
        still_present = [value for value in missing_val_codes if (df_clean == value).any().any()]
        print('    WARNING ! The following missing values have not been converted to NaN', still_present)



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


            
        