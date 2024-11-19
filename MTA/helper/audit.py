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

def set_dtypes_and_nan(df, df_info, missing_val_codes, dtype_dict):
    # these columns are not present in the info files 
    subject_unspecific_cols = [col for col in df.columns if col not in df_info['ElementName'].tolist()]
    subject_unspecific_cols.append('interview_date')
    print("Removing subject_unspecific columns ..  N = ", len(subject_unspecific_cols))
    df_clean = df.drop(columns = subject_unspecific_cols)
    df_clean  = df_clean.replace(np.nan, -999)
    print("Setting dtypes..")
    df_clean = df_clean.astype({
    col: dtype_dict[df_info.loc[ df_info['ElementName'] == col,'DataType' ].values[0]] for col in df_clean.columns 


 #### all column get converted to float not int even if in info_dict 
})
    print(df_clean.dtypes)
    df_clean= df_clean.replace(missing_val_codes, np.nan).copy()
    return df_clean
    


### add a way to select specifi columns where to conversion to NaN does not hapen (e.g. in snap , relationship has value 88 for professional, but elsewhere it is a missing value)
def pre_audit(df, df_info, missing_val_codes, dtype_dict,  cols_known_to_remove = None, thr_drop_missing = 50):
    df_clean = prep.set_baseline_dtypes(df) # set dtypes fro baseline vars
    df_clean= df_clean.replace(missing_val_codes, np.nan).copy()
    df_clean = set_dtypes_and_nan(df_clean, df_info, missing_val_codes, dtype_dict)    # set dtypes for all, replcace missing vas with NaN
    print(df_clean.shape)
    
    
    empty_cols = df.isna().all(axis = 1)
    print("Removing empty columns ..  N = ", empty_cols.sum())
    df_clean  = df_clean.drop(columns = empty_cols[empty_cols]) # drop empty columns 
    print(df_clean.shape)
    
    
    constant_cols = df_clean.nunique(dropna=True) <= 1
    print("Removing constant columns .. N = ", constant_cols.sum())
    df_clean = df_clean.drop(columns=constant_cols[constant_cols].index)
    print(df_clean.shape)
    
    cols2rem = find_mean_raw_vs_t_cols(df_clean, cols_known_to_remove) #find mean and raw columns where t column exist
    print("Removing known and raw columns..  N =  :" , len(cols2rem))
    print("COLS @ REM : ", cols2rem)


    try:
        df_clean = df_clean.drop(columns = cols2rem) # drop all unwanted columns
        print(df_clean.shape)
    except KeyError as e:
        print('ERROR')
        print(e, "Removing non-existing columns from list .. ")
        cols2rem = [col for col in cols2rem if col in df_clean.columns]
        df_clean = df_clean.drop(columns = cols2rem) # drop all unwanted columns
        print("Success. Removing known and raw columns..  N =  :" , len(cols2rem))
        print(df_clean.shape)
        
    missing_perc = df_clean.isna().mean(axis = 0) * 100 # find columns with too many missing values
    cols_drop_miss = df_clean.columns[np.where(missing_perc > thr_drop_missing)]
    print("Removing above threshold empty columns.. N =  :" , len(cols_drop_miss))
    df_clean = df_clean.drop(columns = cols_drop_miss)
    print(df_clean.shape)

    
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


            
        