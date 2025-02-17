from pathlib import Path 
import ast 
import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from decimal import Decimal, ROUND_DOWN


# ----------------------------
# Custom Transformers
# ------------------------
def truncate(value, decimals):
    value = Decimal(value)
    return float(value.quantize(Decimal(f"1.{'0' * decimals}"), rounding=ROUND_DOWN))




def get_params_from_best_result(file_path_save): 


    df_result  = pd.read_csv(file_path_save)

    best_result = df_result.loc[df_result['R² Score'] == df_result['R² Score'].max(), :].drop(columns='Unnamed: 0')


    params = ast.literal_eval(best_result['Hyperparameters'].iloc[0])
    model_type = best_result['Model Name'].iloc[0]
    col_out = best_result['Outcome Variable'].iloc[0]
    rater_pred = best_result['Input Data'].iloc[0].lower()
    corr_select=  best_result['Feature Selection Method'].iloc[0]

    if corr_select.startswith('correlation_selector'):
        name_part, dict_part = corr_select.split(' ', 1)
        thr_corr = ast.literal_eval(dict_part)['threshold']
    else: 
        thr_corr = None
        
    if rater_pred == 'all':
        rater_pred = None

    thr_drop_missing = best_result['Threshold Drop Row'].iloc[0]
    original_r2 = best_result['R² Score']

    col_out_reduced= col_out[:-4]
    print("original r2 result : ", original_r2.iloc[0])
    print('model type', model_type)
    print('corr_select :', corr_select)
    print("thr_corr: ", thr_corr )
    print('outcome : ', col_out_reduced)
    print('rater input data : ', rater_pred)
    print("thr drop missing: ", thr_drop_missing)
    print("params model : ", params)


    # thr_corr = params['subsample']
    # params.pop('subsample', None)
    return  model_type,  corr_select,   thr_corr, params,  col_out_reduced, rater_pred, thr_drop_missing, original_r2.iloc[0]


def get_data_types_from_file(data, types_file_path, sheet_name ):
    col_names_data = list(data.columns)

    ord_vars, num_vars, cat_vars = [], [], []

    types_df = pd.read_excel(types_file_path, sheet_name=sheet_name)

    for _, row in types_df.iterrows():
        var_name = row.iloc[1]  # e.g. variable name in the spreadsheet
        var_type = row.iloc[4]  # e.g. "ord" / "num" / "cat"
        var_in_data = [col for col in col_names_data if var_name+"_" in col] # add underscore to ensure exact match


        if var_type == "ord":
            ord_vars.append(var_in_data)
        elif var_type == "num":
            num_vars.append(var_in_data)
        elif var_type == "cat":
            cat_vars.append(var_in_data)
   
    # Example: manually add a column named 'trtname' to cat_vars
    cat_vars.append(['trtname'])
 

    # Flatten each list-of-lists into a single array
    ord_vars= np.concatenate(list(ord_vars))
    cat_vars = np.concatenate(cat_vars)
    num_vars = np.concatenate(num_vars)

    # Convert them to plain Python strings
    ord_vars = [str(col) for col in ord_vars]
    cat_vars = [str(col) for col in cat_vars]
    num_vars = [str(col) for col in num_vars]

    print("Ordinal vars:", ord_vars)
    print("Numeric vars:", num_vars)
    print("Categorical vars:", cat_vars)

    num_vars_in = [str(col) for col in num_vars if not col.endswith("out")] # name of numerical variables present in dataframe X (excludin var names in y)
    
    return ord_vars, cat_vars, num_vars, num_vars_in

def check_overlap(num_vars_in, ord_vars, cat_vars):
    num_set = set(num_vars_in)
    ord_set = set(ord_vars)
    cat_set = set(cat_vars)

    # Pairwise overlaps:
    overlap_num_ord = num_set & ord_set
    overlap_num_cat = num_set & cat_set
    overlap_ord_cat = ord_set & cat_set

    print("Overlaps between numeric and ordinal:", overlap_num_ord)
    print("Overlaps between numeric and categorical:", overlap_num_cat)
    print("Overlaps between ordinal and categorical:", overlap_ord_cat)

    # Overlap across all three:
    overlap_all_three = num_set & ord_set & cat_set
    print("Overlaps across numeric, ordinal and categorical:", overlap_all_three)