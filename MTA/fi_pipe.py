import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from pathlib import Path 
import sys 
import matplotlib as mpl

import pickle
import json
import argparse

import os 
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/Users/alina/Desktop/MIT/code/ADHD/MTA/helper')
from helper import  ml, fi
from helper.pipeline import make_pipeline

from sklearn.metrics import make_scorer ,root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score,  mean_absolute_error

from sklearn import set_config # set outputs of each step of the pipeline to be dataframes to track feature names  easliy
set_config(transform_output="pandas") 

mpl.rcParams['text.usetex'] = False

parser = argparse.ArgumentParser(description="Run script with combinations of inputs ")
parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
args = parser.parse_args()

# Load the configuration from the file
with open(args.config, 'r') as f:
    config = json.load(f)

################ add check if row in table exists already, skip loop
# right now check happens after loop 

save = True 

if Path('/Volumes/alina').exists():
    print("Writing on external drive.. ")
    data_root =     '/Volumes/alina/MIT/code/data'
    data_derived  = '/Volumes/alina/MIT/code/data/output/derived_data'
else: 
    data_root = '/Users/alina/Desktop/MIT/code/data'
    data_derived  = '/Users/alina/Desktop/MIT/code/data/output/derived_data'

save_path = Path(data_derived, 'ML_results')
print(os.listdir(save_path))
types_file_path = Path(data_derived,"all_vars_description_ML.xlsx" ) # get ordinal, numerical or categorical types from file 

pred = pd.read_csv(Path(data_derived, 'mta_data_clean.csv')).drop(columns = 'Unnamed: 0') # read baseline data 
out = pd.read_csv(Path(data_derived, 'out_clean_all_raters.csv')).drop(columns = 'Unnamed: 0') # read outcome dat a

outcome_dict = {'ODD':  "snap_snaoddt", "HYP": "snap_snahypat", "INATT" :"snap_snainatt" , "INTERN": "ssrs_sspintt", "SS": "ssrs_ssptosst", "DOM": "pcrc_pcrcpax", "INTIM": "pcrc_pcrcprx"}
results_how = "index" # how to extract results from file. Other option is "best" to extract best result in file 


for outcome_short in config["outcome"]: # iterate over possible outcomes :  ["ODD", "HYP","INATT", "INTERN", "SS", "DOM", "INTIM"]
    
    print(f"Running script with outcome: {outcome_short}")
    result_file_name = 'results_ML_simple_CV_RF_XGB_{}.csv'.format(outcome_short) # read params from here
    result_path= Path(save_path, result_file_name) 
    result_file_shape = pd.read_csv(result_path).shape[0]


    # iterate over each row in results, to get combination of model, 
    # input data, feature selection methods and hyperparams..
    for index in range(result_file_shape): 
        model_type,  corr_select, thr_corr, params,  outcome_var, rater_out, rater_pred, thr_drop_missing, original_r2 = ml.get_params_from_result(result_path, results_how, index)

        fi_name_save = fi.get_importance_file_name(model_type, outcome_short, corr_select, thr_corr, params,  outcome_var, rater_out, rater_pred, thr_drop_missing)
        fi_save_path = Path(data_derived, "ML_results","feature_importances", fi_name_save)
        
        if fi_save_path.exists(): # if the file already exists, skip computation and go to next interation
            print("\nResult {}exists...".format(fi_name_save))
            print("Skipping.. \n")
            continue
        

        data, df_X, y, rater_count_X = ml.prepare_data(pred, out, rater_pred, rater_out, thr_drop_missing, outcome_var)
        
        if y is None : # if the y col gets removed during praparation (e.g. too many values missing ), skip 
            print_rater = "all raters" if rater_pred is None else rater_pred
            print("Outcome {} from input {} has been removed during audit..".format(outcome_short,print_rater))
            print("Skipping ...")
            continue
        
        dup_cols = ml.check_duplicates(df_X) # print duplicates if any 
        assert dup_cols == [] 

        ord_vars, num_vars, cat_vars_str, cat_vars_num = ml.get_var_types(df_X, types_file_path) # get types of each variable 

        scoring = {
            'r2': 'r2',
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error'
        }

        groups = data['src_subject_id'].values # groups used for groupKFold to ensure each subject it present in exaclty one fold

        pipeline = make_pipeline(model_type= model_type, corr_select= corr_select, 
                                        ord_vars= ord_vars,num_vars=num_vars,  cat_vars_str= cat_vars_str, 
                                        cat_vars_num = cat_vars_num, params= params, include_rand_feature=True)

        cv = GroupKFold(n_splits=5)
        precomputed_splits = list(cv.split(df_X, y, groups)) # precompute splits to ensure consistency across fi methods 


        scorer = {
            'rmse': make_scorer(root_mean_squared_error, greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'r2': make_scorer(r2_score) # main metric for performance 
        }
        
        # compute feature importance : internal to model, permutation importance, and shap
        # all these computations are done in a GroupKFold manner 
        fi_df_model, fold_importance_storage_model, r2_scores_model, train_test_splits_model, raw_predictions_model= fi.compute_feature_importances(pipeline, df_X, y, groups, imp_method="internal", precomputed_splits= precomputed_splits)
        fi_df_perm, fold_importance_storage_perm, r2_scores_perm, train_test_splits_perm, raw_predictions_perm, full_permutation_importances_perm= fi.compute_feature_importances(pipeline, df_X, y, groups, imp_method="perm", n_repeats=2, precomputed_splits=precomputed_splits)
        fi_df_shap, fold_importance_storage_shap, r2_scores_shap, train_test_splits_shap, raw_predictions_shap, shap_values_dict = fi.compute_feature_importances(pipeline, df_X, y, groups, imp_method="shap", precomputed_splits=precomputed_splits)

        # get unique file name for each combination of model/inputs/feature selection methods


        # Define a dictionary to store all results
        feature_importance_results = {
            "internal": {
                "fi_df": fi_df_model,
                "fold_importance": fold_importance_storage_model,
                "r2_scores": r2_scores_model,
                "train_test_splits": train_test_splits_model,
                "raw_predictions": raw_predictions_model
            },
            "perm": {
                "fi_df": fi_df_perm,
                "fold_importance": fold_importance_storage_perm,
                "r2_scores": r2_scores_perm,
                "train_test_splits": train_test_splits_perm,
                "raw_predictions": raw_predictions_perm,
                "full_permutation_importances": full_permutation_importances_perm  # Only for permutation importance
            },
            "shap": {
                "fi_df": fi_df_shap,
                "fold_importance": fold_importance_storage_shap,
                "r2_scores": r2_scores_shap,
                "train_test_splits": train_test_splits_shap,
                "raw_predictions": raw_predictions_shap,
                "shap_values_dict": shap_values_dict  # Only for SHAP importance
            }
        }

        # Save the dictionary to file
        if save:
            print("Saving to file {} ...".format(fi_save_path))
            with open( fi_save_path, "wb") as f:
                pickle.dump(feature_importance_results, f)
            print("Success\n")
            
            