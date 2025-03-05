import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from pathlib import Path 
import sys 
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer ,root_mean_squared_error

from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from skrub import TableVectorizer

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 

from sklearn.metrics import r2_score,  mean_absolute_error

sys.path.append('/Users/alina/Desktop/MIT/code/ADHD/MTA/helper')
from helper import  audit, ml, fi
from helper.transformers import CorrelationSelector,PreserveFeatureNames, PreserveFeatureNamesRegressor
from helper.pipeline import make_pipeline

mpl.rcParams['text.usetex'] = False

from sklearn import set_config # set6 output of each ste of the pipeline to be dataframes
set_config(transform_output="pandas") # to track feature names 

import json
from itertools import product
import argparse

parser = argparse.ArgumentParser(description="Run script with combinations of inputs ")
parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
args = parser.parse_args()

use_params_from_result = False
n_remove_top_features = None
compute_perm_importances = False
save_fig = False
save_importance_df = False

params = None # set if compute feature imprtance from existing result
# Load the configuration from the file
with open(args.config, 'r') as f:
    config = json.load(f)

################ add check if row in table exists already, skip loop
# right now check happens after loop 


if Path('/Volumes/Samsung_T5/MIT/mta').exists():
    data_root =     '/Volumes/Samsung_T5/MIT/mta'
    data_derived  = '/Volumes/Samsung_T5/MIT/mta/output/derived_data'
else: 
    data_root = '/Users/alina/Desktop/MIT/code/data'
    data_derived  = '/Users/alina/Desktop/MIT/code/data/output/derived_data'

info_path = Path(data_root, "files") # dewcipion of vars as written out questions 
save_path = Path(data_derived, 'ML_results')
types_file_path = Path(data_derived,"all_vars_description_ML.xlsx" ) # deescription of variables aas ordinal, numeric or categorical 

pred = pd.read_csv(Path(data_derived, 'mta_data_clean.csv')).drop(columns = 'Unnamed: 0')
out = pd.read_csv(Path(data_derived, 'out_clean_all_raters.csv')).drop(columns = 'Unnamed: 0')

outcome_dict = {'ODD':  "snap_snaoddt", "HYP": "snap_snahypat", "INATT" :"snap_snainatt" , "INTERN": "ssrs_sspintt", "SS": "ssrs_ssptosst", "DOM": "pcrc_pcrcpax", "INTIM": "pcrc_pcrcprx"}


for outcome_short in config["outcome"]: # iterate over possible outcomes 
    
    print(f"Running script with outcome: {outcome_short}")

    file_name_save = 'results_ML_simple_CV_RF_XGB_{}.csv'.format(outcome_short)
    file_name_save_dist = 'dist_results_ML_simple_CV_RF_XGB_{}.csv'.format(outcome_short)

    file_path_save = Path(save_path, file_name_save) # path to save and load ML resluts tabke 
    result_file_shape = pd.read_csv(types_file_path).shape[0]

    results_how = "index"

    for index in range(result_file_shape): # iterate iver each results, mostly getting right hyperparams :
        model_type,  corr_select, thr_corr, params,  outcome_var, rater_out, rater_pred, thr_drop_missing, original_r2 = ml.get_params_from_result(file_path_save, results_how, index)


        col_out = outcome_dict[file_name_save.split(".")[0].split("_")[-1]] + "_"+ rater_out
        outcome_var = outcome_dict[outcome_short]


        # data : X +y, df X and y are formatted to be fed into pipeline, rater_count gives count of input data from each rater 
        data, df_X, y, rater_count_X = ml.prepare_data(pred, out, rater_pred, rater_out, thr_drop_missing, outcome_var)
        
        if y is None : # if the y col gets removed during audit, skip the loop with this outcome 
            continue
        
        dup_cols = ml.check_duplicates(df_X) # print duplicates if any 
        assert dup_cols == [] 

        ord_vars, num_vars, cat_vars_str, cat_vars_num = ml.get_var_types(df_X, types_file_path) # get types of each variable 


        scoring = {
            'r2': 'r2',
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error'
        }

        groups = data['src_subject_id'].values

        pipeline, param_distributions, = make_pipeline(model_type= model_type, corr_select= corr_select, 
                                                            ord_vars= ord_vars,num_vars=num_vars,  cat_vars_str= cat_vars_str, 
                                                            cat_vars_num = cat_vars_num, params= params)



        # Define the GroupKFold cross-validation
        cv = GroupKFold(n_splits=5)

        # Define the scorer
        scorer = {
            'rmse': make_scorer(root_mean_squared_error, greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'r2': make_scorer(r2_score)
        }
        
        fi_df_internal, fold_importance_storage_internal = fi.compute_feature_importances(pipeline, df_X, y, groups, imp_method="internal")
        fi_df_perm, fold_importance_storage_perm = fi.compute_feature_importances(pipeline, df_X, y, groups, imp_method="perm", n_repeats=10)
