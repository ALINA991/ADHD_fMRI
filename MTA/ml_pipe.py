
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
from helper import  audit, ml
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

for outcome_short, rater_out, rater_pred, model_type, corr_select, thr_drop_missing in product(config["outcome"], config["rater_out"], config["rater_pred"], config["model_type"], config["corr_select"], config["thr_drop_row"]):
    
    print(f"Running script with outcome: {outcome_short},\n"
        f"rater out: {rater_out},\n"
        f"rater pred: {rater_pred},\n"
        f"model_type: {model_type},\n"
        f"corr_select: {corr_select},\n"
        f"threshold drop row: {thr_drop_missing}")
    
    thr_drop_missing = int(thr_drop_missing)
    rater_pred = None if rater_pred == "None" else rater_pred
    

    outcome_dict = {'ODD':  "snap_snaoddt", "HYP": "snap_snahypat", "INATT" :"snap_snainatt" , "INTERN": "ssrs_sspintt", "SS": "ssrs_ssptosst", "DOM": "pcrc_pcrcpax", "INTIM": "pcrc_pcrcprx"}

    file_name_save = 'results_ML_simple_CV_RF_XGB_{}.csv'.format(outcome_short)
    file_name_save_dist = 'dist_results_ML_simple_CV_RF_XGB_{}.csv'.format(outcome_short)
    
    if Path('/Volumes/alina').exists():
        print("Writing on external drive.. ")
        data_root =     '/Volumes/alina/MIT/code/data'
        data_derived  = '/Volumes/alina/MIT/code/data/output/derived_data'
    else: 
        data_root = '/Users/alina/Desktop/MIT/code/data'
        data_derived  = '/Users/alina/Desktop/MIT/code/data/output/derived_data'

    info_path = Path(data_root, "files") # description  of vars as written out questions 
    save_path = Path(data_derived, 'ML_results')
    types_file_path = Path(data_derived,"all_vars_description_ML.xlsx" ) # description  of variables aas ordinal, numeric or categorical 
    file_path_save_mean = Path(save_path, file_name_save) # path to save and load ML resluts tabke 
    file_path_save_dist= Path(save_path, file_name_save_dist) 
    
    result_exists = ml.find_result_in_file(file_path_save_mean, model_type, corr_select, thr_drop_missing, rater_pred, rater_out)

    if result_exists: # if result exists in file, skip loop 
        continue
    ################## DATA ####################
    pred = pd.read_csv(Path(data_derived, 'mta_data_clean.csv')).drop(columns = 'Unnamed: 0') # baseline 0 months 
    out = pd.read_csv(Path(data_derived, 'out_clean_all_raters.csv')).drop(columns = 'Unnamed: 0') # 14 months 


    col_out = outcome_dict[file_name_save.split(".")[0].split("_")[-1]] + "_"+ rater_out
    outcome_var = outcome_dict[outcome_short]


    # data : X +y, df X and y are formatted to be fed into pipeline, rater_count gives count of input data from each rater 
    data, df_X, y, rater_count_X = ml.prepare_data(pred, out, rater_pred, rater_out, thr_drop_missing, outcome_var)
    
    if y is None : # if the y col gets removed during audit, skip the loop with this outcome 
        continue
    
    dup_cols = ml.check_duplicates(df_X) # print duplicates if any 
    assert dup_cols == [] 

    # get the variable types to feed in each sub_pipeline 
    # to to variable type specific preprocessing 
    ord_vars, num_vars, cat_vars_str, cat_vars_num = ml.get_var_types(df_X, types_file_path) # get types of each variable 


    scoring = {
        'r2': 'r2',
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }

    groups = data['src_subject_id'].values

    pipeline, param_distributions, = make_pipeline(model_type= model_type, corr_select= corr_select, 
                                                        ord_vars= ord_vars,num_vars=num_vars,  cat_vars_str= cat_vars_str, 
                                                        cat_vars_num = cat_vars_num, params= params, include_rand_feature=False)



    # Define the GroupKFold cross-validation
    cv = GroupKFold(n_splits=5)

    # Define the scorer
    scorer = {
        'rmse': make_scorer(root_mean_squared_error, greater_is_better=False),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score)
    }

    # Set up the RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,  # Number of parameter settings to sample
        scoring=scorer,
        refit='r2', 
        cv=cv,
        verbose=3,
        random_state=42
    )

    # Perform the search
    random_search.fit(df_X, y, groups=groups)

    # Print best parameters and score
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

    # generate a new standardized row to appand to an existing dataframe, or create new one 
    new_row_mean = ml.get_results_from_random_search(random_search, outcome_short,rater_out,rater_pred,  thr_drop_missing)
    new_row_dist = ml.get_results_from_random_search(random_search, outcome_short,rater_out,rater_pred,  thr_drop_missing, get_dist=True)
    # if verify_before save is set to True, save and reduced parameters will be ignored 
    # and asked again from user input after displaying the tables 
    ml.save_cv_result_to_table(file_path_save_mean, new_row_mean, save=True, reduced= False, nrows2drop=None, verify_before_save=False)
    ml.save_cv_result_to_table(file_path_save_dist, new_row_dist, save=True, reduced= False, nrows2drop=None, verify_before_save=False)