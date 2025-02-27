
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

mpl.rcParams['text.usetex'] = False

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

    if Path('/Volumes/Samsung_T5/MIT/mta').exists():
        data_root =     '/Volumes/Samsung_T5/MIT/mta'
        data_derived  = '/Volumes/Samsung_T5/MIT/mta/output/derived_data'
    else: 
        data_root = '/Users/alina/Desktop/MIT/code/data'
        data_derived  = '/Users/alina/Desktop/MIT/code/data/output/derived_data'

    info_path = Path(data_root, "files") # dewcipion of vars as written out questions 
    save_path = Path(data_derived, 'ML_results')
    types_file_path = Path(data_derived,"all_vars_description_ML.xlsx" ) # deescription of variables aas ordinal, numeric or categorical 
    file_path_save = Path(save_path, file_name_save) # path to save and load ML resluts tabke 


    ################## DATA ####################
    pred = pd.read_csv(Path(data_derived, 'mta_data_clean.csv')).drop(columns = 'Unnamed: 0')
    out = pd.read_csv(Path(data_derived, 'out_clean_all_raters.csv')).drop(columns = 'Unnamed: 0')



    # if use_params_from_result : # use parameters from the best model saved in ML results table ß
    #     model_type,  corr_select,   thr_corr, params,  col_out, rater_pred, thr_drop_missing ,r2 = ml.get_params_from_best_result(file_path_save)
    #     rater_out = col_out[-1]
    # else: ################# get INPUT FROM USER 
    #     model_type ="RandomForestRegressor" # "XGBRegressor"
    #     corr_select = True
    #     thr_corr = 0.8
    #     params = None
    #     rater_pred = None 
    #     #rater_out = "m"
    #     thr_drop_missing = 50
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


    if model_type == "XGBRegressor":
        regress =  XGBRegressor(random_state=42)
    elif model_type ==  "RandomForestRegressor":
        regress = RandomForestRegressor(random_state= 42)


    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), # check paper 
        ('std_scaler',StandardScaler())
    ])

    ord_pipe = Pipeline([
        ('imputer', PreserveFeatureNames(SimpleImputer(strategy='mean'))), # change that to OrdinalEncoder sklearn -- just encodes to catgoreies !!
        ('identity', PreserveFeatureNames(FunctionTransformer(lambda x: x)))
    ])


    cat_str_pipe = Pipeline([
        ('imputer', PreserveFeatureNames(SimpleImputer(strategy='constant', fill_value='missing'))),
        ('ohe', PreserveFeatureNames(TableVectorizer()))
    ])

    cat_num_pipe = Pipeline([
        ('imputer', PreserveFeatureNames(SimpleImputer(strategy='constant', fill_value=-1))),
        ('ohe', PreserveFeatureNames(OneHotEncoder(handle_unknown="ignore")))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_vars),
        ('cat_str', cat_str_pipe, cat_vars_str),
        ('cat_num', cat_num_pipe, cat_vars_num),
        ('ord', ord_pipe, ord_vars)
    ])

    if corr_select : 
        pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('correlation_selector', CorrelationSelector(threshold=0.8)),
        ('regressor',regress)
        ])
    else: 
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor',regress)
        ])

    if model_type == "RandomForestRegressor" :
        param_distributions = {
            'regressor__n_estimators': [100, 300],          # Correct prefix
            'regressor__max_depth': [10, None],
            'regressor__min_samples_split': [2, 10],
            'regressor__min_samples_leaf': [1, 4]
        }
        
    elif model_type == "XGBRegressor":
        param_distributions = {
            'regressor__n_estimators': [100, 300],
            'regressor__max_depth': [10],                    # Removed None
            'regressor__colsample_bytree': [0.5, 0.3],       # Mapped 'sqrt' and 'log2' to numeric values
            'regressor__min_child_weight': [2, 10, 1, 4],    # Combined 'min_samples_split' and 'min_samples_leaf'
        }
        
    # Define the parameter grid

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
        refit='rmse', 
        cv=cv,
        verbose=3,
        random_state=42
    )

    # Perform the search
    random_search.fit(df_X, y, groups=groups)

    # Print best parameters and score
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)


    new_row = ml.get_results_from_random_search(random_search, outcome_short,rater_out,rater_pred,  thr_drop_missing)

    # if verify_before save is set to True, save and reduced parameters will be ignored 
    # and asked again from user input after displaying the tables 
    ml.save_cv_result_to_table(file_path_save, new_row, save=True, reduced= False, nrows2drop=None, verify_before_save=False)