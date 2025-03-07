from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from scipy.stats import randint, uniform
import numpy as np

from helper.transformers import DataFrameToNumpy, CorrelationSelector, LeastDistanceCorrelatedRandomFeature, TableVectorizerWrapper

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import set_config

from skrub import TableVectorizer
set_config(transform_output="pandas")


def make_pipeline(model_type, corr_select, ord_vars, num_vars, cat_vars_str, cat_vars_num, params = None, include_rand_feature=False):

    
    if params is not None:

        if model_type == "XGBRegressor":
            regress =  XGBRegressor(random_state=42, **params)
        elif model_type == "RandomForestRegressor":
            regress = RandomForestRegressor(random_state= 42, **params )
    else: 
        if model_type == "XGBRegressor":
            regress =  XGBRegressor(random_state=42)
            param_distributions = {
            'regressor__n_estimators': randint(100, 600),
            'regressor__max_depth': randint(3, 11),
            'regressor__learning_rate': uniform(0.01, 0.19),
            'regressor__subsample': uniform(0.6, 0.4),
            'regressor__colsample_bytree': uniform(0.5, 0.5),
            'regressor__min_child_weight': randint(1, 11)
        } 

        elif model_type ==  "RandomForestRegressor":
            regress = RandomForestRegressor(random_state= 42)
            param_distributions = {
            'regressor__n_estimators': randint(100, 600),
            'regressor__max_depth': list(range(5, 21, 5)),# [None] 
            'regressor__min_samples_split': randint(2, 12),
            'regressor__min_samples_leaf': randint(1, 7)
        }
        if corr_select:
            param_distributions.update({'correlation_selector__threshold':  [0.7, 0.75, 0.8, 0.85, 0.9]})
            


    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), # check paper # check more sophisticated imputation strategies 
        ('std_scaler',StandardScaler())
    ])

    ord_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)), 
        ('identity', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])


    cat_str_pipe = Pipeline([
        ('imputer',SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', TableVectorizerWrapper()) # wrapper here to extract feature names, TableVectorizer() does not implement get_feature_names_out()
    ])

    cat_num_pipe = Pipeline([
        ('imputer',SimpleImputer(strategy='constant', fill_value=-1)),
        ('ohe', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_vars),
        ('cat_str', cat_str_pipe, cat_vars_str),
        ('cat_num', cat_num_pipe, cat_vars_num),
        ('ord', ord_pipe,ord_vars),

    ])
    
    steps = [('preprocessor', preprocessor)]

    if corr_select: # removes highly correlated features, in a priority order defined by feature importance returned by this model
        steps.append(('correlation_selector', CorrelationSelector(model=XGBRegressor(n_estimators=50, random_state=42))))
    
    if include_rand_feature: # add random feature, with least distance correlation from target vector y 
        steps.append(('rand_feature', LeastDistanceCorrelatedRandomFeature(n_candidates=10, random_state=42, feature_name="random_vector")))
    
    steps.append(('to_numpy', DataFrameToNumpy()))
    steps.append(('regressor', regress))
    
    pipeline = Pipeline(steps=steps)

    """ pipeline including all steps would look like this : 
        Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('correlation_selector', CorrelationSelector(model=XGBRegressor(n_estimators=50, random_state=42))),
                    ('rand_feature', LeastDistanceCorrelatedRandomFeature(n_candidates=10, random_state=42, feature_name="random_vector")),
                    ('to_numpy', DataFrameToNumpy()),
                    ('regressor', regress)])
    """

    if params is not None: # params have been set in regressor object
        return pipeline 

    else: 
        return pipeline, param_distributions # params will be evaluated in Random search later on 
        
        
