from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from scipy.stats import randint, uniform

from helper.transformers import CorrelationSelector,PreserveFeatureNames, LeastDistanceCorrelatedRandomFeature

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from skrub import TableVectorizer

def make_pipeline(model_type, corr_select, ord_vars, num_vars, cat_vars_str, cat_vars_num, params = None):
    if params is not None:

        if model_type == "XGBRegressor":
            regress =  XGBRegressor(random_state=42, **params)
        elif model_type == "RandomForestRegressor":
            regress = RandomForestRegressor(random_state= 42, **params )
    else: 
        if model_type == "XGBRegressor":
            regress =  XGBRegressor(random_state=42)
        elif model_type ==  "RandomForestRegressor":
            regress = RandomForestRegressor(random_state= 42)



    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), # check paper 
        ('std_scaler',StandardScaler())
    ])

    ord_pipe = Pipeline([
        ('imputer', PreserveFeatureNames(SimpleImputer(strategy='constant', fill_value=-1))), # change that to OrdinalEncoder sklearn -- just encodes to catgoreies !!
        ('identity', PreserveFeatureNames(OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
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
        ('rand_feature',  LeastDistanceCorrelatedRandomFeature(n_candidates=10, random_state=42, feature_name = "random_vector")),
        ('regressor',regress)
        ])
    else: 
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('rand_feature', LeastDistanceCorrelatedRandomFeature(n_candidates=10, random_state=42, feature_name = "random_vector")),
            ('regressor',regress)
        ])
        
    if model_type == "RandomForestRegressor":
        param_distributions = {
            'regressor__n_estimators': randint(100, 600),
            'regressor__max_depth': [None] + list(range(5, 21, 5)),
            'regressor__min_samples_split': randint(2, 12),
            'regressor__min_samples_leaf': randint(1, 7)
        }
    elif model_type == "XGBRegressor":
        param_distributions = {
            'regressor__n_estimators': randint(100, 600),
            'regressor__max_depth': randint(3, 11),
            'regressor__learning_rate': uniform(0.01, 0.19),
            'regressor__subsample': uniform(0.6, 0.4),
            'regressor__colsample_bytree': uniform(0.5, 0.5),
            'regressor__min_child_weight': randint(1, 11)
        }
    
    if params is not None: 
        return pipeline 

    else: 
        return pipeline, param_distributions
        