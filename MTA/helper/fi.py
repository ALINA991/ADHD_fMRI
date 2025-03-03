import pandas as pd
import numpy as np 
import shapely




def get_fi_from_model(pipeline, df_X, y):
    model_type = pipeline.named_steps["regressor"].__class__.__name__
    if model_type == "RandomForestRegressor":
        pipeline.fit(df_X, y)
        
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

        rf_step = pipeline.named_steps['regressor']
        importances = rf_step.feature_importances_
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
            
    elif model_type == "XGBRegressor":
        xgb_regressor = pipeline.named_steps['regressor']
        
        booster = xgb_regressor.get_booster()
        gain_importances_dict = booster.get_score(importance_type='gain')
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()


        feature_map = {f"f{i}": feature_names[i] for i in range(len(feature_names))}

        records = []
        for fkey, gain_val in gain_importances_dict.items():
            if fkey in feature_map:  # ensure we have a matching index
                feature_name = feature_map[fkey]
                records.append((feature_name, gain_val))

        fi_df = pd.DataFrame(records, columns=["feature", "importance"]).sort_values(
            by="importance", ascending=False
        )
        
    return fi_df


def get_fi_from_shap(pipeline, df_X, y):
    """
    Fits the pipeline on (df_X, y), computes SHAP values using a TreeExplainer (if the regressor
    is tree-based) and returns a DataFrame with columns ['feature', 'importance'].
    The 'importance' here is the mean absolute SHAP value for each feature.
    
    For tree-based models (RandomForestRegressor, XGBRegressor) we use TreeExplainer;
    otherwise, a generic explainer can be used.
    """
    # Fit the pipeline on all data.
    pipeline.fit(df_X, y)
    
    # Get the transformed features and their names from the preprocessor.
    X_trans = pipeline.named_steps['preprocessor'].transform(df_X)
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Identify the model type.
    model_type = pipeline.named_steps["regressor"].__class__.__name__
    
    # For tree-based models, use TreeExplainer.
    if model_type in ["RandomForestRegressor", "XGBRegressor"]:
        # Create a SHAP explainer using the regressor.
        explainer = shap.TreeExplainer(pipeline.named_steps['regressor'])
        # Compute SHAP values on the preprocessed data.
        shap_values = explainer.shap_values(X_trans)
        # For regression, shap_values is an array of shape (n_samples, n_features)
        # We define feature importance as the average absolute SHAP value.
        importances = np.mean(np.abs(shap_values), axis=0)
    else:
        # For non-tree models, you might use a KernelExplainer (this can be slow).
        explainer = shap.Explainer(pipeline.predict, X_trans)
        shap_values = explainer(X_trans)
        importances = np.mean(np.abs(shap_values.values), axis=0)
    
    # Create a DataFrame of feature importances.
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return fi_df


def get_fi_from_perm(pipeline, df_X, y, n_repeats=5):
    """
    Fits the pipeline on (df_X, y), then computes permutation importance (scoring='r2').
    Returns a DataFrame with columns ['feature', 'importance'],
    sorted by 'importance' (descending).
    """
    # Fit the entire pipeline
    pipeline.fit(df_X, y)

    # Compute permutation importance on the same data
    result = permutation_importance(
        pipeline,
        df_X,
        y,
        n_repeats=n_repeats,
        scoring='r2',
        random_state=42
    )
    importances = result.importances_mean

    # Get feature names from the preprocessor step
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Build a DataFrame mirroring the structure of get_fi_from_model()
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return fi_df

def get_fi_cv(pipeline, df_X, y, importance_meth = "inetrnal",  n_splits = 5, n_repeats= None):

    kf = GroupKFold(n_splits=n_splits)

    weighted_sums = {}
    r2_scores = []
    sum_of_r2 = 0.0

    for train_idx, val_idx in kf.split(df_X, y, groups):
        X_train, X_val = df_X.iloc[train_idx], df_X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        fold_r2 = r2_score(y_val, y_pred)
        r2_scores.append(fold_r2)
        sum_of_r2 += fold_r2
        
        if importance_meth == "internal":
            fi_fold = get_fi_from_model(pipeline, X_train, y_train)
        elif importance_meth == "perm":
            fi_fold = get_fi_from_perm(pipeline, X_train, y_train, n_repeats= n_repeats)

        fold_dict = {
            row['feature']: row['importance']
            for _, row in fi_fold.iterrows()
        }
        
        for feat, imp_val in fold_dict.items():
            weighted_sums[feat] = weighted_sums.get(feat, 0.0) + fold_r2 * imp_val

    records = []
    for feat, weighted_sum in weighted_sums.items():
        avg_importance = weighted_sum / sum_of_r2  # Weighted by total r2
        records.append((feat, avg_importance))

    fi_cv_df = pd.DataFrame(records, columns=["feature", "importance"])
    fi_cv_df.sort_values("importance", ascending=False, inplace=True)
        
    return fi_cv_df