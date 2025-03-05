import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

from sklearn import set_config

set_config(transform_output="pandas") # keep output of each step as dataframe to preserve column (feature) names 
  

def compute_feature_importances(pipeline, df_X, y, groups, imp_method="internal", n_repeats=10):
    """
    Compute feature importances using different methods: 'internal', 'perm', or 'shap'.
    
    Args:
        pipeline: The fitted machine learning pipeline.
        df_X: Feature dataframe.
        y: Target variable.
        groups: Grouping variable for cross-validation.
        imp_method: Method to compute feature importances ('internal', 'perm', or 'shap').
        n_repeats: Number of repeats for permutation importance (only used if imp_method='perm').

    Returns:
        final_df: DataFrame containing sorted feature importances.
        fold_importance_storage: Dictionary with per-fold feature importance DataFrames.
    """
    
    print(f"🚀 Computing feature importances using: {imp_method.upper()} method...")
    
    gkf = GroupKFold(n_splits=3)
    
    feature_importances_list = []
    weighted_importances_list = []
    r2_scores = []
    all_features = pd.Index([])  
    fold_importance_storage = {}  

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df_X, y, groups)):
        print(f"\n📊 Processing Fold {fold_idx + 1}/{gkf.get_n_splits()}")

        X_train, X_test = df_X.iloc[train_idx], df_X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        model = pipeline.named_steps['regressor']

        feature_names = pd.Index(pipeline[:-2].get_feature_names_out())  
        all_features = all_features.union(feature_names)  

        X_train_transformed = pipeline[:-1].transform(X_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

        print(f"📌 Fold {fold_idx} R² Score: {r2:.4f}")
        print(f"📌 Fold {fold_idx} Features: {len(feature_names)}")

        # Extract feature importance based on the selected method
        if imp_method == "internal":
            print("⚙️ Extracting feature importances from the model...")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "get_booster"):  
                importances = model.get_booster().get_score(importance_type="gain")
                importances = np.array([importances.get(f, 0) for f in feature_names])
            else:
                raise ValueError("Selected model does not support internal feature importance.")

        elif imp_method == "perm":
            print("🔄 Computing permutation importance...")
            X_test_transformed = pipeline[:-1].transform(X_test)
            result = permutation_importance(model, X_test_transformed, y_test, n_repeats=n_repeats, random_state=42, scoring="r2")
            importances = result.importances_mean  

        elif imp_method == "shap":
            print("🧩 Computing SHAP values...")
            explainer = shap.Explainer(model, X_train_transformed)
            shap_values = explainer(X_train_transformed)
            importances = np.abs(shap_values.values).mean(axis=0)

        else:
            raise ValueError("Invalid feature importance method. Choose from 'internal', 'perm', or 'shap'.")

        weighted_importance_ = importances * r2

        fold_importance_df = pd.DataFrame({
            'feature': feature_names,
            f'importance_fold_{fold_idx}': importances
        })

        fold_importance_storage[f'fold_{fold_idx}'] = fold_importance_df  

        feature_importances_list.append(pd.DataFrame({'feature': feature_names, 'importance': importances}))
        weighted_importances_list.append(pd.DataFrame({'feature': feature_names, 'weighted_importance': weighted_importance_}))

    print("\n📊 Aggregating feature importances across folds...")

    # Compute mean feature importance
    fi_df = pd.concat(feature_importances_list).groupby('feature', as_index=False).mean()
    weighted_fi_df = pd.concat(weighted_importances_list).groupby('feature', as_index=False).mean()

    # Merge normal and weighted feature importances
    final_df = fi_df.merge(weighted_fi_df, on="feature", how="left")

    # Sort feature importances in descending order
    final_df = final_df.sort_values("importance", ascending=False).reset_index(drop=True)

    # Add a row for R² distribution
    r2_distribution_row = pd.DataFrame({'feature': ['R² Distribution'], 'importance': [r2_scores], 'weighted_importance': [np.nan]})
    final_df = pd.concat([final_df, r2_distribution_row], ignore_index=True)

    print("✅ Feature importance computation completed!")
    print(f"📈 Final DataFrame Shape: {final_df.shape}")

    return final_df, fold_importance_storage    
    
def get_fi_from_model(pipeline, df_X, y, fitted = False):
    model_type = pipeline.named_steps["regressor"].__class__.__name__
    if not fitted:
        pipeline.fit(df_X, y)
    if model_type == "RandomForestRegressor":
        
        feature_names =  pipeline[:-2].get_feature_names_out()

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
        feature_names = pipeline[:-2].get_feature_names_out()


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


def get_fi_from_shap(pipeline, df_X, y, fitted = False):
    """
    Fits the pipeline on (df_X, y), computes SHAP values using a TreeExplainer (if the regressor
    is tree-based) and returns a DataFrame with columns ['feature', 'importance'].
    The 'importance' here is the mean absolute SHAP value for each feature.
    
    For tree-based models (RandomForestRegressor, XGBRegressor) we use TreeExplainer;
    otherwise, a generic explainer can be used.
    """
    # Fit the pipeline on all data.
    if not fitted:
        pipeline.fit(df_X, y)
    
    # Get the transformed features and their names from the preprocessor.
    X_trans = pipeline[:-1].transform(df_X)
    feature_names = pipeline[:-1].get_feature_names_out()
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
        'importance': importances,
    }).sort_values('importance', ascending=False)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    return fi_df, shap_df, explainer


def get_fi_from_perm(pipeline, df_X, y, n_repeats=5, fitted = False):
    """
    Fits the pipeline on (df_X, y), then computes permutation importance (scoring='r2').
    Returns a DataFrame with columns ['feature', 'importance'],
    sorted by 'importance' (descending).
    """
    # Fit the entire pipeline
    if not fitted:
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
    importances = result.importances_mean # do averaging over performance 

    # Get feature names from the preprocessor step
    feature_names = pipeline[:-1].get_feature_names_out()

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