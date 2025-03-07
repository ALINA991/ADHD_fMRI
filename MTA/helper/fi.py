import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 

from sklearn import set_config

set_config(transform_output="pandas") # keep output of each step as dataframe to preserve column (feature) names 
  

def compute_feature_importances(pipeline, df_X, y, groups, imp_method="internal", n_repeats=10, n_splits=5, precomputed_splits=None):
    """
    Compute feature importances using different methods: 'internal', 'perm', or 'shap'.
    
    Args:
        pipeline: The unfitted machine learning pipeline.
        df_X: Feature dataframe.
        y: Target variable.
        groups: Grouping variable for cross-validation.
        imp_method: Method to compute feature importances ('internal', 'perm', or 'shap').
        n_repeats: Number of repeats for permutation importance (only used if imp_method='perm').
        n_splits: Number of cross-validation splits.
        precomputed_splits: Predefined train-test splits (optional).

    Returns:
        final_df: DataFrame containing sorted feature importances.
        fold_importance_storage: Dictionary with per-fold feature importance DataFrames.
        r2_scores: List of R² scores per fold.
        train_test_splits: List of train-test indices per fold.
        raw_predictions: Dictionary containing true and predicted values per fold.
        full_permutation_importances (only if imp_method='perm'): Dictionary with full permutation importances.
        shap_values_dict (only if imp_method='shap'): Dictionary containing SHAP values per fold.
    """
    
    print(f"🚀 Computing feature importances using: {imp_method.upper()} method...")

    if precomputed_splits is None:
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(df_X, y, groups))
    else:
        splits = precomputed_splits  # Use the provided splits
    
    feature_importances_list = []
    weighted_importances_list = []
    r2_scores = []
    all_features = pd.Index([])  
    fold_importance_storage = {}  
    shap_values_dict = {}  
    raw_predictions = {}  
    train_test_splits = []  
    full_permutation_importances = {} if imp_method == "perm" else None  

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n📊 Processing Fold {fold_idx + 1}/{len(splits)}")

        X_train, X_test = df_X.iloc[train_idx], df_X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Store split indices
        train_test_splits.append({"train_idx": train_idx, "test_idx": test_idx})

        pipeline.fit(X_train, y_train)
        model = pipeline.named_steps['regressor']

        feature_names = pd.Index(pipeline.named_steps['rand_feature'].get_feature_names_out())  
        all_features = all_features.union(feature_names)  

        X_train_transformed = pipeline[:-2].transform(X_train)
        X_test_transformed = pipeline[:-2].transform(X_test)  

        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

        # Store raw predictions
        raw_predictions[f'fold_{fold_idx}'] = {
            'y_test': y_test,
            'y_pred': y_pred
        }

        print(f" Fold {fold_idx} R² Score: {r2:.4f}")
        print(f" Fold {fold_idx} Features: {len(feature_names)}")

        # **Extract feature importance based on method**
        if imp_method == "internal":
            print("⚙️ Extracting feature importances from the model...")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "get_booster"):  
                booster_importance = model.get_booster().get_score(importance_type="gain")
                importances = np.array([booster_importance.get(f, 0) for f in feature_names])
            else:
                raise ValueError("Selected model does not support internal feature importance.")

        elif imp_method == "perm":
            print("🔄 Computing permutation importance...")
            result = permutation_importance(model, X_test_transformed, y_test, n_repeats=n_repeats, random_state=42, scoring="r2")
            importances = result.importances_mean  

            # Store full permutation importance values per feature
            full_permutation_importances[f'fold_{fold_idx}'] = {
                'feature': feature_names.tolist(),
                'importances': result.importances  # Raw permutation importances (n_repeats x features)
            }

        elif imp_method == "shap":
            # Identify the model type
            model = pipeline.named_steps['regressor']
            
            # Ensure feature alignment
            X_test_transformed = X_test_transformed.reindex(columns=X_train_transformed.columns)

            # Apply model-specific imputation
            if isinstance(model, XGBRegressor):
                # XGBoost handles NaNs natively, so we leave them as NaN
                X_test_transformed = X_test_transformed.fillna(np.nan)
            elif isinstance(model, RandomForestRegressor):
                # RandomForest does not handle NaNs, so we impute with the mean of training data
                train_means = X_train_transformed.mean()
                X_test_transformed = X_test_transformed.fillna(train_means)
            else:
                raise ValueError("Unsupported model type for SHAP computation")
            
            print("🧩 Computing SHAP values...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test_transformed)  
            importances = np.abs(shap_values.values).mean(axis=0)

            # Store SHAP values per fold
            shap_values_dict[f'fold_{fold_idx}'] = {
                'shap_values': shap_values.values,  # Raw SHAP values
                'expected_value': explainer.expected_value,  
                'X_test': X_test,  
                'feature_names': feature_names.tolist()  
            }

        else:
            raise ValueError("Invalid feature importance method. Choose from 'internal', 'perm', or 'shap'.")

        print("Done.")
        # Compute weighted importance
        weighted_importance_ = importances * r2

        # Store per-fold feature importance
        fold_importance_df = pd.DataFrame({
            'feature': feature_names,
            f'importance_fold_{fold_idx}': importances
        })

        fold_importance_storage[f'fold_{fold_idx}'] = fold_importance_df  
        
        # Append to lists for aggregation
        feature_importances_list.append(pd.DataFrame({'feature': feature_names, 'importance': importances}))
        weighted_importances_list.append(pd.DataFrame({'feature': feature_names, 'weighted_importance': weighted_importance_}))
        
    
    print("\n📊 Aggregating feature importances across folds...")

    # Compute mean feature importance across all folds
    fi_df = pd.concat(feature_importances_list).groupby('feature', as_index=False).mean()

    # Compute weighted importance, but correctly handling missing features
    weighted_fi_df = pd.concat(weighted_importances_list).groupby('feature', as_index=False).sum()
    
    # Count how many folds each feature appeared in (nonzero values)
    feature_presence_count = pd.concat(weighted_importances_list).groupby('feature', as_index=False).count()
    
    # Compute correct weighted mean (sum divided by nonzero count)
    weighted_fi_df['weighted_importance'] = weighted_fi_df['weighted_importance'] / feature_presence_count['weighted_importance']

    # Merge normal and weighted feature importances
    final_df = fi_df.merge(weighted_fi_df[['feature', 'weighted_importance']], on="feature", how="left")

    # **Sort feature importances in descending order**
    final_df = final_df.sort_values("importance", ascending=False).reset_index(drop=True)

    print("✅ Feature importance computation completed!")
    print(f"Final DataFrame Shape: {final_df.shape}\n")

    # **Return Results**
    if imp_method == "shap":
        return final_df, fold_importance_storage, r2_scores, train_test_splits, raw_predictions, shap_values_dict
    elif imp_method == "perm":
        return final_df, fold_importance_storage, r2_scores, train_test_splits, raw_predictions, full_permutation_importances
    else:
        return final_df, fold_importance_storage, r2_scores, train_test_splits, raw_predictions

def get_importance_file_name(model_type, outcome_short, corr_select, thr_corr, params,  outcome_var, rater_out, rater_pred, thr_drop_missing):
    model_name_save = "RF" if model_type == "RandomForestRegressor" else "XGB"
    corr_select_save = "CS{}".format(thr_corr) if corr_select else "noCS"
    outcome_save = outcome_short
    rater_out_save = "out{}".format(rater_out.upper())
    rater_pred_save = "in{}".format(rater_pred.upper()) if rater_pred is not None else "inALL"
    thr_drop_missing_save = "thrDrop{}".format(thr_drop_missing)

    importance_file_name_save = "fi_{}_{}_{}_{}_{}_{}.pkl".format(outcome_save, model_name_save, corr_select_save,rater_out_save, rater_pred_save, thr_drop_missing_save )
    return importance_file_name_save