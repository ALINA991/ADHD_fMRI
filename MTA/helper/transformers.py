import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import FunctionTransformer
import numpy as np 
import dcor

from skrub import TableVectorizer
from xgboost import XGBRegressor 


class DataFrameToNumpy(BaseEstimator, TransformerMixin):
    """Custom transformer to convert a pandas DataFrame to a NumPy array."""

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        if hasattr(X, "to_numpy"):  # Check if X is a DataFrame
            return X.to_numpy()  # Convert to NumPy array
        return np.array(X)  # Convert any other type to NumPy


class TableVectorizerWrapper(TableVectorizer):
    """Wrapper for TableVectorizer to ensure feature names are correctly retrieved."""

    def get_feature_names_out(self, input_features=None):
        return super().get_feature_names_out()  # Retrieve feature names from parent class

    def transform(self, X):
        return super().transform(X)  # Apply transformation to input data

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is not None:
            return self.feature_names_out_  # Return stored feature names if available
        return input_features if input_features is not None else []  # Default to input feature names


# ----------------------------
# Custom Transformers
# ----------------------------
class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Selects features by removing highly correlated ones, keeping the most important."""

    def __init__(self, threshold=0.8, model=None):
        self.threshold = threshold  # Correlation threshold for feature removal
        self.to_drop_ = None  # List of features to drop
        self.features_to_keep_ = None  # List of features to keep
        self.model = model  # Model used for feature importance

    def _ensure_dataframe(self, X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)  # Convert to DataFrame if needed

    def fit(self, X, y=None):
        X_df = self._ensure_dataframe(X)  # Ensure input is a DataFrame

        # **Step 1: Fit a Model and Extract Feature Importance**
        if self.model is None:
            self.model = XGBRegressor(n_estimators=50, random_state=42)  # Use default XGBoost model if none provided
        self.model.fit(X_df, y)  # Train the model

        # Extract feature importance scores
        if hasattr(self.model, "feature_importances_"):
            importance_dict = dict(zip(X_df.columns, self.model.feature_importances_))
        elif hasattr(self.model, "get_booster"):
            booster_importances = self.model.get_booster().get_score(importance_type="gain")
            importance_dict = {f: booster_importances.get(f, 0) for f in X_df.columns}
        else:
            raise ValueError("Model does not support feature importance extraction.")  # Raise error if model is incompatible

        feature_importances = pd.Series(importance_dict)  # Convert to pandas Series

        # **Step 2: Compute Correlation Matrix**
        corr_matrix = X_df.corr().abs()  # Compute absolute correlation values
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Upper triangular matrix

        to_remove = set()
        for col in upper.columns:
            correlated_features = upper.index[upper[col] > self.threshold].tolist()  # Find highly correlated features
            if not correlated_features:
                continue
            correlated_features.append(col)  # Add current feature to comparison
            best_feature = max(correlated_features, key=lambda f: feature_importances.get(f, 0))  # Keep most important feature
            correlated_features.remove(best_feature)  # Remove the redundant ones
            to_remove.update(correlated_features)  # Store features to drop

        self.to_drop_ = list(to_remove)  # Save features to be removed
        self.features_to_keep_ = [col for col in X_df.columns if col not in self.to_drop_]  # Save features to keep

        return self

    def transform(self, X):
        X_df = self._ensure_dataframe(X)  # Convert input to DataFrame if needed
        return X_df[self.features_to_keep_]  # Return DataFrame with only selected features

    def get_feature_names_out(self, input_features=None):
        return self.features_to_keep_  # Return selected feature names


class LeastDistanceCorrelatedRandomFeature(BaseEstimator, TransformerMixin):
    """Transformer to generate a random feature with minimal correlation to y."""

    def __init__(self, n_candidates=20, random_state=None, feature_name="rand_feature"):
        self.n_candidates = n_candidates  # Number of random feature candidates
        self.random_state = random_state  # Random seed for reproducibility
        self.feature_name = feature_name  # Name of generated feature
        self.feature_names_in_ = None  # Store original feature names

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Target vector y must be provided.")  # Ensure y is provided

        rng = np.random.RandomState(self.random_state)  # Initialize random number generator
        n_samples = X.shape[0]  # Get number of samples
        best_dcorr = np.inf  # Initialize best correlation distance
        best_candidate = None  # Store best random feature

        for _ in range(self.n_candidates):
            candidate = rng.normal(size=n_samples)  # Generate a random normal feature
            d_corr = dcor.distance_correlation(candidate, y)  # Compute distance correlation
            if np.isnan(d_corr):
                d_corr = np.inf  # Handle NaN values
            if d_corr < best_dcorr:
                best_dcorr = d_corr  # Store lowest correlation distance
                best_candidate = candidate  # Save best random feature

        self.best_mean_ = np.mean(best_candidate)  # Store mean of selected feature
        self.best_std_ = np.std(best_candidate)  # Store standard deviation

        # Store input feature names
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)  # Keep original column names
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]  # Create placeholder names

        return self

    def transform(self, X):
        n = X.shape[0]  # Get number of samples
        new_rand = np.random.normal(loc=self.best_mean_, scale=self.best_std_, size=n)  # Generate new feature

        if hasattr(X, "assign"):
            X_new = X.copy()
            X_new[self.feature_name] = new_rand  # Add random feature to DataFrame
            return X_new
        else:
            return np.hstack([X, new_rand.reshape(-1, 1)])  # Append new feature to NumPy array

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.feature_names_in_ is None:
                raise ValueError("No input features available; fit the transformer first.")  # Ensure fit was called
            input_features = self.feature_names_in_
        return list(input_features) + [self.feature_name]  # Append new feature name to original list
    
    
    
    
    
    
    # class PreserveFeatureNames(BaseEstimator, TransformerMixin):
    # def __init__(self, transformer):
    #     self.transformer = transformer
    #     self.feature_names_out_ = None

    # def fit(self, X, y=None):
    #     # Fit the underlying transformer
    #     self.transformer.fit(X, y)

    #     # If the underlying transformer knows how to provide expanded names...
    #     if hasattr(self.transformer, "get_feature_names_out"):
    #         # If X has columns, pass them as input_features
    #         # Otherwise, pass None
    #         input_features = list(X.columns) if hasattr(X, "columns") else None
    #         try:
    #             self.feature_names_out_ = self.transformer.get_feature_names_out(input_features)
    #         except TypeError:
    #             # Some transformers only accept no arguments
    #             self.feature_names_out_ = self.transformer.get_feature_names_out()
    #     else:
    #         # Fall back to the original columns (or None) if no name method is available
    #         if hasattr(X, "columns"):
    #             self.feature_names_out_ = list(X.columns)
    #         else:
    #             # If X isn't a DataFrame, we only know how many columns there are after transform
    #             self.feature_names_out_ = None

    #     return self