import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import FunctionTransformer
import numpy as np 
import dcor

from skrub import TableVectorizer


class DataFrameToNumpy(BaseEstimator, TransformerMixin):
    """Custom transformer to convert a pandas DataFrame to a NumPy array."""

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        if hasattr(X, "to_numpy"):  # Check if X is a DataFrame
            return X.to_numpy()
        return np.array(X)  

class TableVectorizerWrapper(TableVectorizer): # table vectorized inherently doesn support get_feature names out 
    def get_feature_names_out(self, input_features=None):
        # Call the original method ignoring the input_features argument.
        return super().get_feature_names_out()
    
class PreserveFeatureNamesRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, feature_names=None):
        self.regressor = regressor
        self.feature_names = feature_names

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        elif self.feature_names is not None:
            self.feature_names_in_ = self.feature_names
        else:
            raise ValueError("Input data has no column names; please provide feature_names.")
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor.predict(X)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
class PreserveFeatureNames(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        # Fit the underlying transformer
        self.transformer.fit(X, y)

        # If the underlying transformer knows how to provide expanded names...
        if hasattr(self.transformer, "get_feature_names_out"):
            # If X has columns, pass them as input_features
            # Otherwise, pass None
            input_features = list(X.columns) if hasattr(X, "columns") else None
            try:
                self.feature_names_out_ = self.transformer.get_feature_names_out(input_features)
            except TypeError:
                # Some transformers only accept no arguments
                self.feature_names_out_ = self.transformer.get_feature_names_out()
        else:
            # Fall back to the original columns (or None) if no name method is available
            if hasattr(X, "columns"):
                self.feature_names_out_ = list(X.columns)
            else:
                # If X isn't a DataFrame, we only know how many columns there are after transform
                self.feature_names_out_ = None

        return self

    def transform(self, X):
        return self.transformer.transform(X)

    def get_feature_names_out(self, input_features=None):
        # If we successfully extracted real names, return them
        if self.feature_names_out_ is not None:
            return self.feature_names_out_
        else:
            # Otherwise, we might do a shape-based placeholder,
            # but typically it's best to rely on the transformer's own method
            return input_features if input_features is not None else []
# ----------------------------
# Custom Transformers
# ----------------------------

class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.to_drop_ = None
        self.features_to_keep_ = None
        self.feature_names_in_ = None

    def _ensure_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        if self.feature_names_in_ is None:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def fit(self, X, y=None):
        X_df = self._ensure_dataframe(X)
        if self.feature_names_in_ is None:
            self.feature_names_in_ = list(X_df.columns)
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.features_to_keep_ = [col for col in X_df.columns if col not in self.to_drop_]
        return self

    def transform(self, X):
        X_df = self._ensure_dataframe(X)
        X_transformed = X_df[self.features_to_keep_]
        if isinstance(X, pd.DataFrame):
            return X_transformed
        return X_transformed.values

    def get_support(self, indices=False):
        if indices:
            return [self.feature_names_in_.index(feat) for feat in self.features_to_keep_]
        return self.features_to_keep_

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.feature_names_in_ is None:
                raise ValueError("feature_names_in_ is not set. Fit the transformer first.")
            input_features = self.feature_names_in_
        return [feat for feat in input_features if feat in self.features_to_keep_]


class LeastDistanceCorrelatedRandomFeature(BaseEstimator, TransformerMixin):
    def __init__(self, n_candidates=20, random_state=None, feature_name="rand_feature"):
        self.n_candidates = n_candidates
        self.random_state = random_state
        self.feature_name = feature_name

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Target vector y must be provided.")
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        best_dcorr = np.inf
        best_candidate = None
        for _ in range(self.n_candidates):
            candidate = rng.normal(size=n_samples)
            d_corr = dcor.distance_correlation(candidate, y)
            if np.isnan(d_corr):
                d_corr = np.inf
            if d_corr < best_dcorr:
                best_dcorr = d_corr
                best_candidate = candidate
        self.best_mean_ = np.mean(best_candidate)
        self.best_std_ = np.std(best_candidate)
        # Store the input feature names from X so we can use them later
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        n = X.shape[0]
        # Generate a fresh random vector with the same distribution as the best candidate
        new_rand = np.random.normal(loc=self.best_mean_, scale=self.best_std_, size=n)
        # If X is a DataFrame, append the new feature column with its name.
        if hasattr(X, "assign"):
            X_new = X.copy()
            X_new[self.feature_name] = new_rand
            return X_new
        else:
            return np.hstack([X, new_rand.reshape(-1, 1)])

    def get_feature_names_out(self, input_features=None):
        # Use stored input features if none are provided
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                input_features = self.feature_names_in_
            else:
                raise ValueError("No input features available; fit the transformer first.")
        # Append the name of the new feature to the provided feature names
        return list(input_features) + [self.feature_name]