import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import FunctionTransformer
import numpy as np 


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

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        self.feature_names_in_ = list(X.columns) if hasattr(X, 'columns') else None
        return self

    def transform(self, X):
        return self.transformer.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else input_features

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

