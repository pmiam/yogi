"""PandasColumnTransformer source originally by Everest Law https://github.com/openerror"""
from itertools import chain
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

class FrameTransformer(BaseEstimator, TransformerMixin):
    """
    A wrapper around sklearn.column.ColumnTransformer facilitating the use of
    Scikit-Learn Transformers with all the benefits of pandas data frames in indexing
    and relational queries.
    
    example:
    df = FrameTransformer(StandardScaler()).fit_transform(df)
    """
    def __init__(self, transformers, **kwargs):
        """
        Initialize by creating ColumnTransformer object
        Args:
            transformers (list of length-3 tuples): (name, Transformer, target columns)
            kwargs: keyword arguments for sklearn.compose.ColumnTransformer

        in each tuple, "name" can be anything, but any choice of ["remainder",
        "default", "rem", "def", "drop", "pass", "exclude"] indicates that the
        columns specified in that section of the transformer pipeline should be
        treated by the ColumnTransformer's Transform-specific .remainder protocol
        """
        self.transformers = ColumnTransformer(transformers, **kwargs)
        self.transformed_col_names: List[str] = []

    def _get_col_names(self, X: pd.DataFrame):
        """
        Get names of transformed columns from a fitted self.transformers
        Args:
            X (pd.DataFrame): DataFrame to be fitted on
        Yields:
            Iterator[Iterable[str]]: column names corresponding to each transformer
        """
        for name, transformer, cols in self.transformers.transformers_:
            remainder_names = ["remainder", "default", "rem", "def", "drop", "pass", "exclude", "ignore"]
            if hasattr(transformer, "get_feature_names_out"):
                colnames = transformer.get_feature_names_out(cols)
                yield colnames
            elif name in remainder_names and self.transformers.remainder=="passthrough":
                yield X.columns[cols].tolist()
            elif name in remainder_names and self.transformers.remainder=="drop":
                continue
            else:
                yield cols        

    def fit(self, X: pd.DataFrame, y: Any=None):
        """
        Fit ColumnTransformer, and obtain names of transformed columns in advance
        Args:
            X (pd.DataFrame): DataFrame to fit the transformer to
            y (Any, optional): API Compliance. Defaults to None.
        """
        assert isinstance(X, pd.DataFrame)
        self.transformers = self.transformers.fit(X, y)
        self.transformed_col_names = list(chain.from_iterable(self._get_col_names(X)))
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a new DataFrame using fitted self.transformers
        Naturally, This is ONLY applicable to inductive estimators.
        No TSNE yet...
        Args:
            X (pd.DataFrame): DataFrame to be transformed
        Returns:
            pd.DataFrame: DataFrame transformed by self.transformers
        """
        assert isinstance(X, pd.DataFrame)
        transformed_X = self.transformers.transform(X)
        if isinstance(transformed_X, np.ndarray):
            return pd.DataFrame(transformed_X, index=X.index, 
            columns=self.transformed_col_names)
        else:
            return pd.DataFrame.sparse.from_spmatrix(
                transformed_X, index=X.index,
                columns=self.transformed_col_names
            )

# Authors: Chirag Nagpal
#          Christos Aridas

### This is an example implementation of a meta etimator specifically for providing the general predict and decision_function methods for any possible constituent 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, clone #cloning an estimator wipes its attributes?
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
# useful lads:
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted


N_SAMPLES = 5000
RANDOM_STATE = 42


def _classifier_has(attr):
    """Check if we can delegate a method to the underlying classifier.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    return lambda estimator: (
        hasattr(estimator.classifier_, attr)
        if hasattr(estimator, "classifier_")
        else hasattr(estimator.classifier, attr)
    )


class InductiveClusterer(BaseEstimator):
    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X)
        self.classifier_.fit(X, y)
        return self

    @available_if(_classifier_has("predict"))
    def predict(self, X):
        check_is_fitted(self)
        return self.classifier_.predict(X)

    @available_if(_classifier_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self)
        return self.classifier_.decision_function(X)
