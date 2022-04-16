from typing import Union

import pandas as pd
import numpy as np

class PandasScoreAdaptor():
    def __init__(self, sk_metric):
        """
        creates an index-aware scoring function which allows sklearn
        compliant model optimizers to weigh samples during evaluation
        according a series of weights with an identical index to that of
        the true targets

        Instantiate with a scikit-learn compliant model performance metric
        or scoring function. Then, call and return the index aware scorer
        through the score method
        """
        self.sk_metric = sk_metric
        
    def score(self,
              y_true:Union[pd.Series,pd.DataFrame],
              y_pred:np.array,
              sample_weight:pd.Series,
              *args, **kwargs) -> any:
        """
        args:
        y_true - accepts a ground-truth target DataFrame or Series.
                 If used as part of a sequential estimator's scoring
                 facility, the estimator should be fit on a dataframe
        y_pred - an np.array returned by transform/predict methods
                 If used as part of a sequential estimator's scoring
                 facility, the estimator's cross-validation strategy
                 will automatically generate this in correspondence
                 with the portion of the training set withheld in that
                 instance
        sample_weight - a pd.Series with identical index to the
                 complete y_true passed to the estimator's fit method
        extra arguments are passed to the wrapped metric function.
        """
        if sample_weight.reindex(y_true.index).dropna().sum() == 0:
            sample_weight = None
        else:
            sample_weight = sample_weight.reindex(y_true.index).dropna().values.reshape(-1)
        return self.sk_metric(y_true.values, y_pred,
                              sample_weight=sample_weight,
                              *args, **kwargs)

def batch_score(estimator,
                X,
                y:Union[pd.DataFrame,pd.Series,None]=None,
                **scorer_dict) -> list:
    """
    metrics must be made scorers before compatible with this
    convenience function. Use sklearn.metrics make_scorer function.
    """
    scores = {}
    for name, scorer in scorer_dict.items():
        scores[name] = scorer(estimator, X, y)
    return scores
