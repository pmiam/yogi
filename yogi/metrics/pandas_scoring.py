from typing import Union

import pandas as pd
import numpy as np

from sklearn.base import clone

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

def _mkname(alist, blist):
    if len(alist) <= 5:
        name = "_&_".join(alist)
    elif len(blist) <=5:
        name = f'except {"_&_".join(blist)}'
    else:
        name = f'{len(alist)} samples'
    return name

def run_cv_report(estimator, cv, scorings:dict,
                  X, Y, groups=None, groups_labels=None):
    """
    test an estimator's ability to extrapolate using cross-validation in a dataset.

    optionally test across categories using group-aware cross-validators.
    in this case, categories must be provided both as ordinal and categorical labels for the time being.

    Score results are reported in a table for granular review. multiple scoring metrics can be used.
    """
    gen = cv.split(X, Y, groups=groups)
    #train and test index generators, in order
    trn_scores = []
    tst_scores = []
    for trn_idx, tst_idx in gen:
        fresh_estimator = clone(estimator) #start from scratch each training cycle
        trn_names = groups_labels.iloc[trn_idx].unique()
        tst_names = groups_labels.iloc[tst_idx].unique()
        fresh_estimator.fit(X.iloc[trn_idx], Y.iloc[trn_idx])
        trn_score_series = pd.Series(
            batch_score(fresh_estimator, X.iloc[trn_idx], Y.iloc[trn_idx], **scorings)
        )
        trn_score_series.name=_mkname(trn_names, tst_names)
        tst_score_series = pd.Series(
            batch_score(fresh_estimator, X.iloc[tst_idx], Y.iloc[tst_idx], **scorings)
        )
        tst_score_series.name=_mkname(tst_names, trn_names)
        trn_scores.append(trn_score_series)
        tst_scores.append(tst_score_series)
    trn_scores = pd.concat(trn_scores, axis=1).T.assign(partition="train")
    tst_scores = pd.concat(tst_scores, axis=1).T.assign(partition="test")
    group_scores = pd.concat([trn_scores, tst_scores])
    return group_scores
