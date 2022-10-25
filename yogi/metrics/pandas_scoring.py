from typing import Union

import pandas as pd
import numpy as np

import copy

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

def test_generality(estimator, groupKfold, scorings:dict,
                    X_tr, y_tr, groups_tr, groups_tr_labels,
                    X_ts, y_ts, groups_ts, groups_ts_labels):
    """
    tests an estimator's ability to extrapolate across categorical labels in a dataset.
    Score results are reported in a table. multiple scoring metrics can be defined.
    The categories must be provided both ordinarily and categorically for the time being.
    """
    gentpl = groupKfold.split(X_tr, y_tr, groups=groups_tr), groupKfold.split(X_ts, y_ts, groups=groups_ts)
    #train and test index generators, in order
    val_scores = []
    tst_scores = []
    for train_idx, val_idx, _, tst_idx in [sum(gengroup, ()) for gengroup in zip(*gentpl)]:
        fresh_estimator = copy.deepcopy(estimator) #start from scratch each training cycle
        tr_val_group_names = groups_tr_labels.iloc[val_idx].unique()
        ts_group_names = groups_ts_labels.iloc[tst_idx].unique()
        #fit to tr part
        fresh_estimator.fit(X_tr.iloc[train_idx], y_tr.iloc[train_idx])
        #get val and test scores
        tr_val_score_series = pd.Series(
            batch_score(fresh_estimator, X_tr.iloc[val_idx], y_tr.iloc[val_idx], **scorings)
        )
        tr_val_score_series.name="_&_".join(tr_val_group_names)
        ts_score_series = pd.Series(
            batch_score(fresh_estimator, X_ts.iloc[tst_idx], y_ts.iloc[tst_idx], **scorings)
        )
        ts_score_series.name="_&_".join(ts_group_names)
        val_scores.append(tr_val_score_series)
        tst_scores.append(ts_score_series)
    tr_val_scores = pd.concat(val_scores, axis=1).assign(partition="validation")
    ts_scores = pd.concat(tst_scores, axis=1).assign(partition="test")
    group_scores = pd.concat([tr_val_scores, ts_scores]).round(5).drop_duplicates(keep="first")
    return group_scores
