import pandas as pd
import numpy as np
from functools import partial

def _build_frame(tarray:np.ndarray, varray:np.ndarray, index,
                 names, partition="tv") -> pd.DataFrame:
    """
    index and names must be lists. partition can be either
    'test/validate' or 'fit/score time
    """
    mi = pd.MultiIndex.from_arrays(index, names=names)
    tframe = pd.DataFrame(tarray, index=mi)
    vframe = pd.DataFrame(varray, index=mi)
    if partition == 'tv':
        tframe = tframe.assign(partition="train")
        vframe = vframe.assign(partition="validate")
    elif partition == 'fs':
        tframe = tframe.assign(partition="fit[sec]")
        vframe = vframe.assign(partition="score[sec]")
    frame = pd.concat([tframe, vframe])
    return frame

def pandas_validation_curve(curve_maker, *args, **kwargs) -> pd.DataFrame:
    """
    Pass a sklearn validation curve creator function
    - learning_curve
    - validation_curve
    and the arguments intended for it. Scoring is handled differently.

    Scoring may be performed for various metrics simultaneously. Either:
    - Pass a dictionary of sklearn compliant scoring function(s) for
      simultaneous scoring
    otherwise, pass compliant scoring functions or recognized strings
    
    Returns a dataframe of validation results which can be readily
    plotted using the pandas ecosystem of EDA tools namely seaborn.
    """
    scoring = kwargs.get('scoring', None)
    kwargs['scoring'] = None
    if not isinstance(scoring, dict):
        scoring = {"default": scoring}
    #intercepting scoring arg
    param_range = kwargs.get('param_range', None)
    train_sizes = kwargs.get('train_sizes', None)
    if list(param_range):
        index = param_range #index in case of validation_curve
        index_name = kwargs.get('param_name')
    curve_maker = partial(curve_maker, *args, **kwargs)
    result_frames = []
    for score_name, scorer in scoring.items():
        curve_maker_spec = partial(curve_maker, scoring=scorer)
        results_tuple = curve_maker_spec()
        #get index in case of learning curve
        if len(results_tuple[0].shape) == 1:
            if len(results_tuple[0]) < len(train_sizes):
                index = train_sizes
                index_name = "train_portions"
            else:
                index = results_tuple[0]
                index_name = "train_sizes"
        score_log = [score_name] * len(index)
        if len(results_tuple) < 3:
            result_frames.append(
                _build_frame(results_tuple[0], results_tuple[1],
                             index=[score_log, index],
                             names=["score", index_name],
                             partition="tv"))
        if len(results_tuple) >= 3:
            result_frames.append(
                _build_frame(results_tuple[1], results_tuple[2],
                             index=[score_log, index],
                             names=["score", index_name],
                             partition="tv"))
        if kwargs.get("return_times"):
            result_frames.append(
                _build_frame(results_tuple[3], results_tuple[4],
                             index=[score_log, index],
                             names=["score", index_name],
                             partition="fs"))
    return pd.concat(result_frames)
