import pandas as pd
import numpy as np

def collect_top_rankings(results:pd.DataFrame, topN:int = 10) -> pd.DataFrame:
    ranks = results.filter(like='rank').columns
    rank_podiums = []
    for rank in ranks:
        rank_podiums.append(results
                            .reindex(index=results[f"{rank}"]
                                     .sort_values().index)
                            .iloc[0:topN, :])
        podium = pd.concat(rank_podiums, axis=0)
        podium = podium.reset_index().drop_duplicates(subset=["index"]).set_index("index")
    return podium

def get_prob(setting:str, finalists_settings:pd.Series) -> float:
    return finalists_settings.to_list().count(setting)/len(finalists_settings)

def shannon_entropy(finalists_settings:pd.Series, settings:list) -> float:
    def gain_information(setting):
        prob = get_prob(setting, finalists_settings)
        if prob == 0:
            return 0
        else:
            logprob = np.log(prob)
            return prob * logprob
    return -sum(map(gain_information, settings))

def get_entropies(podium:pd.DataFrame, grid=list) -> pd.DataFrame:
    entropy_frame = {}
    series_list = []
    for num, subspace in enumerate(grid):
        entropy_frame[str(num)] = {}
        for param, settings in subspace.items():
            finalists_settings = podium.filter(like=param, axis=1).iloc[:, 0]
            entropy_frame[str(num)][f"{param}"] = shannon_entropy(finalists_settings, settings)
        series_list.append(pd.Series(entropy_frame[str(num)]))
    return pd.concat(series_list, axis=1)

def rank_score(setting:str, finalists_ranks:pd.DataFrame, weights=None) -> float:
    if weights == None:
        weights = [1]*finalists_ranks.shape[1] 
    try:
        rank_scores_per_metric = finalists_ranks.applymap(lambda x: x**-1).loc[setting]
        weighted_rank_scores_per_metric = np.dot(rank_scores_per_metric, weights)
        return np.sum(weighted_rank_scores_per_metric)
    except KeyError: 
        #if a certain setting doesn't make it to finals at all, there'll be a KeyError
        #instead, ranking it zero
        return 0


def score(setting:str, finalists_settings:pd.Series, finalists_ranks:pd.DataFrame, weights=None) -> float:
    return get_prob(setting, finalists_settings) + rank_score(setting, finalists_ranks, weights)

def pick_top_settings(settings:list, setting_scores:list) -> list:
    return [settings[np.argmax(setting_scores)]]

def pick_over_average_settings(settings:list, setting_scores:list) -> list:
    return [setting for setting, score in zip(settings, setting_scores) if score >= np.average(setting_scores)]

def reduce_grid_via_scoring_heuristic(settings:list,
                                      finalists_settings:pd.Series,
                                      finalists_ranks:pd.DataFrame,
                                      weights=None,
                                      strategy:str="oavg") -> list:
    setting_scores = []
    for setting in settings:
        setting_scores.append(score(setting, finalists_settings, finalists_ranks, weights=weights))
    if strategy.casefold() == "oavg":
        return pick_over_average_settings(settings, setting_scores), setting_scores
    elif strategy.casefold() == "max":
        return pick_top_settings(settings, setting_scores), setting_scores

def reduce_grid_via_entropy_rules(grid:dict,
                                  grid_entropy_series:pd.Series,
                                  podium:pd.DataFrame,
                                  weights=None,
                                  strategy:str="oavg") -> dict:
    finalists_ranks = podium.filter(like='rank', axis=1)
    new_grid = {}
    grid_scores = {}
    for param, settings in grid.items():
        finalists_settings = podium.filter(like=param, axis=1).iloc[:,0]
        finalists_ranks = finalists_ranks.set_index(
            podium.filter(like=param, axis=1).iloc[:,0])
        if grid_entropy_series[param] == 0.0:
            if len(settings) == 1:
                new_grid[param] = settings
            if len(settings) > 1:
                new_grid[param] = [finalists_settings.iloc[0]]
        elif pd.isna(grid_entropy_series[param]):
            pass
        else:
            reduced, scores = reduce_grid_via_scoring_heuristic(settings,
                                                                finalists_settings,
                                                                finalists_ranks,
                                                                weights=weights,
                                                                strategy=strategy)
            new_grid[param] = reduced
            grid_scores[param] = scores
    return new_grid, grid_scores

def recommend_next_grids(podium, grid_entropies, grids:list, weights=None, strategy:str="oavg") -> list:
    recommendation = []
    for idx, grid in enumerate(grids):
        recommendation.append(reduce_grid_via_entropy_rules(grid, grid_entropies.iloc[:, idx], podium,
                                                            weights=weights, strategy=strategy))
    new_grids = list(map(lambda x: x[0], recommendation))
    score_summary = list(map(lambda x: {k:[round(s, 2) for s in v] for k,v in x[1].items()}, recommendation))
    return new_grids, score_summary

def summarize_HPO(fitted_HPO_pipeline, gridspace:list, topN=10, metric_weights=None, strategy="oavg") -> list:
    """
    helper function for performing rigorous gridsearch

    arguments:
    fitted_HPO_pipline: a gridsearch (or other sklearn compliant
          search) estimator containing a cv_results_ dictionary
    gridspace: The list of dictionaries used to instantiate the search
          cross-validator
    topN: the N best performing estimators (according to each scoring
          metric) to consider when narrowing the parameter space
    metric_weights: defaults to a list of 1s of equal length to the
          number of scoring metrics.

    this parameter offers extensive control over the scoring huristic,
    which is a sum of the probability of a setting appearing in the
    topN candidates and the sum of the inverse rankings used to
    collect the finalists. If high rank is more important than
    representation in the distribution of finalist configuration,
    evenly increase the weights over 1. If vice-versa, evenly
    decrease. If a partiuclar scoring metric is more important,
    increase it's weight.

    strategy: finally, a string indicating the method for choosing
           which parameter to keep. Currently "max" and "oavg" offer,
           respectively, an aggressive cut (only the top scoring
           parameter is kept) and a gentle cut (all parameters scoring
           greater than the average are kept).

    returns a tuple:
    In first position: a dataframe summarizing the scoring process
    In second position: The gridspaces are returned with low
    performing settings (according to the scoring heuristic)
    automatically discarded, for easy of iterating
    """
    results = pd.DataFrame(fitted_HPO_pipeline.cv_results_)
    podium = collect_top_rankings(results=results, topN=topN)
    grid_entropies = get_entropies(podium, gridspace)
    next_grid, scores = recommend_next_grids(podium, grid_entropies, gridspace,
                                             weights=metric_weights, strategy=strategy)
    original = pd.DataFrame(gridspace).T
    trim_summary = pd.DataFrame(next_grid).T
    score_summary = pd.DataFrame(scores).T
    original.rename(columns={k:"space_"+str(k) for k in trim_summary.columns}, inplace=True)
    trim_summary.rename(columns={k:"next_"+str(k) for k in trim_summary.columns}, inplace=True)
    score_summary.rename(columns={k:"scores_"+str(k) for k in score_summary.columns}, inplace=True)
    grid_entropies.rename(columns={k:"entropy_"+str(k) for k in grid_entropies.columns}, inplace=True)
    return pd.concat([original, grid_entropies, score_summary, trim_summary], axis=1), next_grid
