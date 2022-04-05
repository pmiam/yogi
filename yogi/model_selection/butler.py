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
    rank_scores_per_metric = finalists_ranks.applymap(lambda x: x**-1).loc[setting]
    weighted_rank_scores_per_metric = np.dot(rank_scores_per_metric, weights)
    return np.sum(weighted_rank_scores_per_metric)


def score(setting:str, finalists_settings:pd.Series, finalists_ranks:pd.DataFrame, weights=None) -> float:
    return get_prob(setting, finalists_settings) + rank_score(setting, finalists_ranks, weights)

def reduce_grid_via_scoring_heuristic(settings:list,
                                      finalists_settings:pd.Series,
                                      finalists_ranks:pd.DataFrame,
                                      weights=None) -> list:
    setting_scores = []
    for setting in settings:
        setting_scores.append(score(setting, finalists_settings, finalists_ranks, weights=weights))

    print(settings, setting_scores)


def reduce_grid_via_entropy_rules(grid:dict,
                                  grid_entropy_series:pd.Series,
                                  podium:pd.DataFrame,
                                  weights=None) -> dict:
    finalists_ranks = podium.filter(like='rank', axis=1)
    new_grid = {}
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
            new_grid[param] = reduce_grid_via_scoring_heuristic(settings,
                                                                finalists_settings,
                                                                finalists_ranks,
                                                                weights=weights)
    return new_grid

def recommend_next_grids(podium, grid_entropies, grids:list) -> list:
    new_grids = []
    for idx, grid in enumerate(grids):
        new_grids.append(reduce_grid_via_entropy_rules(grid, grid_entropies.iloc[:, idx], podium))
    return new_grids

