from typing import Union
from collections.abc import Sequence

import pandas as pd
import numpy as np

from ast import literal_eval

def join3(left:pd.DataFrame, middle:pd.DataFrame, right:pd.DataFrame,
          thru:str, right_on:Union[str, Sequence], level:int=0):
    """convenience function for performing a database style 3-way join where:
    1. the left set is related to the right through the values in the middle
    2. the middle set should be provided with it's row dimension
       matching the left's and a named MultiIndex providing grouping
       collections
    3. thru identifies the label in middle through which to make the join
    4. right on can identify a corresponding key or array containing
       comparable values as that indicated by thru

    thru can be any of either middle's column names (names in a
    multiindex, not column labels) or, left's column labels.

    right labels can be row labels or columns labels.

    The grouping is always performed by the highest level column
    index.
    
    The join result is widened and re-indexed to be consistent with
    the original row dimension of left and middle.
    """
    #left.columns = list(map(literal_eval, left.add_suffix('",""').add_prefix('"').columns))
    #interesting but not robust for deep multiindex
    relations = pd.concat([left, middle], axis=1).set_index(left.columns.to_list(), append=False)
    relations.columns = pd.MultiIndex.from_tuples(relations.columns, names=middle.columns.names)
    relations = relations.reset_index().melt(id_vars=left.columns.to_list())
    relations = relations.replace(0, np.NaN) #avoid DIV by 0 
    relations = relations.dropna(axis=0, subset=["value"])
    join = pd.merge(left=relations, right=right, left_on=thru, right_on=right_on)
    join = join.set_index(left.columns.to_list(), append=False)
    derived = join.groupby(level=left.columns.to_list()).apply(
        lambda df: df.groupby(middle.columns.names[level]).apply(
            lambda df: pd.DataFrame(np.average(
                a=df.select_dtypes(include=np.number), axis=0, weights=df.value),
                                    index=df.select_dtypes(include=np.number).columns)))
    derived = derived.unstack(level=middle.columns.names[level]).unstack(level=-1)
    derived.columns=derived.columns.droplevel([0])
    derived = derived.drop(columns="value", level=-1)
    return derived

#make this a suitable function to feed to the df.transform method??

def robust_compare(compare1, on1, compare2, on2, how='inner', suf1='x', suf2='y'):
    """from comparisons, should add to yogi
    be careful, make sure that the difference between on1 and on2 is not too excessive
    """
    df1 = pd.concat([compare1, on1], axis=1)
    df2 = pd.concat([compare2, on2], axis=1)
    intersection = [v for v in on1.columns.to_list() if v in on2.columns.to_list()]
    difference = [v for v in on1.columns.to_list() + on2.columns.to_list() if v not in intersection]
    join = pd.merge(df1, df2, on=intersection,
                    how=how, suffixes=("_"+suf1, "_"+suf2))
    if difference:
        return join[join[difference].isna().agg('prod', axis=1).apply(bool)]
    else:
        return join
