import pandas as pd
import json
import re

class SerialMI:
    """
    Context manager within which Pandas DataFrame MultiIndexed
    Columns are appropriately indexed with strings obtained by
    serializing their constituent tuples.

    intended to handle contexts in which transformations performed
    on the DataFrame which manipulating the columns as strings.
    The DataFrame exits the context block with a MultiIndex

    example:
    
    pipe = sklearn.compose.Pipeline(steps)
    with SerialMI(df) as df:
        pipe.fit(df)
        columns=pipe[-1].get_feature_names_out(df.columns.to_list())
    tdf = pd.DataFrame(
             pipe.transform(df),
             columns=columns
                       )
    """
    def __init__(self, df):
        self.df = df

    def __enter__(self):
        serial_columns = self._serialize_columns(self.df.columns.to_list())
        self._df = pd.DataFrame(self.df.values,
                                index = self.df.index,
                                columns = serial_columns
                                )
        return self._df

    def __exit__(self, type, value, traceback):
        """ anticipate possible column mismatch, de-allocate resources """
        return True

    def _serialize_columns(self, columns:list):
        """
        Ensure pandas columns are always a series of strings, even if
        MultiIndexed.
        """
        scol = []
        for label in columns:
            scol.append(json.dumps(label))
        return scol

### Some Implementation ideas for generally handling transformed df construction...
# not easy, probably too opinionated to be broadly applicable

# in fact, it is probably better to offer a "serialization artifact
# removal" function as a utility in spyglass for easing the use of the
# tdf directly in tabulating and plot rendering

# def _prefix_last_label(column_label):
#     """
#     handle circumstance where applied transformer modifies the column
#     labels by prefixing it's name
#     """
#     return column_label
#     #if column_label[0] == "[":
#     #    return column_label
#     #else:
#     #    ll = re.split("__", column_label)
#     #    sll = list(re.split(",", ll[-1]))
#     #    lll = sll[-1]
#     #    slll = re.split("", lll)
#     #    lll = slll.insert(1, ll[:-1]+"__")
#     #    last = "".join(lll)
#     #    sll[-1] = last
#     #    return "".join(sll)
# 
# def _deserialize_columns(df:pd.DataFrame, names=None):
#     """
#     reconstruct tuple from earlier json.dumps prefixing the innermost
#     (highest) level label names as needed.  Mutates the dataframe
#     directly.
#     """
#     final_col_labels = []
#     for dump_label in df.columns:
#         ready_label = _prefix_last_label(dump_label)
#         final_col_labels.append(tuple(json.loads(ready_label)))
#     df.columns = pd.MultiIndex.from_tuples(final_col_labels, names=names)
# 
