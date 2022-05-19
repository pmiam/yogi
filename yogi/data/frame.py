import logging
logfmt = '[%(levelname)s] %(asctime)s - %(message)s'
logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", format=logfmt)

import pandas as pd
import numpy as np

#scikit accessor implementation
import json
import re
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

def _serialize_columns(df:pd.DataFrame):
    """
    Ensure pandas columns are always a series of strings, even if
    MultiIndexed.
    Base method useful for circumventing column name futurewarning via
    yogi decorator.
    """
    new_col_labels = []
    for label in df.columns:
        new_col_labels.append(json.dumps(label))
    df.columns=new_col_labels
    return df

def _deserialize_columns(df:pd.DataFrame):
    """
    reconstruct tuple from earlier json.dumps prefixing
    the highest level label names as needed
    """
    def prefix_last_label(prefix_label):
        if prefix_label[0] == "[":
            return prefix_label
        else:
            ll = re.split("__", prefix_label)
            sll = list(re.split(",", ll[-1]))
            lll = sll[-1]
            lll = lll.insert(1, ll[:-1]+"__")
            last = "".join(lll)
            sll[-1] = last
            return "".join(sll)
    final_col_labels = []
    for dump_label in df.columns:
        ready_label = prefix_last_label(dump_label)
        final_col_labels.append(tuple(json.loads(ready_label)))
    df.columns = pd.MultiIndex.from_tuples(final_col_labels)
    return df

@pd.api.extensions.register_dataframe_accessor("sk")
class SciKitAccessor():
    """
    Convenience accessor for applying sklearn transforms to dataframes.

    example:
    df = df.sk.pipe(StandardScaler().fit_transform)
    Xpca = X.sk.pca()

    In this way, transforms stay fully contextualized by the dataframe
    indices
    """
    def __init__(self, df):
        """
        sklearn transformers are very careful about the types they work on.
        object columns should definitely contain strings.
        numeric columns should definitely contain numbers or NaN

        DataFrame column labels absolutely must be strings
        Column names are preserved for restructuring
        """
#        self._df = pd.concat(
#            [df.select_dtypes(include=np.number).apply(pd.to_numeric, errors="coerce"),
#             # also handle categorical/sparse cols
#             df.select_dtypes(include=object).applymap(lambda x: str(x))],
#             axis=1
#        ) #reorders columns by type....
        self._df = _serialize_columns(df)
    
    def pipe(self, transformer):
        transformed_data = transformer(self._df)
        columns = _deserialize_columns(self._df).columns
        df = pd.DataFrame(
            transformed_data,
            index=self._df.index,
            columns=columns
        )
        return df
    
    @staticmethod
    def _gen_transform_tuple(num_transformer=None, obj_transformer=None):
        """generate a 1or2 tuple of 2-tuples"""
        if num_transformer and obj_transformer:        
            ct_list = [(num_transformer,
                        make_column_selector(dtype_include=np.number)),
                       (obj_transformer,
                        make_column_selector(dtype_include=object))]
        elif num_transformer:
            ct_list = [(num_transformer,
                        make_column_selector(dtype_include=np.number))]
        elif obj_transformer:
            ct_list = [(obj_transformer,
                        make_column_selector(dtype_include=object))]
        else:
            raise ValueError("Specify at least one compliant transformer")
        return ct_list

    def _fit(self, num_transformer, obj_transformer):
        ct = make_column_transformer(*self._gen_transform_tuple(num_transformer,
                                                                obj_transformer))
        # it'd be nice to shorten the generated names here
        transformer = FrameTransformer(ct.transformers)
        self.FT = transformer.fit(self._df)
        for ttpl in self.FT.transformers.transformers:
            yield ttpl[1]

    def fit(self, num_transformer=None, obj_transformer=None):
        """
        Pass transformers to apply to numeric (arg1) and non numeric
        (arg2) columns of DataFrame

        Valid estimators include FeatureUnions, Pipelines ending in
        a transformer, and any custom sklearn compliant estimators

        not sure if it will work with transformers that expect to work
        with methods of DataFrames (test df.dt?)

        returns:
        1-or-2-tuple of the fitted transformers passed by
        the user.
        
        To use the sk accessor's transform method on another dataframe
        of the same dimensions, a fitted FrameTransformer must be
        supplied. The Full FrameTransformer can be obtained from the
        accessor's TF attribute directly if induction is desired

        example:
        df.sk.fit(make_union(StandardScaler(), MinMaxScaler()), OneHotEncoder())

        Note:
        it's a column_transformer underneath so transductive
        estimators will raise TypeError: transform method not
        implemented
        """
        transgen = self._fit(num_transformer=num_transformer,
                             obj_transformer=obj_transformer)
        return tuple(transgen) #eh, pretty convenient but could be cleaner
        
    def transform(self, FT):
        """
        takes a FrameTransformer fit on another dataframe

        Assumes FrameTransformer has been fit previously.
        """
        #ideally check that the damned thing's been fit
        df = FT.transform(self._df)
        df = self._deserialize_columns(df)
        df.columns.names = self._col_names
        return df

    def fit_transform(self, num_transformer=None, obj_transformer=None):
        self.fit(num_transformer=num_transformer,
                 obj_transformer=obj_transformer)
        return self.transform(self.FT)

    def target(self, y):
        """
        supply a target dataframe to the accessor in order to perform
        supervised transformations
        """
        pass

@pd.api.extensions.register_dataframe_accessor("model")
class ModelAccessor():
    """
    
    - df.sk.transform() method supports generic Scikit-Learn compliant
      contextual transforms to a dataframe
    
    - df.sk.model() method supports applying SciKit-Learn compliant estimators


    Generate models of tables based on other tables
    
    automates the construction of a sklearn compliant pipeline. This includes:
    - pandas index manipulation such that results are appropriately labeled
    - 0.8train/0.2test split (default if not specified by the user)
    - labeling records with their respective partition
    - application of estimator
    
    An accessed table will have models created for it. models are
    stored with the accessor instance, predictions are not.

    Y.model.pipe(RandomForestRegressor.fit(X, Y).predict(X)))
    """
    def __init__(self, Y):
        self._validate(Y)
        #getting original
        self._df = Y
        #existing models
        self._RFR = None
        
    @staticmethod
    def _validate(Y):
        """
        verify Y contains numerical data

        flag weather categorical or continuous?
        """
        pass
    
    def base(self):
        return self._df
    
    def _do_RFR(self, X, **kwargs): #extend args?
        modeler = RFR(X, self._df, **kwargs)
        modeler.train_test_return()
        self._RFR = modeler
    
    def RFR(self, X, **kwargs):
        """
        return a model of Y based on X, The form of X used,
        and optionally the model used to get Y.

        Both Y and X are returned with pandas multindex

        this can be used to access the train/test split for each
        dataframe conveniently using pandas tuple indexing, the pandas
        IndexSlice module, or the .xs (cross section) method.
        """
        if self._RFR is None:
            self._do_RFR(X, **kwargs)
        return self._RFR.Y_stack, self._RFR.X_stack, self._RFR.r
