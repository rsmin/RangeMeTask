import pandas as pd
import pandasql as ps
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import json


def data_loader():
    return pd.read_csv('Data/Dataset - Sheet1.csv')


def params_loader():
    param_path = 'params.json'
    return_json = {}
    try:
        return_json = json.load(open(param_path))
    except Exception as e:
        print(e)
        pass
    return return_json


def exec_sql(query, env, task_id):
    result = ps.sqldf(query, env)
    print('Task ', task_id)
    print('With input query:\n', query)
    print('Result: \n', result)
    print('\n')


# formating the invalid values
def formatting_valid(input_df, invalid_values):
    for k, v_list in invalid_values.items():
        input_df.loc[input_df[k].isin(v_list),k] = np.nan
    return input_df


# formating the column types
def formatting_numeric(input_df, numeric_cols):
    for c in numeric_cols:
        if input_df[c].dtype.name == 'object':
            input_df.loc[:,c] = input_df[c].str.replace(',', '')
        input_df.loc[:,c] = input_df[c].astype(float)
    return input_df


# identify columns with missing value
def missing_columns(input_df):
    cols = [c for c in input_df if input_df[c].isnull().values.any()]
    return cols


# imputering missing values to moust frequent value
def imputering(input_df):
    mis_cols = missing_columns(input_df)

    for c in mis_cols:
        input_df.loc[:,c + '_was_missing'] = input_df[c].isnull()

    imputer = SimpleImputer(strategy='most_frequent')
    imputered_df = pd.DataFrame(imputer.fit_transform(input_df),
                                columns=input_df.columns)
    return imputered_df


def onehot_encoder(input_df, cols):
    if not isinstance(cols, list):
        cols = [cols]
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(input_df[cols])
    column_name = encoder.get_feature_names(cols)
    one_hot_df = pd.DataFrame(encoded, columns=column_name)
    return one_hot_df


# todo: test_size should be dynamically decided.
def data_split(input_df, feature_cols, target_col):
    X_train, X_test, Y_train, Y_test = train_test_split(input_df[feature_cols],
                                                        input_df[target_col],
                                                        test_size=0.3,
                                                        random_state=11)

    return X_train, X_test, Y_train, Y_test


# Pipeline modules
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self._cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return x[self._cols]


class CategoryFormatter(BaseEstimator, TransformerMixin):
    def __init__(self, invalid_values):
        self._invalid_values = invalid_values

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x = formatting_valid(x, self._invalid_values)
        for c in x.columns:
            x.loc[:,c + '_was_missing'] = x[c].isnull()
        return x


class NumericFormatter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x = formatting_numeric(x, x.columns)
        for c in x.columns:
            x.loc[:,c + '_was_missing'] = x[c].isnull()
        return x


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy):
        self.imputer = SimpleImputer(strategy=strategy)
        self.column_name = None
    def fit(self, x, y=None):
        self.imputer.fit(x)
        self.column_name = x.columns
        return self

    def transform(self, x, y=None):
        x = pd.DataFrame(self.imputer.transform(x),
                         columns=x.columns)
        return x

    def get_feature_names(self):
        return self.column_name


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(sparse=False)
        self.column_name = None

    def fit(self, x, y=None):
        self.encoder.fit(x)
        self.column_name = self.encoder.get_feature_names(x.columns)
        return self

    def transform(self, x, y=None):
        encoded = self.encoder.transform(x)
        one_hot_df = pd.DataFrame(encoded, columns=self.column_name)
        return one_hot_df

    def get_feature_names(self):
        return self.column_name


class FeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.filter = VarianceThreshold()
        self.selected_cols = []

    def fit(self, x, y=None):
        self.filter.fit(x)
        return self

    def transform(self, x, y=None):
        # filtered = self.filter.transform(x)
        filtered_df = x.loc[:, self.filter.get_support()]
        return filtered_df


class Concat(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_list):
        self.feature_union = FeatureUnion(transformer_list)

    def fit(self, x, y=None):
        self.feature_union.fit(x)
        return self

    def transform(self, x, y=None):
        column_name = self.feature_union.get_feature_names()
        concat_x = self.feature_union.transform(x)
        concat_df = pd.DataFrame(concat_x, columns=column_name)
        return concat_df
