from pathlib import Path
import logging
from collections import OrderedDict

# for manipulating data
import numpy as np
import pandas as pd
import math
from typing import Callable
import copy
import re
from pandas.api.types import is_string_dtype, is_numeric_dtype

# for Machine Learning
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

# for visualization
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')



def fix_missing(df, col, name, na_dict):
    """
    Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name + '_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

    
def numericalize(df: pd.DataFrame, col: str, name: str, max_n_cat: int | None) -> pd.DataFrame:
    """
    Changes the column col from a categorical type to it's integer codes.
    """
    df = copy.deepcopy(df)
    if (not is_numeric_dtype(col) 
        and (max_n_cat is None or len(col.cat.categories) > max_n_cat)):
        df[name] = pd.Categorical(col).codes + 1
    return df


def process_df(df: pd.DataFrame,y_field: str | None = None,skip_flds: list | None = None):
    df=df.copy()
    if skip_flds is None:
        skip_flds = []
    else:
        skip_flds = list(skip_flds)
    if y_field is None:
        y = None
    else:
        y=df[y_field].values
        skip_flds.append(y_field)
    df = df.drop(columns=skip_flds)
    na_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col +'_na'] = df[col].isna().astype(int)
                na_cols.append(col + '_na')
                df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('Missing')
    df = pd.get_dummies(df,dummy_na=False)
    return df, y

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col == 'price_log':
            continue
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df