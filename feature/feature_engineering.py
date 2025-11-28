import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
import warnings
warnings.filterwarnings('ignore')


def outliers_proc(data,col_name,scale=3):
    def box_plot_outliers(data_serie,box_scale):
        iqr=box_scale*(data_serie.quantile(0.9)-data_serie.quantile(0.1))
        val_low=data_serie.quantile(0.1)-iqr
        val_up=data_serie.quantile(0.9)+iqr
        rule_low=(data_serie<val_low)
        rule_up=(data_serie>val_up)
        return(rule_low,rule_up),(val_low,val_up)
    data_n=data.copy()
    data_series = data_n[col_name].dropna()
    rule,value=box_plot_outliers(data_series,box_scale=scale)
    index=np.arange(data_series.shape[0])[rule[0]| rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n=data_n.drop(index)
    data_n.reset_index(drop=True,inplace=True)
    print("Now row count is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())
    return data_n

def feature_engineering(df):
    df = df.copy()

    df['notRepairedDamage'] = df['notRepairedDamage'].replace({'-': np.nan}).astype(float)
    df['used_time'] = (pd.to_datetime(df['creatDate'], format='%Y%m%d', errors='coerce') - pd.to_datetime(df['regDate'], format='%Y%m%d', errors='coerce')).dt.days
    df['city'] = df['regionCode'].apply(lambda x: int(str(x)[-2:]))
    df['km_per_year'] = df['kilometer'] / (df['used_time'] / 365 + 1)
    df['power_age'] = df['power'] * (df['used_time'] / 365)
    bin = [i*10 for i in range(31)]
    df['power_bin'] = pd.cut(df['power'], bin, labels=False)
    df[['power_bin', 'power']].head()
    df.drop(['creatDate','regDate','regionCode'], axis=1, inplace=True)

    return df
