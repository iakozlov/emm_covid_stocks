import math
import warnings
import pandas as pd
import statsmodels.api as sm

from copy import deepcopy
from scipy.spatial.distance import cosine
import numpy as np


warnings.filterwarnings('error')
items = None
cache = None
EPSILON = 1e-10


def cleanup():
    global items, cache
    items = None
    cache = None


def entropy(subgroup_target, dataset_target):
    """

    Args:
        subgroup_target:
        dataset_target:

    Returns:

    """
    n_c = max(1, len(dataset_target) - len(subgroup_target))
    n = len(subgroup_target)
    N = len(dataset_target)
    return -n/N * math.log(n/N) - n_c/N * math.log(n_c/N)


def distribution_cosine(subgroup_target, dataset_target, use_complement=False):
    global items, cache
    if len(subgroup_target.columns) > 1:
        raise ValueError("Distribution cosine expect exactly 1 column as target variable")
    column = list(subgroup_target.columns)[0]
    if cache is None:
        cache = dataset_target[column].value_counts()
        items = pd.Series([0] * len(cache.index), index=cache.index)
    values = subgroup_target[column].value_counts()
    target = deepcopy(items)
    target[values.index] = values.values
    # return math.sqrt(len(subgroup_target)) * cosine(target.values, cache.values), target
    return entropy(subgroup_target, dataset_target) * cosine(target.values, cache.values), target


def WRAcc(subgroup_target, dataset_target, use_complement=False):
    global items, cache
    if len(subgroup_target.columns) > 1:
        raise ValueError("Distribution cosine expect exactly 1 column as target variable")
    column = list(subgroup_target.columns)[0]
    if cache is None:
        cache = dataset_target[column].value_counts()
        items = pd.Series([0] * len(cache.index), index=cache.index)
    values = subgroup_target[column].value_counts()
    target = deepcopy(items)
    target[values.index] = values.values
    max_Wc = target.values.max() + EPSILON
    max_W = cache.values.max() + EPSILON
    score = 0
    for Wce, We in zip(target.values, cache.values):
        score += (max_Wc / max_W) * ((Wce / max_Wc) - (We / max_W))
    return score * 1000, target


def avg(collection):
    try:
        return sum(collection) / len(collection)
    except ZeroDivisionError:
        return 0


def get_average_price_change(df, col_x):
    #avg_x = avg(df[col_x])
    #avg_y = avg(df[col_y])
    vec_len = len(list(df['Ticker_diff'].items())[0][1])
    vec1 = np.zeros(vec_len)
    for x, y in df['Ticker_diff'].items():
        vec1 = np.add(vec1, np.array(list(y)))
    #print(vec1 / len(df))
    return vec1 / len(df)
    #top = df.apply(lambda row: (row[col_x] - avg_x) * (row[col_y] - avg_y), axis=1)
    #bottom_x = df.apply(lambda row: (row[col_x] - avg_x) ** 2, axis=1)
    #bottom_y = df.apply(lambda row: (row[col_y] - avg_y) ** 2, axis=1)
    #try:
        #return top.sum() / math.sqrt(bottom_x.sum() * bottom_y.sum())
    #except Warning:  # Both x.sum() and y.sum() equal zero
        #return 0


def heatmap(subgroup_target, dataset_target, use_complement=False):
    global cache, items
    if len(subgroup_target.columns) != 2:
        raise ValueError("Correlation metric expects exactly 2 columns as target variables")
    x_col, y_col = list(subgroup_target.columns)

    if cache is None:
        cache = pd.pivot_table(dataset_target, values=x_col, index=x_col, fill_value=0,
                               columns=y_col, aggfunc=lambda x: len(x)).stack()
        items = pd.Series([0] * len(cache.index), index=cache.index)
    pv = pd.pivot_table(subgroup_target, values=x_col, index=x_col, fill_value=0,
                        columns=y_col, aggfunc=lambda x: len(x)).stack()
    target = deepcopy(items)
    target[pv.index] = pv.values
    return entropy(subgroup_target, dataset_target) * cosine(target.values, cache.values), target.unstack()


def correlation(subgroup_target, dataset_target, use_complement=False):
    """

    :param subgroup_target:
    :param dataset_target:
    :param use_complement:
    :return:
    """
    global cache
    print("In correlation func")
    #if len(subgroup_target.columns) != 2:
    #    raise ValueError("Correlation metric expects exactly 2 columns as target variables")
    #x_col, y_col = list(subgroup_target.columns)
    x_col = list(subgroup_target.columns)
    if cache is None:
        cache = get_average_price_change(dataset_target, x_col)
    # print(subgroup_target, x_col, y_col)
    r_gd = get_average_price_change(subgroup_target, x_col)
    corr_coeff = np.corrcoef(cache, r_gd)[0][1]
    print(f'Correlation: {corr_coeff}')
    #if math.isnan(r_gd):
        #return 0, 0
    entr = entropy(subgroup_target, dataset_target)
    reverse_coeff = len(dataset_target) / len(subgroup_target)
    print(f'Measure: {entr*corr_coeff}')
    return entr * abs(1 - corr_coeff), corr_coeff


def regression(subgroup_target, dataset_target, use_complement=False):
    global cache
    if len(subgroup_target) < 20:
        return 0, None
    if len(subgroup_target.columns) != 2:
        raise ValueError("Correlation metric expects exactly 2 columns as target variables")
    x_col, y_col = list(subgroup_target.columns)
    if cache is None:
        est2 = sm.OLS(dataset_target[y_col], dataset_target[x_col])
        est2 = est2.fit()
        cache = est2.summary2().tables[1]['Coef.'][x_col]
    est = sm.OLS(subgroup_target[y_col], subgroup_target[x_col])
    est = est.fit()
    coef = est.summary2().tables[1]['Coef.'][x_col]
    p = est.summary2().tables[1]['P>|t|'][x_col]
    if math.isnan(p):
        return 0, 0
    if (1 - p) < 0.99:
        return 0, 0
    return entropy(subgroup_target, dataset_target) * abs(coef - cache), coef


metrics = dict(
    correlation=correlation,
    distribution_cosine=distribution_cosine,
    regression=regression,
    WRAcc=WRAcc,
    heatmap=heatmap
)
