import pandas as pd
from pyentrp import entropy as ent
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.stats import boxcox
from Genetic_algorithm.GA import *
from sklearn.decomposition import PCA



def read_database():

    work_sheets = ['M3Year', 'M3Month', 'M3Quart', 'M3Other']
    df = {}
    for ws in work_sheets:
        df[ws] = pd.read_excel('Data/M3C.xls', sheetname=ws)

    return df, work_sheets


def read_data(sheet='M3Month'):

    df = pd.read_excel('Data/M3C.xls', sheet=None, header=None)
    data = df.values
    data = np.array([exclude_nans(serie) for serie in data])

    return data


def exclude_nans(serie):

    serie = [s for s in serie if not np.isnan(s)]

    return serie


def calc_entropy(serie):

    shannon_entropy = ent.shannon_entropy(serie)

    return shannon_entropy


def calc_strength_trend(serie, freq = 1):


    result = seasonal_decompose(serie, freq=freq, model='additive')

    a = serie - result.seasonal

    if np.isnan(a).any():
        a = exclude_nans(a)

    var = np.var(a)
    if var == 0:
        return 0
    else:
        result = 1 - (np.var(result.resid) / var)
        if result == 0 or np.isnan(result):
            return 0
        else:
            return result


def calc_strength_seasonality(serie, freq = 1):

    result = seasonal_decompose(serie, freq=freq, model='additive')

    a = serie - result.trend
    if np.isnan(a).any():
        a = exclude_nans(a)

    var = np.var(a)
    if var == 0:
        return 0
    else:
        result = 1 - (np.var(result.resid) / var)
        if result == 0 or np.isnan(result):
            return 0
        else:
            return result


def calc_first_order_correlation(serie):

    aacf, confint, pvalue = acf(serie, nlags=1, qstat=True, unbiased=True)

    return aacf[1]


def calc_box_cox_optmization_parameter(serie):
    lambd = 0
    try:
        _, lambd1 = boxcox(serie)
        lambd = lambd1
    except:
        return lambd
    return lambd


def extract_features(serie, freq):
    spectral = calc_entropy(serie)
    trend = calc_strength_trend(serie, freq=freq)
    season = calc_strength_seasonality(serie, freq=freq)
    period = freq
    corr = calc_first_order_correlation(serie)
    lamb = calc_box_cox_optmization_parameter(serie)

    return np.array([spectral, trend, season, period, corr, lamb])


def extract_database_features():
    pca = PCA(n_components=6)
    data, sheets = read_database()
    freqs = [1, 12, 4, 1]
    lentao = []
    a = [3098, 4923, 4322, 5254, 4913, 2997, 3136, 5228, 5031, 5943, 4063, 3506, 5798, 4600, 4097, 5264, 5712, 5067, 5555, 5840]

    for sheet, freq in zip(sheets, freqs):
        df = data[sheet]
        df = df.values
        df = [exclude_nans(serie) for serie in df]
        for serie in df:
            try:
                lentao.append(extract_features(serie, freq))
            except:
                print serie

    lentao.append(extract_features(a, 1))

    lentao = np.array(lentao)

    pca.fit(lentao)
    feature_pca = pca.transform(lentao)

    new_series = []
    new_features = []

    for sheet, freq in zip(sheets, freqs):
        df = data[sheet]
        df = df.values
        df = [exclude_nans(serie) for serie in df]
        for serie in df:

            nova_serie = generate_time_serie(serie, freq)
            new_series.append(nova_serie)
            new_features.append(pca.transform(extract_features(nova_serie, freq)))
            #count += 1

    plt.scatter(feature_pca[:, 4], feature_pca[:, 0])



    file = open('generated.txt', 'w')
    for item in new_features:
        file.write(item[0])
        plt.scatter(item[:, 4], item[:, 0], color='r')
    plt.show()


if __name__ == '__main__':
    extract_database_features()
