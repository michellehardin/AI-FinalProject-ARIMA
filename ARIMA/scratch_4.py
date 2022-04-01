# AI Group - Michelle Hardin, Emma Wade, Austin Reed, Hunter, Pradeep
# Univariate Forecasting of COVID using ARIMA Model
# Assignment: Final Project for Artificial Intelligence

import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.simplefilter('ignore')

data = pd.read_csv("time_series_covid19_confirmed_US.csv", parse_dates=['date'], index_col='date')

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

sns.set(palette='Set1', style='darkgrid')
# Function for making a time series and plotting the rolled mean and standard

def roll_v2(tempTs, case='ConfirmedCases'):
    return (tempTs.rolling(window=4, center=False).mean().dropna())


def rollPlot_v2(tempTs, country, case='ConfirmedCases'):
    plt.figure(figsize=(16, 6))
    plt.plot(tempTs.rolling(window=7, center=False).mean().dropna(), label='Rolling Mean')
    plt.plot(tempTs[case])

    plt.legend()
    plt.title(case + ' distribution in %s with rolling mean and standard' % country)
    plt.xticks([])


def tempTimeSeries(country, case='ConfirmedCases'):
    ts = df.loc[(df['Country_Region'] == country)]
    ts = ts[['Date', case]]
    ts = ts.set_index('Date')
    ts.astype('int64')
    a = len(ts.loc[(ts[case] >= 10)])
    ts = ts[-a:]
    return ts


def createTimeSeries(country):
    tempTsCase = tempTimeSeries(country)
    tempTsFatalities = tempTimeSeries(country, 'Fatalities')
    timeSeriesCases = roll_v2(tempTsCase)
    timeSeriesFatalities = roll_v2(tempTsFatalities, 'Fatalities')

    rollPlot_v2(tempTsCase, country)
    rollPlot_v2(tempTsFatalities, country, 'Fatalities')
    return timeSeriesCases, timeSeriesFatalities

timeSeriesUSCases, timeSeriesUSFatalities = createTimeSeries('US')

# Usual Differencing
axes[0].plot(data[:], label='Original Series')
axes[0].plot(data[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)

def stationarity(ts):
    print('Results of Dickey-Fuller Test:')
    test = adfuller(ts, autolag='AIC')
    results = pd.Series(test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    # for i,val in test[4].items():
    # results['Critical Value (%s)'%i] = val
    print('P-Value: ' + str(results['p-value']))


def checkStationarity(case_ts, fatal_ts):
    tsC = case_ts['ConfirmedCases'].values
    stationarity(tsC)
    tsF = fatal_ts['Fatalities'].values
    stationarity(tsF)


checkStationarity(timeSeriesUSCases, timeSeriesUSFatalities)


def split(timeseries):
    # splitting 85%/15% because of little amount of data
    size = int(len(timeseries) * 0.85)
    train = timeseries[:size]
    test = timeseries[size:]
    return (train, test)


def plot_pred_vs_true(pred, test, country, case):
    f, ax = plt.subplots()
    plt.plot(pred, c='green', label='predictions')
    plt.plot(test, c='red', label='real values')
    plt.xticks(rotation=90)
    plt.legend()
    plt.title('True vs predicted values for ' + country + ' ' + case)


def forecast(model, timeseries):
    result = model.fit()
    result.plot_predict(start=int(len(timeseries) * 0.7), end=int(len(timeseries) * 1.2))
    return result


def plots_result(model, ts, test, country, case):
    result = forecast(model, ts)
    pred = result.forecast(steps=len(test))[0]
    # Plotting results
    plot_pred_vs_true(pred, test, country, case)


# Arima modeling for ts
def arima(timeseries, country, case):
    train, test = split(timeseries)

    p = d = q = range(0, 6)
    lowest_aic = 99999
    pdq = list(itertools.product(p, d, q))

    # Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(train, order=var)
            result = model.fit()

            if (result.aic <= lowest_aic):
                lowest_aic = result.aic
                best_parameters = var
        except:
            continue

    # Modeling
    model = ARIMA(train, order=best_parameters)

    plots_result(model, train, test, country, case)

    print('Best Parameters(p,d,q) for ' + country + ' ' + case + ': ' + str(best_parameters))
    return best_parameters


case_parameter_dict = {}
fatal_parameter_dict = {}

p1 = arima(timeSeriesUSCases,'US', 'ConfirmedCase')
p2 = arima(timeSeriesUSFatalities,'US', 'Fatalities')
case_parameter_dict['US'] = p1
fatal_parameter_dict['US'] = p2
print(case_parameter_dict)
print(fatal_parameter_dict)

#Confirmed Cases
df_case_parameter_dict = pd.DataFrame.from_dict(case_parameter_dict, orient='index',columns=['p', 'd', 'q'])
df_case_parameter_dict.head(10)

#Fatalities
df_fatal_parameter_dict = pd.DataFrame.from_dict(fatal_parameter_dict, orient='index',columns=['p', 'd', 'q'])
df_fatal_parameter_dict.head(10)

#Plotting Best Case
df_fatal_parameter_dict.plot(kind="bar",title="Best Parameter for Fatalities Time Series ARIMA Model")
df_case_parameter_dict.plot(kind="bar",title="Best Parameter for ConfirmedCases Time Series ARIMA Model")

