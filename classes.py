import json
import datetime
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from pandas import read_csv
import pandas_datareader as web

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
'''
CRT + SHIFT + A = ACTION MENU
'''


def loadTickers():
    """
    load all ticker from ./ticker.json
    """
    with open('./tickers.json', 'r') as f:
        tickers = json.load(f)
    return tickers


def fetchData():
    """
    FETCHES DATA PER TICKET OF LAST 100 DAYS
    """
    start = datetime.date.today() - datetime.timedelta(days=150)
    end = datetime.date.today() - datetime.timedelta(days=0)
    for ticker in loadTickers():
        dataframe = web.get_data_yahoo(ticker, start, end)

        dataframe.to_csv('./stock_dfs/{}.csv'.format(ticker))
        print('{}.csv has been downloaded.'.format(ticker))


def predictingData(ticker):
    # modifying data
    dataset = read_csv('./stock_dfs/{}.csv'.format(ticker))
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset.set_index('Date', inplace=True)
    dataset['Year'] = dataset.index.year
    dataset['Month'] = dataset.index.month
    dataset['Weekday Name'] = dataset.index.day_name()

    array = dataset.values

    y = array[:, 5]
    y = y.astype('int')
    X = array[:, 0:5]
    X = X.astype('int')

    # x0 -> y7
    # y0 - y7 word weggehaald
    # x-1 - x-7 = X_new

    y = y[7:]
    X_new = X[-7:]
    X = X[:-7]

    # print(len(X))
    # print(len(y))
    '''training model'''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1)
    #
    # model = Ridge()
    # model.fit(X_train, y_train)
    # print(model.score(X_test, y_test))

    '''saving/loading model'''
    filename = 'model.sav'
    #joblib.dump(model, filename)
    model = joblib.load(filename)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    last_date = dataset.index[-1]
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    # visualising prediction
    forecast_set = model.predict(X_new)
    dataset['Forecast'] = np.nan
    a = dataset['Forecast'][-1]
    dataset['Forecast'][-1] = dataset['Adj Close'][-1]
    a = dataset['Adj Close'][-1]
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        dataset.loc[next_date] = [
            np.nan for _ in range(len(dataset.columns) - 1)] + [i]
    plt.figure()
    dataset['Adj Close'][-15:].plot()
    dataset['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(ticker)
    plt.show()


def main():
    for ticker in loadTickers():
        predictingData(ticker)
