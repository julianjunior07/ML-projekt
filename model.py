import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from read_file import *

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

def model_func():
    #przerobienie listy na dataframe pandas
    dict = {'value': measurements}
    df = pd.DataFrame(dict)
    
    #podział danych na treningowe i testowe
    ratio = 0.85
    total_rows = df.shape[0]
    train_size = int(total_rows*ratio)
    
    train_data = df[0:train_size]
    test_data = df[train_size:]
    
    # print("train")
    # print(train_data)
    # print("test")
    # print(test_data)
    
    print(df.shape)
    print(train_data.shape)
    print(test_data.shape)
    
    
    #model
    regressor = RandomForestRegressor(max_depth=5, n_estimators=1000, random_state=123)
    steps = len(test_data)
    forecaster = ForecasterAutoreg(regressor = regressor,
                                   lags=1600
                                   )
    forecaster.fit(y=train_data['value'])
    predictions = forecaster.predict(steps=steps)
    
    
    #ploty, ploteczki
    fig, ax = plt.subplots(figsize=(15,5))
    train_data['value'].plot(ax=ax, label='Train Set')
    test_data['value'].plot(ax=ax, label='Test Set')
    predictions.plot(ax=ax, label='predictions')
    plt.xlabel('number')
    plt.ylabel('bitrate')
    ax.legend()
    plt.show()
    
    #blad
    error_mse = mean_squared_error(
        y_true= test_data['value'],
        y_pred= predictions
    )
    print(f"test error mse: {error_mse}")
    








