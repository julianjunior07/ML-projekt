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

from sklearn.preprocessing import MinMaxScaler
import torch # Library for implementing Deep Neural Network 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train = scaler.fit_transform(train_data)
    
    print(scaled_train[:train_size])
    
    scaled_test = scaler.fit_transform(test_data)
    print(*scaled_test[:5]) #prints the first 5 rows of scaled_test
    
    # #model
    # regressor = RandomForestRegressor(max_depth=5, n_estimators=1000, random_state=123)
    # steps = len(test_data)
    # forecaster = ForecasterAutoreg(regressor = regressor,
    #                                lags=1600
    #                                )
    # forecaster.fit(y=train_data['value'])
    # predictions = forecaster.predict(steps=steps)
    
    # Create sequences and labels for training data
    sequence_length = 1000  # Number of time steps to look back
    X_train, y_train = [], []
    for i in range(len(scaled_train) - sequence_length):
        X_train.append(scaled_train[i:i+sequence_length])
        y_train.append(scaled_train[i+1:i+sequence_length+1])
    X_train, y_train = np.array(X_train), np.array(y_train)
 
    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    print(X_train.shape,y_train.shape)
    
    
    # Create sequences and labels for testing data
    sequence_length = 200  # Number of time steps to look back
    X_test, y_test = [], []
    for i in range(len(scaled_test) - sequence_length):
        X_test.append(scaled_test[i:i+sequence_length])
        y_test.append(scaled_test[i+1:i+sequence_length+1])
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Convert data to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    X_test.shape, y_test.shape

    
    
    class LSTMModel(nn.Module):
        # input_size : number of features in input at each time step
        # hidden_size : Number of LSTM units 
        # num_layers : number of LSTM layers 
        def __init__(self, input_size, hidden_size, num_layers): 
            super(LSTMModel, self).__init__() #initializes the parent class nn.Module
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)
    
        def forward(self, x): # defines forward pass of the neural network
            out, _ = self.lstm(x)
            out = self.linear(out)
            return out

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    
    input_size = 1
    num_layers = 2
    hidden_size = 64
    output_size = 1
    
    # Define the model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    
    loss_fn = torch.nn.MSELoss(reduction='mean')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    
    batch_size = 16
    # Create DataLoader for batch training
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create DataLoader for batch training
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_epochs = 10
    train_hist =[]
    test_hist =[]
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average training loss and accuracy
        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)

        # Validation on test data
        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0

            for batch_X_test, batch_y_test in test_loader:
                batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                predictions_test = model(batch_X_test)
                test_loss = loss_fn(predictions_test, batch_y_test)

                total_test_loss += test_loss.item()

            # Calculate average test loss and accuracy
            average_test_loss = total_test_loss / len(test_loader)
            test_hist.append(average_test_loss)
        if (epoch+1)%10==0:
            print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')


    # x = np.linspace(1,num_epochs,num_epochs)
    # plt.plot(x,train_hist,scalex=True, label="Training loss")
    # plt.plot(x, test_hist, label="Test loss")
    # plt.legend()
    # plt.show()

        # Define the number of future time steps to forecast
    num_forecast_steps = len(test_data)

    # Convert to NumPy and remove singleton dimensions
    sequence_to_plot = X_test.squeeze().cpu().numpy()

    # Use the last 30 data points as the starting point
    historical_data = sequence_to_plot[-1]
    print(historical_data.shape)

    # Initialize a list to store the forecasted values
    forecasted_values = []

    # Use the trained model to forecast future values
    with torch.no_grad():
        for _ in range(num_forecast_steps):
            # Prepare the historical_data tensor
            historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)
            # Use the model to predict the next value
            predicted_value = model(historical_data_tensor).cpu().numpy()[0, 0]

            # Append the predicted value to the forecasted_values list
            forecasted_values.append(predicted_value[0])

            # Update the historical_data sequence by removing the oldest value and adding the predicted value
            historical_data = np.roll(historical_data, shift=-1)
            historical_data[-1] = predicted_value

            
    # # Generate futute dates
    # last_date = test_data.index[-1]

    # # Generate the next 30 dates
    # # future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=len(test_data))

    # # Concatenate the original index with the future dates
    # combined_index = test_data.index.append(test_data.index[-1])


        
    # #set the size of the plot 
    # plt.rcParams['figure.figsize'] = [14, 4] 
    
    
    # #Test data
    # plt.plot(test_data.index[-100:-30], test_data.Open[-100:-30], label = "test_data", color = "b") 
    # #reverse the scaling transformation
    # original_cases = scaler.inverse_transform(np.expand_dims(sequence_to_plot[-1], axis=0)).flatten() 
    
    # #the historical data used as input for forecasting
    # plt.plot(test_data.index[-30:], original_cases, label='actual values', color='green') 
    
    # #Forecasted Values 
    # #reverse the scaling transformation
    forecasted_cases = scaler.inverse_transform(np.expand_dims(forecasted_values, axis=0)).flatten() 
    # # plotting the forecasted values
    # plt.plot(combined_index[-60:], forecasted_cases, label='forecasted values', color='red')  
    
    # plt.xlabel('Time Step')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.title('Time Series Forecasting')
    # plt.grid(True)

    
    #ploty, ploteczki
    fig, ax = plt.subplots(figsize=(15,5))
    # train_data['value'].plot(ax=ax, label='Train Set')
    # test_data['value'].plot(ax=ax, label='Test Set')
    plt.plot(train_data)
    plt.plot(test_data)
    #predictions.plot(ax=ax, label='predictions')
    plt.plot(test_data.index, forecasted_cases)
    #forecasted_cases.plot(ax=ax, label='Prediction Set')
    plt.plot([])
    plt.xlabel('number')
    plt.ylabel('bitrate')
    ax.legend()
    plt.show()
    
    #blad
    error_mse = mean_squared_error(
        y_true= test_data['value'],
        y_pred= forecasted_cases
    )
    print(f"test error mse: {error_mse}")
    








