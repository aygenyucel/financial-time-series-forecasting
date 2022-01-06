import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import math

from numpy import array
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

#Load Data
company = 'TRY=X'

start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()      #(year, month, date)

data = web.DataReader(company, 'yahoo', start, end)

#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation="relu"))

print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error')  
model.fit(x_train, y_train, epochs=10, batch_size=64)

'''Test the Model  Accuracy on Existing Data'''

#Load Test Data
test_start = dt.datetime(2020,7,1)
test_end = dt.datetime.now()
tomorrow = test_end + dt.timedelta(days=1)

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0 )

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days: ].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Make Prediction on Test Data
x_test = []
y_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days: x, 0])
    y_test.append(model_inputs[x,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = np.array(y_test)

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)  

#Plot the Test Predictions
plt.figure(figsize=(12,5), dpi=100)
plt.plot(actual_prices, color = 'black', label = f"Actual {company} Price")
plt.plot(predicted_prices, color = 'red', label = f"Predicted {company} Price")
plt.title(f"USD / {company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

# Forecasting Next Day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Date: {tomorrow}: USD/{company} CLOSE Price Forecasting: {prediction}") # Stock

# MAPE(Mean Absolute Percentage Error) Expected Forecasting Accuracy SCORE
mape = np.mean(np.abs(predicted_prices - actual_prices)/np.abs(actual_prices)) # test_data['Close']
print('\nMAPE: %.3f' % float(mape), ', MAPE SCORE: %.2f' % ((mape)*100))
print('\nFORECASTING ACCURACY: %.2f' % ((1.00 - mape)*100), '% expected accurate in Forecasting the Test Set observations.\n' )

# calculate accuracy
y_test_dummies = pd.get_dummies(y_test).values
score = model.evaluate(x_test, y_test_dummies, batch_size=64)
LSTM_accuracy = score*100
print('LSTM MODEL PREDICTED Accuracy SCORE: %.2f' % (score*100), '(%)')

# calculate root mean squared error
testY = scaler.inverse_transform([y_test])
testScore = math.sqrt(mean_squared_error(testY[0], predicted_prices[:,0]))
print('LSTM MODEL PREDICTED RMSE: %.2f\n' % (testScore))

# Prediction Process Plotting
plt.figure(figsize=(12,5), dpi=100)
plt.plot(total_dataset, color = 'orange', label = f"Training USD / {company} Price DATA")
plt.plot(test_data['Close'], color = 'green', label = f"Predicted USD / {company} Price DATA")
plt.title(f"USD / {company} Prediction Process")
plt.xlabel('Time')
plt.ylabel(f"{company} Price")
plt.legend(loc='upper left', fontsize=8)
plt.show()

