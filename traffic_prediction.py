df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set DateTime as the index
df.set_index('DateTime', inplace=True)

# Check the data structure
print(df.head())
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path =  'C:/Users/user/Downloads/archive.zip'
df = pd.read_csv(file_path)

# Convert the DateTime column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set DateTime as the index
df.set_index('DateTime', inplace=True)

# Select data for a specific junction
junction = 1
df_junction = df[df['Junction'] == junction]['Vehicles']

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
df_junction_scaled = scaler.fit_transform(df_junction.values.reshape(-1, 1))

# Split data into training and testing sets
train_size = int(len(df_junction_scaled) * 0.8)
train, test = df_junction_scaled[:train_size], df_junction_scaled[train_size:]

# Create dataset function
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 24
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=3)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Create a new dataframe to align the predictions with the dates
train_predict_plot = np.empty_like(df_junction_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

test_predict_plot = np.empty_like(df_junction_scaled)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(df_junction_scaled) - 1, :] = test_predict

# Plot LSTM forecast
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(df_junction_scaled), label='Actual Data')
plt.plot(train_predict_plot, label='LSTM Training Forecast')
plt.plot(test_predict_plot, label='LSTM Test Forecast')
plt.title('LSTM Forecast for Junction 1')
plt.xlabel('DateTime')
plt.ylabel('Number of Vehicles')
plt.legend()
plt.show()

# Performance evaluation
train_mse = mean_squared_error(y_train[0], train_predict)
train_mae = mean_absolute_error(y_train[0], train_predict)
test_mse = mean_squared_error(y_test[0], test_predict)
test_mae = mean_absolute_error(y_test[0], test_predict)
train_r2 = r2_score(y_train[0], train_predict)
test_r2 = r2_score(y_test[0], test_predict)
train_mape = mean_absolute_percentage_error(y_train[0], train_predict)
test_mape = mean_absolute_percentage_error(y_test[0], test_predict)

print(f'Train MSE: {train_mse}')
print(f'Train MAE: {train_mae}')
print(f'Train R²: {train_r2}')
print(f'Train MAPE: {train_mape}')
print(f'Test MSE: {test_mse}')
print(f'Test MAE: {test_mae}')
print(f'Test R²: {test_r2}')
print(f'Test MAPE: {test_mape}')
y_test_persistence = df_junction.values[train_size + look_back:-1]
y_pred_persistence = df_junction.values[train_size + look_back - 1:-2]

# Calculate MSE and MAE for the persistence model
persistence_mse = mean_squared_error(y_test_persistence, y_pred_persistence)
persistence_mae = mean_absolute_error(y_test_persistence, y_pred_persistence)

print(f'Persistence Model MSE: {persistence_mse}')
print(f'Persistence Model MAE: {persistence_mae}')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/kaggle/input/traffic-prediction-dataset/traffic.csv'
df = pd.read_csv(file_path)

# Convert the DateTime column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set DateTime as the index
df.set_index('DateTime', inplace=True)

# Select data for a specific junction
junction = 1
df_junction = df[df['Junction'] == junction]['Vehicles']

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
df_junction_scaled = scaler.fit_transform(df_junction.values.reshape(-1, 1))

# Split data into training and testing sets
train_size = int(len(df_junction_scaled) * 0.8)
train, test = df_junction_scaled[:train_size], df_junction_scaled[train_size:]

# Create dataset function
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 24
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)
# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model with dropout
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with more epochs
model.fit(X_train, y_train, batch_size=1, epochs=100)
# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Create a new dataframe to align the predictions with the dates
train_predict_plot = np.empty_like(df_junction_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

test_predict_plot = np.empty_like(df_junction_scaled)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2):len(train_predict) + (look_back * 2) + len(test_predict), :] = test_predict

# Plot LSTM forecast
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(df_junction_scaled), label='Actual Data')
plt.plot(train_predict_plot, label='LSTM Training Forecast')
plt.plot(test_predict_plot, label='LSTM Test Forecast')
plt.title('LSTM Forecast for Junction 1')
plt.xlabel('DateTime')
plt.ylabel('Number of Vehicles')
plt.legend()
plt.show()

# Performance evaluation
train_mse = mean_squared_error(y_train, train_predict)
train_mae = mean_absolute_error(y_train, train_predict)
test_mse = mean_squared_error(y_test, test_predict)
test_mae = mean_absolute_error(y_test, test_predict)

print(f'Train MSE with dropout: {train_mse}')
print(f'Train MAE with dropout: {train_mae}')
print(f'Test MSE with dropout: {test_mse}')
print(f'Test MAE with dropout: {test_mae}')
# Persistence model (predict the previous hour's traffic volume)
y_test_persistence = df_junction.values[train_size + look_back:-1]
y_pred_persistence = df_junction.values[train_size + look_back - 1:-2]

# Calculate MSE and MAE for the persistence model
persistence_mse = mean_squared_error(y_test_persistence, y_pred_persistence)
persistence_mae = mean_absolute_error(y_test_persistence, y_pred_persistence)

print(f'Persistence Model MSE: {persistence_mse}')
print(f'Persistence Model MAE: {persistence_mae}')import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# Load the dataset
file_path = '/kaggle/input/traffic-prediction-dataset/traffic.csv'
df = pd.read_csv(file_path)

# Convert the DateTime column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
# Set DateTime as the index
df.set_index('DateTime', inplace=True)

# Select data for a specific junction
junction = 1
df_junction = df[df['Junction'] == junction]['Vehicles']

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
df_junction_scaled = scaler.fit_transform(df_junction.values.reshape(-1, 1))

# Split data into training and testing sets
train_size = int(len(df_junction_scaled) * 0.8)
train, test = df_junction_scaled[:train_size], df_junction_scaled[train_size:]

# Create dataset function
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 24
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Build the LSTM model with dropout and early stopping
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25, kernel_regularizer='l2'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the model with early stopping
history = model.fit(X_train, y_train, batch_size=1, epochs=100, validation_split=0.2, callbacks=[early_stop])

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
# Create a new dataframe to align the predictions with the dates
train_predict_plot = np.empty_like(df_junction_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

test_predict_plot = np.empty_like(df_junction_scaled)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2):len(train_predict) + (look_back * 2) + len(test_predict), :] = test_predict

# Plot LSTM forecast
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(df_junction_scaled), label='Actual Data')
plt.plot(train_predict_plot, label='LSTM Training Forecast')
plt.plot(test_predict_plot, label='LSTM Test Forecast')
plt.title('LSTM Forecast for Junction 1')
plt.xlabel('DateTime')
plt.ylabel('Number of Vehicles')
plt.legend()
plt.show()

# Performance evaluation
train_mse = mean_squared_error(y_train, train_predict)
train_mae = mean_absolute_error(y_train, train_predict)
test_mse = mean_squared_error(y_test, test_predict)
test_mae = mean_absolute_error(y_test, test_predict)

print(f'Train MSE with dropout and early stopping: {train_mse}')
print(f'Train MAE with dropout and early stopping: {train_mae}')
print(f'Test MSE with dropout and early stopping: {test_mse}')
print(f'Test MAE with dropout and early stopping: {test_mae}')