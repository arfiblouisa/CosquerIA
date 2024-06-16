import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler



filename='csv_files/Data_hydro.csv'
df = pd.read_csv(filename, skiprows=18, delimiter=';')
print(df)

df.set_index('Date/time hiver', inplace=True)
df.index = pd.to_datetime(df.index)



## Preparation of the data
# Data cleansing, scaling (careful with data leakage), divide into train and test

#look for outliers
columns = ['Heau(m_sw)', 'Pair(m_sw)', 'TairS1(°C)', 'Hmer_PM(m_sw)', 'Patm(m_sw)', 'Nair(mol)filt']
df = df[columns]

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
for i, col in enumerate(columns):
    row = i // 3
    col_pos = i % 3
    axs[row, col_pos].boxplot(df[col].dropna())
    axs[row, col_pos].set_title(col)
    axs[row, col_pos].set_ylabel('Values')
    axs[row, col_pos].grid(True)
plt.show()
#A lot of outliers for all

#remove outliers and replace by the median value
#Replace also nan by the median value
median = df.loc[df['Heau(m_sw)']>-0.6, 'Heau(m_sw)'].median()
df.loc[df['Heau(m_sw)'] < -0.60, 'Heau(m_sw)'] = np.nan
df['Heau(m_sw)'].fillna(median,inplace=True)

median = df.loc[df['Pair(m_sw)']<11.26, 'Pair(m_sw)'].median()
df.loc[df['Pair(m_sw)'] > 11.26, 'Pair(m_sw)'] = np.nan
df['Pair(m_sw)'].fillna(median,inplace=True)

median = df.loc[df['TairS1(°C)']<23.15, 'TairS1(°C)'].median()
df.loc[df['TairS1(°C)'] > 23.15, 'TairS1(°C)'] = np.nan
df['TairS1(°C)'].fillna(median,inplace=True)

median = df.loc[df['Hmer_PM(m_sw)']<0.8, 'Hmer_PM(m_sw)'].median()
df.loc[df['Hmer_PM(m_sw)'] > 0.8, 'Hmer_PM(m_sw)'] = np.nan
df.loc[df['Hmer_PM(m_sw)'] < 0.235, 'Hmer_PM(m_sw)'] = np.nan
df['Hmer_PM(m_sw)'].fillna(median,inplace=True)

median = df.loc[(df['Patm(m_sw)']<10.27) & (df['Patm(m_sw)']>9.94) , 'Patm(m_sw)'].median()
df.loc[df['Patm(m_sw)'] > 10.27, 'Patm(m_sw)'] = np.nan
df.loc[df['Patm(m_sw)'] < 9.94, 'Patm(m_sw)'] = np.nan
df['Patm(m_sw)'].fillna(median,inplace=True)

median = df.loc[df['Nair(mol)filt']<267125 , 'Patm(m_sw)'].median()
df.loc[df['Nair(mol)filt'] > 267125, 'Nair(mol)filt'] = np.nan
df['Nair(mol)filt'].fillna(median,inplace=True)

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
for i, col in enumerate(columns):
    row = i // 3
    col_pos = i % 3
    axs[row, col_pos].boxplot(df[col])
    axs[row, col_pos].set_title(col)
    axs[row, col_pos].set_ylabel('Values')
    axs[row, col_pos].grid(True)
plt.show()


#create the dataset

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


column='Heau(m_sw)'
df = df[column]

# train-test split for time series
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df[:train_size], df[train_size:]

lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test=scaler.trasnform(X_test)


#LSTM
#can create a LSTM with more features by changing "column" above

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    


model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    
with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
# plot
plt.plot(timeseries)
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()    


#LSTM inspired by https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/