import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_loader
from sklearn.preprocessing import MinMaxScaler

#param à améliorer :
#quel lookback, combien d'epochs, batch size
#avec  ou sans resampler ?

#est ce qu'on resample pour que ça tourne plus vite + on n'a pas besoin de prévoir aux 5 min près !
#resampler toutes les heures


filename='csv_files/Data_hydro.csv'
df = pd.read_csv(filename, skiprows=18, delimiter=';')
print(df)

df.set_index('Date/time hiver', inplace=True)
df.index = pd.to_datetime(df.index)



## Preparation of the data
# Data cleansing, scaling (careful with data leakage), divide into train and test
#detrend the data ?

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
df_wi_outliers=df

median = df_wi_outliers.loc[df_wi_outliers['Heau(m_sw)']>-0.6, 'Heau(m_sw)'].median()
df_wi_outliers.loc[df_wi_outliers['Heau(m_sw)'] < -0.60, 'Heau(m_sw)'] = np.nan
df_wi_outliers['Heau(m_sw)'].fillna(median,inplace=True)

median = df_wi_outliers.loc[df_wi_outliers['Pair(m_sw)']<11.26, 'Pair(m_sw)'].median()
df_wi_outliers.loc[df_wi_outliers['Pair(m_sw)'] > 11.26, 'Pair(m_sw)'] = np.nan
df_wi_outliers['Pair(m_sw)'].fillna(median,inplace=True)

median = df_wi_outliers.loc[df_wi_outliers['TairS1(°C)']<23.15, 'TairS1(°C)'].median()
df_wi_outliers.loc[df_wi_outliers['TairS1(°C)'] > 23.15, 'TairS1(°C)'] = np.nan
df_wi_outliers['TairS1(°C)'].fillna(median,inplace=True)

median = df_wi_outliers.loc[df_wi_outliers['Hmer_PM(m_sw)']<0.8, 'Hmer_PM(m_sw)'].median()
df_wi_outliers.loc[df_wi_outliers['Hmer_PM(m_sw)'] > 0.8, 'Hmer_PM(m_sw)'] = np.nan
df_wi_outliers.loc[df_wi_outliers['Hmer_PM(m_sw)'] < 0.235, 'Hmer_PM(m_sw)'] = np.nan
df_wi_outliers['Hmer_PM(m_sw)'].fillna(median,inplace=True)

median = df_wi_outliers.loc[(df_wi_outliers['Patm(m_sw)']<10.27) & (df_wi_outliers['Patm(m_sw)']>9.94) , 'Patm(m_sw)'].median()
df_wi_outliers.loc[df_wi_outliers['Patm(m_sw)'] > 10.27, 'Patm(m_sw)'] = np.nan
df_wi_outliers.loc[df_wi_outliers['Patm(m_sw)'] < 9.94, 'Patm(m_sw)'] = np.nan
df_wi_outliers['Patm(m_sw)'].fillna(median,inplace=True)

median = df_wi_outliers.loc[df_wi_outliers['Nair(mol)filt']<267125 , 'Patm(m_sw)'].median()
df_wi_outliers.loc[df_wi_outliers['Nair(mol)filt'] > 267125, 'Nair(mol)filt'] = np.nan
df_wi_outliers['Nair(mol)filt'].fillna(median,inplace=True)

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
for i, col in enumerate(columns):
    row = i // 3
    col_pos = i % 3
    axs[row, col_pos].boxplot(df_wi_outliers[col])
    axs[row, col_pos].set_title(col)
    axs[row, col_pos].set_ylabel('Values')
    axs[row, col_pos].grid(True)
plt.show()


#dataset with ouliers but without nan
#replace nan by the median value

median = df['Heau(m_sw)'].median()
df['Heau(m_sw)'].fillna(median,inplace=True)

median = df['Pair(m_sw)'].median()
df['Pair(m_sw)'].fillna(median,inplace=True)

median = df['TairS1(°C)'].median()
df['TairS1(°C)'].fillna(median,inplace=True)

median = df['Hmer_PM(m_sw)'].median()
df['Hmer_PM(m_sw)'].fillna(median,inplace=True)

median = df['Patm(m_sw)'].median()
df['Patm(m_sw)'].fillna(median,inplace=True)

median = df['Nair(mol)filt'].median()
df['Nair(mol)filt'].fillna(median,inplace=True)


#create the dataset

def create_dataset(dataset_feature, dataset_target, lookback):
    """Transform a time series into a prediction dataset
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset_feature)-lookback):
        feature = dataset_feature[i:i+lookback]
        target = dataset_target[i+lookback]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


#variables extérieures
column=['Pair(m_sw)','TairS1(°C)','Hmer_PM(m_sw)','Patm(m_sw)']
df_wi_outliers_target = df_wi_outliers[['Heau(m_sw)']].values.astype('float32')
df_target = df[['Heau(m_sw)']].values.astype('float32')
df_wi_outliers_feature = df_wi_outliers[column].values.astype('float32')
df_feature = df[column].values.astype('float32')
print('df defined')

# train-test split for time series
#X and y for both with and without outliers to compare the performance
train_size = int(len(df_wi_outliers) * 0.8)
test_size = len(df_wi_outliers) - train_size
#train_wi_outliers, test_wi_outliers = df_wi_outliers[:train_size], df_wi_outliers[train_size:]
train_feature, test_feature = df_feature[:train_size], df_feature[train_size:]
train_target, test_target = df_target[:train_size], df_target[train_size:]
print('train and test defined')

#trouver le bon lookback
#si considère les 3 derniers mois : 90 jours soit 25920 créneaux de 5min. Pas assez de mémoire
#on va prendre 1 mois seulement
lookback = 5
#X_train_wi_outliers, y_train_wi_outliers = create_dataset(train_wi_outliers, lookback=lookback)
#X_test_wi_outliers, y_test_wi_outliers = create_dataset(test_wi_outliers, lookback=lookback)
X_train, y_train = create_dataset(train_feature,train_target, lookback=lookback)
print("train dataset ok")
X_test, y_test = create_dataset(test_feature,test_target, lookback=lookback)
print('test dataset ok')
#print(X_train_wi_outliers.shape, y_train_wi_outliers.shape)
#print(X_test_wi_outliers.shape, y_test_wi_outliers.shape)
print(X_train)
print(y_train)

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))

X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(-1, lookback, len(column))
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(-1, lookback, len(column))
#X_train_wi_outliers = scaler.fit_transform(X_train_wi_outliers.reshape(-1, 1)).reshape(-1, lookback, 1)
#X_test_wi_outliers = scaler.transform(X_test_wi_outliers.reshape(-1, 1)).reshape(-1, lookback, 1)
#y_train_wi_outliers = scaler.fit_transform(y_train_wi_outliers.reshape(-1, 1))
#y_test_wi_outliers = scaler.transform(y_test_wi_outliers.reshape(-1, 1))
print('scaling done')

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
#X_train_wi_outliers = torch.tensor(X_train_wi_outliers, dtype=torch.float32)
#X_test_wi_outliers = torch.tensor(X_test_wi_outliers, dtype=torch.float32)
#y_train_wi_outliers = torch.tensor(y_train_wi_outliers, dtype=torch.float32)
#y_test_wi_outliers = torch.tensor(y_test_wi_outliers, dtype=torch.float32)


#LSTM
#can create a LSTM with more features by changing "column" above

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=len(column), hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])  # Récupérer seulement la dernière sortie de la séquence
        return x
    

#with outliers :

model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
#see which batch size is best
loader = data_loader.DataLoader(data_loader.TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                                          torch.tensor(y_train, dtype=torch.float32)), 
                                shuffle=True, batch_size=10000)

n_epochs = 1
print('entrée dans le nn')
for epoch in range(n_epochs):
    model.train()
    print('model trained')
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 5 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

#plotting results    
with torch.no_grad():
    train_plot = np.ones_like(df_target) * np.nan
    test_plot = np.ones_like(df_target) * np.nan
    y_train_pred = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
    y_test_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    train_plot[lookback:train_size] = y_train_pred
    test_plot[train_size+lookback:] = y_test_pred
print(train_plot,test_plot)

plt.plot(df_target, label='True Data')
plt.plot(train_plot, label='Train Predictions', color='r')
plt.plot(test_plot, label='Test Predictions', color='g')
plt.legend()
plt.show() 


#without outliers

model_wi_outliers = AirModel()
optimizer = optim.Adam(model_wi_outliers.parameters())
loss_fn = nn.MSELoss()
loader = data_loader.DataLoader(data_loader.TensorDataset(torch.tensor(X_train_wi_outliers, dtype=torch.float32), 
                                                          torch.tensor(y_train_wi_outliers, dtype=torch.float32)), 
                                shuffle=True, batch_size=8)
n_epochs = 200
for epoch in range(n_epochs):
    model_wi_outliers.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model_wi_outliers.eval()
    with torch.no_grad():
        y_pred = model_wi_outliers(X_train_wi_outliers)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train_wi_outliers))
        y_pred = model_wi_outliers(X_test_wi_outliers)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test_wi_outliers))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    
with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(df_wi_outliers) * np.nan
    y_pred = model_wi_outliers(X_train_wi_outliers)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model_wi_outliers(X_train_wi_outliers)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(df_wi_outliers) * np.nan
    test_plot[train_size+lookback:len(df_wi_outliers)] = model_wi_outliers(X_test_wi_outliers)[:, -1, :]
# plot
plt.plot(df_wi_outliers)
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()    
  

#LSTM inspired by https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/