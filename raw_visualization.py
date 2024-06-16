import pandas as pd
import matplotlib.pyplot as plt

filename='csv_files/Data_hydro.csv'

df = pd.read_csv(filename, skiprows=18, delimiter=';')
print(df)

#Type of each attribute
types = df.dtypes
print(types)


#plot Heau
#takes some time because havy data
df.set_index('Date/time hiver', inplace=True)
df.index = pd.to_datetime(df.index)

# Resample the data to 2-hour intervals
resampled_df = df[['Heau(m_sw)','Pair(m_sw)']].resample('2H').mean()

fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# First plot (Heau)
axs[0].plot(resampled_df.index, resampled_df['Heau(m_sw)'], label='Heau(m_sw)', color='b')
axs[0].set_ylabel('Heau(m_sw)')
axs[0].set_title("Hauteur d'eau dans la grotte [m]")
axs[0].legend()
axs[0].grid(True)

# Second plot (Pair)
axs[1].plot(resampled_df.index, resampled_df['Pair(m_sw)'], label='Pair(m_sw)', color='r')
axs[1].set_xlabel('Date/time hiver')
axs[1].set_ylabel('Pair(m_sw)')
axs[1].set_title("Pression de l'air dans la grotte [m]")
axs[1].legend()
axs[1].grid(True)

plt.show()


## Number of missing data

nb_nan_Heau = df['Heau(m_sw)'].isna().sum()
print(nb_nan_Heau) #24004
nb_nan_Pair = df['Pair(m_sw)'].isna().sum()
print(nb_nan_Pair) #0