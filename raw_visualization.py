import pandas as pd
import matplotlib.pyplot as plt

filename='csv_files/Data_hydro.csv'

df = pd.read_csv(filename, skiprows=18, delimiter=';')
print(df)

#Type of each attribute
types = df.dtypes
print(types)


#plot Heau
#takes some timebecause havy data
df.set_index('Date/time hiver', inplace=True)
df.index = pd.to_datetime(df.index)

# Resample the data to 2-hour intervals
mseries = df['Heau(m_sw)'].resample('2H').mean()

# Create a new DataFrame from the resampled series
df1 = pd.DataFrame({'Date/time hiver': mseries.index, 'Heau(m_sw)': mseries.values})

# Plotting the resampled data
plt.figure(figsize=(24, 6))
plt.plot(df1['Date/time hiver'], df1['Heau(m_sw)'], label="Niveau d'eau")
plt.title("Niveau d'eau dans la grotte", fontsize=20)
plt.xlabel('Date', fontsize=10)
plt.ylabel("Niveau d'eau [m_sw]", fontsize=10)
plt.legend()
plt.show()

print('ok')