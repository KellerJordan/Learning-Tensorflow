import pandas as pd
import numpy as np

df = pd.io.parsers.read_csv(
    'https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
     header=None,
     usecols=[0,1,2]
)

df.columns = ['Class Label', 'Alcohol', 'Malic acid']


from sklearn.preprocessing import StandardScaler, MinMaxScaler

df_std = StandardScaler().fit_transform(df[['Alcohol', 'Malic acid']])
df_minmax = MinMaxScaler().fit_transform(df[['Alcohol', 'Malic acid']])

print('Mean after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}\n'
      .format(df_std[:,0].mean(), df_std[:,1].mean()))
print('Standard deviation after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}\n'
      .format(df_std[:,0].std(), df_std[:,1].std()))
print('Mean after minmax:\nAlcohol={:.2f}, Malic acid={:.2f}\n'
      .format(df_minmax[:,0].mean(), df_minmax[:,1].mean()))
print('Standard deviation after minmax:\nAlcohol={:.2f}, Malic acid={:.2f}\n'
      .format(df_minmax[:,0].std(), df_minmax[:,1].std()))


from matplotlib import pyplot as plt

plt.figure(figsize=(8,6))
plt.rc('text', usetex=True)

plt.scatter(df['Alcohol'], df['Malic acid'],
    color='green', label='input scale', alpha=0.5)

plt.scatter(df_std[:,0], df_std[:,1],
    color='red', label='Standardized $N  (\mu=0, \; \sigma=1)$', alpha=0.3)

plt.scatter(df_minmax[:,0], df_minmax[:,1],
    color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)

plt.title('Alcohol and Malic Acid content of the wine dataset')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.legend(loc='upper left')
plt.grid()

plt.tight_layout()
plt.show()

