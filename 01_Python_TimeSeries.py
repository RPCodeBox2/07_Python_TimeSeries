# In[1] - Documentation
"""
Script - 01_Python_TimeSeries.py
Decription - Sample Timeseries graphs
Author - Rana Pratap
Date - 2021
Version - 1.0
https://www.datacamp.com/community/tutorials/time-series-analysis-tutorial
"""
print(__doc__)

# In[2] - Import packages and Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#% matplotlib inline
sns.set()

#pip install numpy

df = pd.read_csv('multiTimeline.csv', skiprows=1)
df.head()
df.info()
df.columns = ['month', 'diet', 'gym', 'finance']
df.head()

df.month = pd.to_datetime(df.month)
df.set_index('month', inplace=True)
df.head()
# In[3] - Timeseries view 1

df.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

# In[4] - Timeseries view 2
df[['diet']].plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

# In[5] - Timeseries view 3
diet = df[['diet']]
diet.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

# In[6] - Timeseries view 4
gym = df[['gym']]
gym.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

# In[7] - Timeseries view 5
df_rm = pd.concat([diet.rolling(12).mean(), gym.rolling(12).mean()], axis=1)
df_rm.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

# In[8] - Timeseries view 6
diet.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

# In[9] - Timeseries view 7
df.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

# In[10] - Timeseries view - #Periodicity and Autocorrelation
df.diff().corr()
pd.plotting.autocorrelation_plot(diet);

# In[] -
del(df)
