#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import pylab
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from reliability.Fitters import Fit_Weibull_2P
from fitter import Fitter
from scipy.stats import lognorm
import statsmodels.api as sm
from scipy.stats import lognorm, kstest


# In[2]:


import glob
path = 'D:\R\Statistical-Analysis-and-Forecasting-of-Solar-Energy-main\Statistical-Analysis-and-Forecasting-of-Solar-Energy-main\Renewable Energy Data\Andhra Pradesh'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename,index_col=None, header=0,skiprows=2,usecols=[0,1,2,3,4,7])
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)


# In[3]:


frame['Date']=frame['Year'].astype(str) + frame['Month'].astype(str).str.zfill(2) + frame['Day'].astype(str).str.zfill(2) + frame['Hour'].astype(str).str.zfill(2)+frame['Minute'].astype(str)

frame['Date'] = pd.to_datetime(frame['Date'], format='%Y%m%d%H%M')


# In[4]:


import datetime as dt

df_all = frame[['Date','GHI']]
df_all.set_index('Date',inplace=True)
df_all = df_all.between_time('09:00','15:00')


# In[5]:


df_all.head()


# In[6]:


y = df_all['GHI']


# In[7]:


stats.kstest(y, 'norm')


# In[8]:


y


# In[9]:


y = y[y>0]


# In[10]:


stats.kstest(y, 'norm')


# In[11]:


y.replace([np.inf, -np.inf,0], 1, inplace=True)


# In[12]:


y


# In[13]:


stats.kstest(y, 'norm')


# In[14]:


sigma, loc, scale = lognorm.fit(y, floc=0)


# In[15]:


stat, p = kstest(y, 'lognorm', args=(sigma, 0, scale), alternative='two-sided')


# In[16]:


stat


# In[17]:


p


# In[18]:


params = stats.lognorm.fit(y)


# In[19]:


params


# In[20]:


stats.kstest(y, "lognorm", params)


# In[ ]:





# In[21]:


stats.probplot(y, dist="norm", plot=pylab)


# In[22]:


f = Fitter(y)


# In[23]:


f.fit()


# In[24]:


f.summary()


# In[25]:


ig, ax = plt.subplots(1, 2, figsize=(14, 4))
probplot = sm.ProbPlot(y, dist=lognorm, fit=True)
probplot.ppplot(line='45', ax=ax[0])
probplot.qqplot(line='45', ax=ax[1])
ax[0].set_title('P-P Plot')
ax[1].set_title('Q-Q Plot')
plt.show()


# In[28]:


stats.kstest(y, 'gamma', (15.5, 7))


# In[29]:


stats.kstest(y, 'norm')


# In[30]:


stats.kstest(y, "lognorm", params)


# In[31]:


stats.kstest(y, 'loggamma', (15.5, 7))


# In[32]:


stats.kstest(y, 'expon')


# In[33]:


args = stats.weibull_min.fit(y)
kstest(y, 'weibull_min', args=args, N=100000)


# In[ ]:




