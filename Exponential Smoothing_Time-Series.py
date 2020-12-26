#!/usr/bin/env python
# coding: utf-8

# # Exponential smoothing
# 
# 

# In[5]:


#Importing Libraries
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
import seaborn as sns

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.filters.hp_filter import hpfilter
plt.style.use('ggplot')


# In[1]:


ticker_details = pd.read_excel("Ticker List.xlsx")
ticker_details.head(20)


# In[2]:


ticker = ticker_details['Ticker'].to_list()
names = ticker_details['Description'].to_list()


# In[3]:


#Extracting Data from Yahoo Finance and Adding them to Values table using date as key
end_date= "2020-09-30"
start_date = "2010-01-01"
date_range = pd.bdate_range(start=start_date,end=end_date)
"""
Create a date-range and write it to an empty dataframe named values. Here I would extract and 
past the values pulled from yahoofinancials.
"""
df = pd.DataFrame({ 'Date': date_range})
df['Date']= pd.to_datetime(df['Date'])


# In[7]:


from pandas import DataFrame 
#Extracting Data from Yahoo Finance and Adding them to Values table using date as key
for i in ticker:
    raw_data = YahooFinancials(i)
    raw_data = raw_data.get_historical_price_data(start_date, end_date, "daily")
    data = DataFrame(raw_data[i]['prices'])[['formatted_date','adjclose']]
    data.columns = ['Date1',i]
    data['Date1']= pd.to_datetime(data['Date1'])
    df = df.merge(data,how='left',left_on='Date',right_on='Date1')
    df = df.drop(labels='Date1',axis=1)

#Renaming columns to represent instrument names rather than their ticker codes for ease of readability
names.insert(0,'Date')
df.columns = names
print(df.shape)
print(df.isna().sum())
df.tail()


# In[8]:


# Front filling the NaN values in the data set
df = df.fillna(method="ffill",axis=0)
df = df.fillna(method="bfill",axis=0)
df.isna().sum()


# In[9]:


df.tail()


# # Trend
# 
# 
# ## Detecting Trend Using a Hodrick-Prescott Filter

# In[10]:


#df = pd.read_excel('India_Exchange_Rate_Dataset.xls', parse_dates=True)
Gold_cycle, Gold_trend = hpfilter(df['Gold'], lamb=1600)
Gold_trend.plot(figsize=(15,6)).autoscale(axis='x',tight=True)


# ## Detrending Using Differencing

# In[11]:


diff = df.Gold.diff()
plt.figure(figsize=(15,6))
plt.plot(diff)
plt.title('Detrending using Differencing', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Gold prices')
plt.show()


# ## Detrending Using a SciPy Signal

# In[12]:


from scipy import signal
import warnings
warnings.filterwarnings("ignore")

detrended = signal.detrend(df.Gold.values)
plt.figure(figsize=(15,6))
plt.plot(detrended)
plt.xlabel('Gold prices')
plt.ylabel('Frequency')
plt.title('Detrending using Scipy Signal', fontsize=16)
plt.show()


# ## Detrend Using an HP Filter

# In[13]:


Gold_cycle,Gold_trend = hpfilter(df['Gold'], lamb=1600)
df['trend'] = Gold_trend
detrended = df.Gold - df['trend']
plt.figure(figsize=(15,6))
plt.plot(detrended)
plt.title('Detrending using HP Filter', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Gold prices')
plt.show()


# # Seasonality
# 
# The following methods can be used to detect seasonality:
# - Multiple box plots
# - Autocorrelation plots

# In[17]:


#df = pd.read_excel('India_Exchange_Rate_Dataset.xls', parse_dates=True)
df['month'] = df['Date'].dt.strftime('%b')
df['year'] = [d.year for d in df.Date]
df['month'] = [d.strftime('%b') for d in df.Date]
years = df['year'].unique()
plt.figure(figsize=(15,6))
sns.boxplot(x='month', y='Gold', data=df).set_title("Multi Month-wise Box Plot")
plt.show()


# ## Autocorrelation Plot

# In[18]:


from pandas.plotting import autocorrelation_plot

plt.rcParams.update({'figure.figsize':(14,6), 'figure.dpi':220})
autocorrelation_plot(df.Gold.tolist())
plt.show()


# # Deseasoning of Time-Series Data
# 
# ## Time-series data contains four main components.
# - Level means the average value of the time-series data.
# - Trend means an increasing or decreasing value in time-series data.
# - Seasonality means repeating the pattern of a cycle in the time-series data.
# - Noise means random variance in time-series data.
# 
# Additive model is when time-series data combines these four components for linear trend and seasonality, and a multiplicative
# model is when components are multiplied to gather for nonlinear trends and seasonality.
# 
# ## Seasonal Decomposition
# Decomposition is the process of understanding generalizations and problems related to time-series forecasting. We can leverage seasonal decomposition to remove seasonality from data and check the data only with the trend, cyclic, and irregular variations.
# 
# ### Errors, Unexpected Variations, and Residuals
# When trend and cyclical variations are removed from time-series data, the patterns left behind that cannot be explained are called errors, unexpected variations, or residuals. Various methods are available to check for irregular variations such as probability theory, moving averages, and autoregressive time-series methods. If we can find any cyclic variation in data, it is considered to be part of the residuals. These variations that occur due to unexpected circumstances are called unexpected variations or unpredictable errors.
# 
# ### Decomposing a Time Series into Its Components
# Decomposition is a method used to isolate the time-series data into different elements such as trends, seasonality, cyclic variance, and residuals. We can leverage seasonal decomposition from a stats model to decompose the data into its constituent parts, considering series as additive or multiplicative.
# 
# - Trends(T(t)) means an increase or decrease in the value of ts data.
# - Seasonality(S[t]) means repeating a short-term cycle of ts data.
# - Cyclic variations(c[t]) means a fluctuation in long trends of ts data.
# - Residuals(e[t]) means an irregular variation of ts data.
# 
# The additive model works with linear trends of time-series data such as changes constantly over time. The additive model formula is as follows: Y[t] = T[t] + S[t] + c[t] + e[t]
# 
# The multiplicative model works with a nonlinear type of data such as quadric or exponential. The multiplicative model formula is as follows: Y[t] = T[t] * S[t] * c[t] * e[t]

# In[28]:


from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# Additive
result = seasonal_decompose(df['Gold'], model='ad', freq=252)
result.plot()
plt.show()


# In[29]:


# Multiplicative
result = seasonal_decompose(df['Gold'], model='mul', freq=252)
result.plot()
plt.show()


# ## Cyclic Variations
# Cyclical components are fluctuations around a long trend observed every few units of time; this behavior is less frequent compared to seasonality. It is a recurrent process in a time series. In the field of business/economics,
# the following are three distinct types of cyclic variations examples:
# - Prosperity: As we know, when organizations prosper, prices go up, but the benefits also increase. On the other hand, prosperity also causes over-development, challenges in transportation, increments in wage rate, insufficiency in labor, high rates of returns, deficiency of cash in the market and price concessions, etc., leading to depression
# - Depression: As we know, when there is cynicism in exchange and enterprises, processing plants close down, organizations fall flat, joblessness spreads, and the wages and costs are low.
# - Accessibility: This causes idealness of money, accessibility of cash at a low interest, an increase in demand for goods or money at a low interest rate, an increase in popular merchandise and ventures described by the circumstance of recuperation that at last prompts for prosperity or boom.
# 
# ## Detecting Cyclical Variations
# The following code shows how to decompose time-series data and visualize only a cyclic pattern:

# In[25]:


Gold_cycle, Gold_trend = hpfilter(df['Gold'], lamb=1600)
df['cycle'] = Gold_cycle
df['trend'] = Gold_trend
df[['cycle']].plot(figsize=(15,6)).autoscale(axis='x',tight=True)
plt.title('Extracting Cyclic Variations', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Gold prices')
plt.show()


# ## Simple Exponential Smoothing

# In[30]:


from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn import metrics


# Modeling will be done only for the Gold column in the dataset. 

# In[31]:


X = df['Gold']
"""
Making a copy of the data to perform the train/test split, where the train is used for
training, and after getting satisfactory results, we will evaluate the results
on the test data.

The train will have all the data expected for the last 30 days, and the
test contains only the last 30 days to evaluate against predictions.
"""
test = X.iloc[-30:]
train = X.iloc[:-30]


# In[43]:


print(test.tail())
print(train.tail())


# In[32]:


def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')


# In[33]:


"""
The following snippet is used to find the best smoothing parameter,
which ranges from 0 to 1. The smoothing level model will be fit, and its
results will be captured in temp_df. The following results show that 1.0
gives the least RMSE.
"""
resu = []
temp_df = pd.DataFrame()
for i in [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,1]:
    print(f'Fitting for smoothing level= {i}')
    fit_v = SimpleExpSmoothing(np.asarray(train)).fit(i)
    fcst_pred_v= fit_v.forecast(30)
    timeseries_evaluation_metrics_func(test,fcst_pred_v)
    rmse = np.sqrt(metrics.mean_squared_error(test, fcst_pred_v))
    df3 = {'smoothing parameter':i, 'RMSE': rmse}
    temp_df = temp_df.append(df3, ignore_index=True)
temp_df.sort_values(by=['RMSE']).head(3)


# RMSE was achieved with smoothing_level equal to 0.1. Let’s use the same value and train the model.

# In[34]:


fitSES = SimpleExpSmoothing(np.asarray(train)).fit(smoothing_level = 0.1,optimized= False)
fcst_gs_pred = fitSES.forecast(30)
timeseries_evaluation_metrics_func(test,fcst_gs_pred)


# Finding the best parameters by adjusting a few settings.
# - optimized = True estimates the model parameters by maximizing the log likelihood.
# - use_brute= True searches for good starting values using a brute-force (grid) optimizer.

# In[35]:


fitSESauto = SimpleExpSmoothing(np.asarray(train)).fit(optimized= True, use_brute = True)
fcst_auto_pred = fitSESauto.forecast(30)
timeseries_evaluation_metrics_func(test,fcst_auto_pred)


# In[36]:


fitSESauto.summary()


# In[56]:


df_fcst_gs_pred = DataFrame(fcst_gs_pred, columns=['Gold_grid_Search'])
df_fcst_gs_pred["new_index"] = range(2773, 2803)
df_fcst_gs_pred = df_fcst_gs_pred.set_index("new_index")
df_fcst_auto_pred = DataFrame(fcst_auto_pred,columns=['Gold_auto_search'])
df_fcst_auto_pred["new_index"] = range(2773, 2803)
df_fcst_auto_pred = df_fcst_auto_pred.set_index("new_index")

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [16,9]

plt.plot( train, label='Train')
plt.plot(test, label='Test')
plt.plot(fcst_auto_pred, label='Simple Exponential Smoothing using optimized =True')
plt.plot(fcst_gs_pred, label='Simple Exponential Smoothing using custom grid search')
plt.legend(loc='best')
plt.show()


# We can clearly see that the simple exponential smoothing is not
# performing well as the stock market data, which will have trends and
# seasonality. Our basic model will not be able to capture these details.
# 
# ## Holt's Exponential Smoothing
# In Holt-Winter exponential smoothing, we have three smoothing
# constants.
# - Level: Lt = α (yt/St-M) + (1- α) (Lt-1 +Tt-1); S is a seasonal component. When we yt/St-M, we are the deseasonalizing value of y.
# - In the level equation, we are updating the previous level by Lt-1, adding Tt-1, and then combining and then combining the deseasonalizing value of yt.
# - Trend: Tt = β (Lt- Lt-1) + (1- β) Tt-1 ( Additive trend) We can update the previous trend by considering the latest difference between levels.
# - Seasonality: St = γ (Yt/Lt) + (1+ γ) St-M (multiplicative seasonality); Yt is divided by level component Lt. This gives the detrended value of Y. So, the seasonality is updated by combining the most recent seasonal component St-M with the detrended value of Yt.
# 
# The main idea here is to use SES and advance it to capture the trend component.

# In[46]:


from sklearn.model_selection import ParameterGrid
param_grid = {'smoothing_level': [0.10, 0.20,.30,.40,.50,.60,.70,.80,.90], 
              'smoothing_slope':[0.10, 0.20,.30,.40,.50,.60,.70,.80,.90],
              'damping_slope': [0.10, 0.20,.30,.40,.50,.60,.70,.80,.90],
              'damped' : [True, False]}
pg = list(ParameterGrid(param_grid))


# In[54]:


from timeit import default_timer as timer
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing, Holt

df_results_moni = pd.DataFrame(columns=['smoothing_level',
                                        'smoothing_slope', 'damping_slope','damped','RMSE','r2'])
start = timer()

for a,b in enumerate(pg):
    smoothing_level = b.get('smoothing_level')
    smoothing_slope = b.get('smoothing_slope')
    damping_slope = b.get('damping_slope')
    damped = b.get('damped')
    print(smoothing_level, smoothing_slope, damping_slope,damped)
    fit1 = Holt(train,damped =damped ).fit(smoothing_level=smoothing_level, 
                                           smoothing_slope=smoothing_slope,
                                           damping_slope = damping_slope, 
                                           optimized=False)
    #fit1.summary
    z = fit1.forecast(30)
    print(z)
    df_pred = pd.DataFrame(z, columns=['Forecasted_result'])
    RMSE = np.sqrt(metrics.mean_squared_error(test, df_pred.Forecasted_result))
    r2 = metrics.r2_score(test, df_pred.Forecasted_result)
    print( f' RMSE is {np.sqrt(metrics.mean_squared_error(test, df_pred.Forecasted_result))}')
    df_results_moni = df_results_moni.append({'smoothing_level':smoothing_level, 
                                              'smoothing_slope':smoothing_slope,
                                              'damping_slope' :damping_slope,
                                              'damped':damped,'RMSE':RMSE,'r2':r2}, 
                                             ignore_index=True)
end = timer()
print(f' Total time taken to complete grid search in seconds:{(end - start)}')
print(f' Below mentioned parameter gives least RMSE and r2')
df_results_moni.sort_values(by=['RMSE','r2']).head(1)


# In[57]:


fit1 = Holt(train,damped =False ).fit(smoothing_level=0.9,
smoothing_slope=0.6, damping_slope = 0.1, optimized=False)
Forecast_custom_pred = fit1.forecast(30)
fit1.summary()


# In[58]:


timeseries_evaluation_metrics_func(test,Forecast_custom_pred)


# In[59]:


# Let’s check whether the double exponential smoothing is able to find the best parameters
fitESAUTO = Holt(train).fit(optimized= True, use_brute = True)
fitESAUTO.summary()


# In[60]:


fitESAUTOpred = fitESAUTO.forecast(30)
timeseries_evaluation_metrics_func(test,fitESAUTOpred)


# In[62]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [16,9]
plt.plot( train, label='Train')
plt.plot(test, label='Test')
plt.plot(fitESAUTOpred, label='Automated grid search')
plt.plot(Forecast_custom_pred, label='Double Exponential Smoothing with custom grid search')
plt.legend(loc='best')
plt.show()


# From the evaluation metrics and graph, we can see that the double
# exponential smoothing performed significantly better than simple
# exponential smoothing.
# 
# ## Triple Exponential Smoothing
# Triple exponential smoothing forecasting method enforces
# exponential smoothing three times. This method can be applied when the
# data consumes trends and seasonality over time. It includes all smoothing
# component equations such as trends and seasonality. Seasonality
# comprises two different types, such as additive and multiplicative, which
# is a similar operation in mathematics. The Winter method uses the idea
# of the Holt method and adds seasonality.

# In[67]:


from sklearn.model_selection import ParameterGrid
from timeit import default_timer as timer

param_grid = {'trend': ['add', 'mul'],'seasonal' :['add','mul'],
              'seasonal_periods':[3,6,12], 'smoothing_level': [0.10,0.20,.30,.40,.50,.60,.70,.80,.90], 
              'smoothing_slope':[0.10, 0.20,.30,.40,.50,.60,.70,.80,.90],
              'damping_slope': [0.10, 0.20,.30,.40,.50,.60,.70,.80,.90],
              'damped' : [True, False], 'use_boxcox':[True, False],
              'remove_bias':[True, False],'use_basinhopping':[True, False]}
pg = list(ParameterGrid(param_grid))
df_results_moni = pd.DataFrame(columns=['trend','seasonal_periods','smoothing_level', 
                                        'smoothing_slope','damping_slope','damped','use_boxcox',
                                        'remove_bias','use_basinhopping','RMSE','r2'])
start = timer()
print('Starting Grid Search..')
for a,b in enumerate(pg):
    trend = b.get('trend')
    smoothing_level = b.get('smoothing_level')
    seasonal_periods = b.get('seasonal_periods')
    smoothing_level = b.get('smoothing_level')
    smoothing_slope = b.get('smoothing_slope')
    damping_slope = b.get('damping_slope')
    damped = b.get('damped')

use_boxcox = b.get('use_boxcox')
remove_bias = b.get('remove_bias')
use_basinhopping = b.get('use_basinhopping')
#print(trend,smoothing_level, smoothing_slope,damping_slope,damped,use_boxcox,remove_bias,use_basinhopping)
fit1 = ExponentialSmoothing(train,trend=trend,damped=damped,
                            seasonal_periods=seasonal_periods ).fit(smoothing_level=smoothing_level,
                                                                    smoothing_slope=smoothing_slope, 
                                                                    damping_slope = damping_slope,
                                                                    use_boxcox=use_boxcox,optimized=False)

#fit1.summary
z = fit1.forecast(30)
#print(z)
df_pred = pd.DataFrame(z, columns=['Forecasted_result'])
RMSE = np.sqrt(metrics.mean_squared_error(test, df_pred.Forecasted_result))
r2 = metrics.r2_score(test, df_pred.Forecasted_result)
#print( f' RMSE is {np.sqrt(metrics.mean_squared_error(test, df_pred.Forecasted_result))}')
df_results_moni = df_results_moni.append({'trend':trend,'seasonal_periods':seasonal_periods,
                                          'smoothing_level':smoothing_level, 
                                          'smoothing_slope':smoothing_slope,
                                          'damping_slope':damping_slope,'damped':damped,
                                          'use_boxcox':use_boxcox,'use_basinhopping':use_basinhopping,
                                          'RMSE':RMSE, 'r2':r2}, ignore_index=True)
print('End of Grid Search')
end = timer()
print(f' Total time taken to complete grid search in seconds:{(end - start)}')


# In[68]:


print(f' Below mentioned parameter gives least RMSE and r2')
df_results_moni.sort_values(by=['RMSE','r2']).head(1)


# In[69]:


fit1 = ExponentialSmoothing(train,trend='mul',
                            damped=False,seasonal_periods=3 ).fit(smoothing_level=0.9,
                                                                  smoothing_slope=0.6, damping_slope = 0.6,
                                                                  use_boxcox=False,use_basinhopping = True,
                                                                  optimized=False)
Forecast_custom_pred = fit1.forecast(30)
fit1.summary()
timeseries_evaluation_metrics_func(test,Forecast_custom_pred)


# In[70]:


fitESAUTO = ExponentialSmoothing(train).fit(optimized= True,use_brute = True)
fitESAUTO.summary()


# In[71]:


fitESAUTOpred = fitESAUTO.forecast(30)


# In[72]:


timeseries_evaluation_metrics_func(test,fitESAUTOpred)


# In[73]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [16,9]
plt.plot( train, label='Train')
plt.plot(test, label='Test')
plt.plot(fitESAUTOpred, label='Automated grid search')
plt.plot(Forecast_custom_pred, label='Triple Exponential Smoothing with custom grid search')
plt.legend(loc='best')
plt.show()


# By carefully observing evaluation metrics and graphs, we can conclude
# that triple exponential smoothing behaves the same way as double exponential
# smoothing.
# Therefore, we can conclude the following:
# - If there is no presence of trend or seasonality, then use simple exponential smoothing.
# - If there is a presence of trend and no seasonality, use double exponential smoothing.
# - If there is a presence of trend and seasonality, then use triple exponential smoothing.

# In[ ]:




