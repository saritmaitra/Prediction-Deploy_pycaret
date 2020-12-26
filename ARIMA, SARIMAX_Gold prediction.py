#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install pmdarima


# In[1]:


import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pmdarima import auto_arima
from sklearn import metrics
from yahoofinancials import YahooFinancials
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings("ignore")


# In[ ]:


ticker_details = pd.read_excel("Ticker List.xlsx")
ticker_details.head(20)


# In[ ]:


ticker = ticker_details['Ticker'].to_list()
names = ticker_details['Description'].to_list()


# In[ ]:


# Extracting Data from Yahoo Finance and Adding them to Values table using date as key
end_date= "2020-09-30"
start_date = "2010-01-01"
date_range = pd.bdate_range(start=start_date,end=end_date)
"""
Create a date-range and write it to an empty dataframe named values. Here I would extract and 
past the values pulled from yahoofinancials.
"""
df = pd.DataFrame({ 'Date': date_range})
df['Date']= pd.to_datetime(df['Date'])


# In[ ]:


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


# In[6]:


# Front filling the NaN values in the data set
df = df.fillna(method="ffill",axis=0)
df = df.fillna(method="bfill",axis=0)
df.isna().sum()


# In[7]:


df.tail()


# In[9]:


plt.style.use('dark_background')
df["Gold"].plot(figsize=(15, 6))
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Gold stock price")
plt.show()


# In[10]:


plt.figure(1, figsize=(15,6))
plt.subplot(211)
df["Gold"].hist()
plt.subplot(212)
df["Gold"].plot(kind='kde')
plt.show()


# In[21]:


def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    #print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MSE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true,y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')


# In[12]:


def Augmented_Dickey_Fuller_Test_func(series , column_name):
    print (f'Results of Dickey-Fuller Test for column: {column_name}')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                             'p-value','No Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print (dfoutput)
    if dftest[1] <= 0.05:
        print("Conclusion:====>")
        print("Reject the null hypothesis")
        print("Data is stationary")
    else:
        print("Conclusion:====>")
        print("Fail to reject the null hypothesis")
        print("Data is non-stationary")


# In[13]:


Augmented_Dickey_Fuller_Test_func(df['Gold' ],'Gold')


# We can see that Gold is nonstationary, and auto-arima handles this
# internally.
# 
# Model training will be done only for the Gold column from the
# dataset. Make a copy of the data, and let’s perform the test/train split.
# The train will have all the data except the last 30 days, and the test will
# contain only the last 30 days to evaluate against predictions.

# In[14]:


X = df[['Gold' ]]
train, test = X[0:-30], X[-30:]


# In[15]:


stepwise_model = auto_arima(train,start_p=1, start_q=1,
                            max_p=7, max_q=7, seasonal=False,
                            d=None, trace=True,error_action='ignore',
                            suppress_warnings=True, stepwise=True)
stepwise_model.summary()


# Auto-arima says ARIMA(0,1,0) is the optimal selection for the dataset.
# 
# We will Forecast both results and the confidence for the next 30 days and store
# it in a DataFrame

# In[16]:


test


# In[22]:


forecast,conf_int = stepwise_model.predict(n_periods=30,return_conf_int=True)
forecast = pd.DataFrame(forecast,columns=['close_pred'])
df_conf = pd.DataFrame(conf_int,columns= ['Upper_bound','Lower_bound'])
df_conf["new_index"] = range(2773, 2803)
df_conf = df_conf.set_index("new_index")
timeseries_evaluation_metrics_func(test, forecast)


# In[23]:


# Rearrange the indexes for the plots to align, as shown here:
forecast["new_index"] = range(2773, 2803)
forecast = forecast.set_index("new_index")


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [15,7]
plt.plot( train, label='Train ')
plt.plot(test, label='Test ')
plt.plot(forecast, label='Predicted ')
plt.plot(df_conf['Upper_bound'], label='Confidence Interval Upper bound ')
plt.plot(df_conf['Lower_bound'], label='Confidence Interval Lower bound ')
plt.legend(loc='best')
plt.show()


# In[26]:


stepwise_model.plot_diagnostics();


# # Seasonal ARIMA
# The seasonal ARIMA model combines both nonseasonal and seasonal
# components in a multiplicative model. The notation can be defined as follows:
# - ARIMA (p, d, q) X (P, D, Q)m
# - m is the number of observations per year.
# 
# Three trend elements need to be configured. It is same as the ARIMA
# model, as shown here:
# - (p, d, q) is a nonseasonal component, as shown here:
#   - p: Trend autoregressive order
#   - d: Trend differencing order
#   - q: Trend moving average order
# 
# (P, Q, D) is a seasonal component.
# 
# There are four seasonal components that are not part of the ARIMA
# model that are essential to be configured.
# - P: Seasonal autoregressive order
# - D: Seasonal differencing order
# - Q: Seasonal moving average order
# - m: Timestamp for single-season order
# 
# 
# Let’s configure and run seasonal arima for the parameters given in the
# for loop and check the optimal number of periods in each season suitable
# for our dataset.
# Here is the search space:
# - p → 1 to 7.
# - q → 1 to 7.
# - d → None means find the optimal value.
# - P → 1 to 7.
# - Q → 1 to 7.
# - D → None means find the optimal value.
# 
# m refers to the number of periods in each season.
# 
# - 7 → Daily
# - 12 → Monthly
# - 52 → Weekly
# - 4 → Quarterly
# - 1 → Annual (non-seasonal)

# In[28]:


for m in [1, 4,7,12,52]:
    print("="*100)
    print(f' Fitting SARIMA for Seasonal value m = {str(m)}')
    
    stepwise_model = auto_arima(train, start_p=1, start_q=1,
                                max_p=7, max_q=7, seasonal=True, start_P=1,
                                start_Q=1, max_P=7, max_D=7, max_Q=7, m=m,
                                d=None, D=None, trace=True, error_action='ignore', 
                                suppress_warnings=True,
                                stepwise=True)
    
    print(f'Model summary for m = {str(m)}')
    print("-"*100)
    stepwise_model.summary()
    
    forecast ,conf_int= stepwise_model.predict(n_periods=30,return_conf_int=True)
    df_conf = pd.DataFrame(conf_int,columns= ['Upper_bound','Lower_bound'])
    df_conf["new_index"] = range(2773, 2803)
    df_conf = df_conf.set_index("new_index")
    forecast = pd.DataFrame(forecast, columns=['close_pred'])
    forecast["new_index"] = range(2773, 2803)
    forecast = forecast.set_index("new_index")
    timeseries_evaluation_metrics_func(test, forecast)
    
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.rcParams["figure.figsize"] = [15, 7]
    plt.plot(train, label='Train ')
    plt.plot(test, label='Test ')
    plt.plot(forecast, label=f'Predicted with m={str(m)} ')
    plt.plot(df_conf['Upper_bound'], label='Confidence Interval Upper bound ')
    plt.plot(df_conf['Lower_bound'], label='Confidence Interval Lower bound ')
    plt.legend(loc='best')
    plt.show()
    print("-"*100)
    print(f' Diagnostic plot for Seasonal value m = {str(m)}')
    display(stepwise_model.plot_diagnostics());
    print("-"*100)


# After checking for different m values, we can see that m does not have
# any influence on the results.
# 
# ## SARIMAX
# SARIMAX model is SARIMA model with external influencing variables, called SARIMAX (p, d, q) X(P,D,Q)m (X), where X is the vector of exogenous variables. The exogenous variables perhaps modeled by the multilinear regression equation are articulated as follows:
# - (1−ϕ1B) (1−Φ1Bm) (1−B) (1−Bm) yt = (1+θ1B) (1+Θ1Bm) εt (Xk,t). where X1,, X2,𝑡, …. X𝑘, are observations of k number of exogenous variables corresponding to the dependent variable.

# In[30]:


df.tail(2) # checking the dataset


# In[34]:


X = df[['Gold' ]] # dependent variable
actualtrain, actualtest = X[0:-30], X[-30:]
exoX = df[['S&P500' ]]
#exoX = df.drop(columns = ['Gold'], axis=1) # independent variables
exotrain, exotest = exoX[0:-30], exoX[-30:]


# In[36]:


# Let’s configure and run seasonal arima with an exogenous variable.

for m in [1, 4,7,12,52]:
    print("="*100)
    print(f' Fitting SARIMAX for Seasonal value m = {str(m)}')
    stepwise_model = auto_arima(actualtrain,exogenous =exotrain ,
                                start_p=1, start_q=1,max_p=7, max_q=7, seasonal=True,
                                start_P=1,start_Q=1,max_P=7,max_D=7,max_Q=7,m=m, d=None,D=None,
                                trace=True,error_action='ignore',suppress_warnings=True,
                                stepwise=True)
    print(f'Model summary for m = {str(m)}')
    print("-"*100)
    stepwise_model.summary()
    
    forecast,conf_int = stepwise_model.predict(n_periods=30,
                                               exogenous =exotest,return_conf_int=True)
    df_conf = pd.DataFrame(conf_int,columns= ['Upper_bound','Lower_bound'])
    df_conf["new_index"] = range(2773, 2803)
    df_conf = df_conf.set_index("new_index")
    forecast = pd.DataFrame(forecast, columns=['close_pred'])
    forecast["new_index"] = range(2773, 2803)
    forecast = forecast.set_index("new_index")
    timeseries_evaluation_metrics_func(actualtest, forecast)
    
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.rcParams["figure.figsize"] = [15, 7]
    plt.plot(actualtrain, label='Train ')
    plt.plot(actualtest, label='Test ')
    plt.plot(forecast, label=f'Predicted with m={str(m)} ')
    plt.plot(df_conf['Upper_bound'], label='Confidence IntervalUpper bound ')
    plt.plot(df_conf['Lower_bound'], label='Confidence IntervalLower bound ')
    plt.legend(loc='best')
    plt.show()
    print("-"*100)
    print(f' Diagnostic plot for Seasonal value m = {str(m)}')
    display(stepwise_model.plot_diagnostics());
    print("-"*100)


# We can see that the exogenous variable S&P500 is not contributing to model accuracy, and we notice that m does not have any
# influence on prediction.

# In[ ]:




