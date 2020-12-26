#!/usr/bin/env python
# coding: utf-8

# # 1 Predicting return from Gold:
# ## Background:
# Gold has been the original store of value and medium of exchange for mankind for centuries till paper/or fiat currency took over a couple of centuries ago. However, most of the sustainable paper currencyies were backed by Gold as late as 1971, when the Bretton Woods agreement was scrapped and world currencies became a true $'Fiat'$ currency.
# 
# Gold however continues to be os interest not only as metal of choice for jewellery, but also as store of value and often advisable part of investment portfolio as it tends to be a hedge and safe haven when economies tend to (or atleat appear to) be in or at brink of collapse.
# 
# Currently there are numerous instruments which can give an investor exposure to Gold and they not necessarily need to keep it physically in their vaults. Exchange traded Funds (ETFs) is the most widely used instrument. As of April 2020, a total of USD175bn is invested in Gold ETFs across the globe. This was corpus was just USD24bn in 2008 before the Global Financial Crisis (GFC)
# 
# ## Goal setting:
# - Primary goal is to predict return from Gold prices using Machine learning. 
# - We will use supervised learning methods of regression and classification. 
# - We will then use Time Series methods. 
# - Finally we will try to integrate them to see of their predictive powers increases due to integration.
# 
# ## Project charter:
# 
# - First we will go the regression route to predict future returns of Gold over next 2 weeks and 3 weeks period. 
# - We will do this by using historical returns of different instruments which I beleive impact or likely to impact the outlook towards Gold. 
# - The fundamental reason is, I term Gold as a 'reactionary' asset. It has little fundamentals of its own and movement in prices is often is a derivative of how investors view other asset classes (equities and commdities).

# In[1]:


#Importing Libraries
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials


# ## 1.1: Retrieving and Preparing Data:
# For this and subsequent exercises we will need closing price of several instruments for past 10 years . There are various paid (Reuters, Bloomberg) and free resources (IEX, Quandl, Yahoofinance, Google finance) that we can use to either extract and load data in csv or we can directly call their APIs. Since in this project I needed different type of asset classes (Equities, Commodities, Debt and precious metals)

# In[2]:


ticker_details = pd.read_excel("Ticker List.xlsx")
ticker_details.head(20)


# In[3]:


ticker = ticker_details['Ticker'].to_list()
names = ticker_details['Description'].to_list()


# - Now that we have the list, we need to define what date range we need to import the data for. 
# - The period I have chosen is Jan 2010 till 30 Sept 2020. 
# - The reason I did not pull data prior to that is because the GFC in 2008-09 massively changed the economic and market landscapes. Relationships pririo to that peirod might be of less relevance now. 
# - I also dont want to feed very less data as the models might tend to overfit.

# In[4]:


#Extracting Data from Yahoo Finance and Adding them to Values table using date as key
end_date= "2020-09-30"
start_date = "2010-01-01"
date_range = pd.bdate_range(start=start_date,end=end_date)
"""
Create a date-range and write it to an empty dataframe named values. Here I would extract and 
past the values pulled from yahoofinancials.
"""
values = pd.DataFrame({ 'Date': date_range})
values['Date']= pd.to_datetime(values['Date'])


# - Once we have the date range in dataframe, we need to use ticker symbols to pull out data from the API. 
# - yahoofinancials returns the output in a JSON format. 
# - Below code loops over the the list of ticker symbols and extracts just the closing prices for all the historical dates and keeps them adding to the dataframe horizontally. 
# - I have used the merge function to mantain the sanctity of dates. 
# - These asset classes might have different regional and trading holidays, the date ranges are not bound to be the same.
# - By merging, we will eventually have several NAs which we will frontfill later on.

# In[5]:


from pandas import DataFrame 
#Extracting Data from Yahoo Finance and Adding them to Values table using date as key
for i in ticker:
    raw_data = YahooFinancials(i)
    raw_data = raw_data.get_historical_price_data(start_date, end_date, "daily")
    df = DataFrame(raw_data[i]['prices'])[['formatted_date','adjclose']]
    df.columns = ['Date1',i]
    df['Date1']= pd.to_datetime(df['Date1'])
    values = values.merge(df,how='left',left_on='Date',right_on='Date1')
    values = values.drop(labels='Date1',axis=1)

#Renaming columns to represent instrument names rather than their ticker codes for ease of readability
names.insert(0,'Date')
values.columns = names
print(values.shape)
print(values.isna().sum())
values.tail()


# ## 1.2: Cleansing, integrating, and transforming data:

# In[6]:


# Front filling the NaN values in the data set
values = values.fillna(method="ffill",axis=0)
values = values.fillna(method="bfill",axis=0)
values.isna().sum()


# In[7]:


# Co-ercing numeric type to all columns except Date
cols=values.columns.drop('Date')
values[cols] = values[cols].apply(pd.to_numeric,errors='coerce').round(decimals=1)
values.tail()


# In[8]:


values.to_csv("Reg_Training Data_Values.csv")


# In[9]:


imp = ['Gold','Silver', 'Crude Oil', 'S&P500','MSCI EM ETF']

# Calculating Short term -Historical Returns
change_days = [1,3,5,14,21]

data = DataFrame(data=values['Date'])
for i in change_days:
    print(data.shape)
    x= values[cols].pct_change(periods=i).add_suffix("-T-"+str(i))
    data=pd.concat(objs=(data,x),axis=1)
    x=[]
print(data.shape)

# Calculating Long term Historical Returns
change_days = [60,90,180,250]

for i in change_days:
    print(data.shape)
    x= values[imp].pct_change(periods=i).add_suffix("-T-"+str(i))
    data=pd.concat(objs=(data,x),axis=1)
    x=[]
print(data.shape)


# - Besides just the lagged returns, we also see how far the current Gold price is from its moving average for with different window. 
# - This is a very commonly used metric in technical analysis where moving averages offer supports and resistances for asset prices. 
# - I will use a combination of simple and exponential moving averages. 

# In[10]:


#Calculating Moving averages for Gold
ma = pd.DataFrame(values['Date'],columns=['Date'])
ma['Date']=pd.to_datetime(ma['Date'],format='%Y-%b-%d')
ma['15SMA'] = (values['Gold']/(values['Gold'].rolling(window=15).mean()))-1
ma['30SMA'] = (values['Gold']/(values['Gold'].rolling(window=30).mean()))-1
ma['60SMA'] = (values['Gold']/(values['Gold'].rolling(window=60).mean()))-1
ma['90SMA'] = (values['Gold']/(values['Gold'].rolling(window=90).mean()))-1
ma['180SMA'] = (values['Gold']/(values['Gold'].rolling(window=180).mean()))-1
ma['90EMA'] = (values['Gold']/(values['Gold'].ewm(span=90,adjust=True,ignore_na=True).mean()))-1
ma['180EMA'] = (values['Gold']/(values['Gold'].ewm(span=180,adjust=True,ignore_na=True).mean()))-1
ma = ma.dropna(axis=0)
print(ma.shape)
ma.head()


# ## 1.3: Joining tables

# In[11]:


# Add moving averages to the existing feature space.

from pandas import merge

print(data.shape)
data['Date']=pd.to_datetime(data['Date'],format='%Y-%b-%d')
data = merge(left=data,right=ma,how='left',on='Date')
print(data.shape)
data.isna().sum()


# ## 1.4: Setting target variables

# - We are predicting returns here, so, we need to pick a horizon for which we need to predict the returns. 
# - I have chosen 14-day and 22-day horizons because other smaller horizons tend to be very volatile and lack and predictive power. 

# In[12]:


from pandas import DataFrame
#Caluculating forward returns for Target
y = DataFrame(data=values['Date'])
print(y.shape)
y['T+14']=values["Gold"].pct_change(periods=-14)
y['T+22']=values["Gold"].pct_change(periods=-22)
print(y.shape)
y.isna().sum()


# ##### Target variables here are the return percentages 

# In[13]:


# Removing NAs
print(data.shape)
data = data[data['Gold-T-250'].notna()]
y = y[y['T+22'].notna()]
print(data.shape)
print(y.shape)


# In[14]:


from pandas import merge
#Adding Target Variables
data = merge(left=data,right=y,how='inner',on='Date',suffixes=(False,False))
print(data.shape)
data.isna().sum()


# In[15]:


data.to_csv("Reg_Training Data_Values.csv",index=False)


# #### Data is clean now and transformed to supervised ML algorithms

# In[16]:


plt.style.use('dark_background')
corr = data.corr().iloc[:,-2:].drop(labels=['T+14','T+22'],axis=0)
sns.distplot(corr.iloc[:,0])
plt.show()


# In[17]:


pd.set_option('display.max_rows', None)
corr_data = data.tail(2000).corr()
corr_data = pd.DataFrame(corr_data['T+14'])
#corr_data = corr_data.iloc[3:,]
corr_data = corr_data.sort_values('T+14',ascending=False)
#corr_data

sns.distplot(corr_data)
plt.show()


# # 2. Regression:
# I will use PyCaret machine learning library to automate the machine learning workflow and speed up the productivity.
# ## 2.1 22 Day Model:

# In[18]:


from pycaret.regression import *
data_22= data.drop(['T+14'],axis=1)
data_22.head()


# In[19]:


a = setup(data_22,target='T+22',
        ignore_features=['Date'],session_id=11,
        silent=True,profile=False,remove_outliers=False);
        #transformation=True,
        #pca=True,pca_method='kernel',
        #pca_components=10,
        #create_clusters=True,
        #cluster_iter=10,
        #feature_ratio=True,
        #normalize=True,
        #transform_target=True,
       #silent=True);


# In[20]:


compare_models(turbo=True)


# ## 2.2 Model building: 
# ### 2.2.1 KNN:

# In[21]:


print('creating knn model:')
knn = create_model('knn')


# In[22]:


print('Tunning knn model:')
knn_tuned = tune_model(knn,n_iter=150)


# ### 2.2.2 CatBoost:

# In[23]:


print('creating catboost model:')
catb = create_model('catboost')


# In[24]:


catb_tuned = tune_model(catb)


# ### 2.2.3 ExtraTree:

# In[25]:


print('creating extratree model:')
et = create_model('et')


# In[26]:


et_tuned = tune_model(et)


# ## 2.3 Model evaluation:

# In[27]:


evaluate_model(knn_tuned)


# In[28]:


evaluate_model(et_tuned)


# ## 2.4 Ensembling Models:

# In[32]:


et_bagged = ensemble_model(et,method='Bagging')


# In[33]:


knn_tuned_bagged = ensemble_model(knn_tuned, method='Bagging')


# ## 2.5 Blending Models:

# In[34]:


blend_knn_et = blend_models(estimator_list=[knn_tuned,et])


# In[49]:


stack1 = stack_models(estimator_list=[catb,knn_tuned],restack=True)


# In[50]:


stack2 = stack_models(estimator_list=[catb,et,knn_tuned], meta_model = blend_knn_et, restack=True)


# In[52]:


stack3 = stack_models(estimator_list=[catb,et,knn_tuned,blend_knn_et], restack=True,meta_model=blend_knn_et)


# In[53]:


save_model(model=stack2, model_name='22Day Regressor')


# # 3 14 Day Model:

# In[54]:


data_14= data.drop(['T+22'],axis=1)
data_14.head()


# In[55]:


c=setup(data_14,target='T+14',
        ignore_features=['Date'],session_id=11,
        silent=True,profile=False,remove_outliers=True);
        #transformation=True,
        #pca=True,pca_method='kernel',
        #pca_components=10,
        #create_clusters=True,
        #cluster_iter=10,
        #feature_ratio=True,
        #normalize=True,
        #transform_target=True,
       #silent=True);


# In[57]:


compare_models(turbo=True)


# In[59]:


knn_tuned = tune_model(knn,n_iter=150)
catb = create_model('catboost')
et = create_model('et')
knn_tuned_bagged = ensemble_model(knn_tuned, method='Bagging')
blend_knn_et = blend_models(estimator_list=[knn_tuned,et])


# In[60]:


stack1 = stack_models(estimator_list=[catb,blend_knn_et],restack=True)


# In[61]:


stack2 = stack_models(estimator_list=[catb,blend_knn_et], restack=True)


# In[62]:


save_model(model=stack2, model_name='14Day Regressor')


# # Deploy model on Google Cloud Platform (GCP)

# In[30]:


import os
# Upload model
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'c:/path-to-json-file.json'from pycaret.regression import deploy_model
deploy_model(model = model, model_name = 'model-name', platform = 'gcp', 
             authentication = {'project' : 'project-name', 'bucket' : 'bucket-name'})


# In[31]:


import os
# access the model from the GCP bucket
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'c:/path-to-json-file.json'from pycaret.regression import load_model
loaded_model = load_model(model_name = 'model-name', platform = 'gcp', 
                          authentication = {'project' : 'project-name', 'bucket' : 'bucket-name'})from pycaret.regression import predict_model
predictions = predict_model(loaded_model, data = new-dataframe)

