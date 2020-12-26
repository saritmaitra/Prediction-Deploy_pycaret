#!/usr/bin/env python
# coding: utf-8

# # Predicting return from Gold:
# ## Importing and Preparing Data:
# For this and subsequent exercises we will need closing price of several instruments for past 10 years . There are various paid (Reuters, Bloomberg) and free resources (IEX, Quandl, Yahoofinance, Google finance) that we can use to either extract and load data in csv or we can directly call their APIs. Since in this project I needed different type of asset classes (Equities, Commodities, Debt and precious metals)

# In[1]:


get_ipython().system('pip install pyforest')
from pyforest import *
import datetime, pickle, copy
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
get_ipython().system('pip install yahoofinancials')
from yahoofinancials import YahooFinancials


# In[2]:


# Import tickers 
ticker_details = pd.read_excel("Ticker List.xlsx")
ticker_details.head(20)


# In[3]:


ticker = ticker_details['Ticker'].to_list()
names = ticker_details['Description'].to_list()


# Once we have the list, we need to define what date range we need to import the data for. The period I have chosen is Jan 2010 till 1st Mar 2020. The reason I did not pull data prior to that is because the GFC in 2008-09 massively changed the economic and market landscapes. Relationships pririo to that peirod might be of less relevance now. We also dont want to feed very less data as the models might tend to overfit.
# 
# We create a date-range and write it to an empty dataframe named values where we would extract and past the values we pull from yahoofinancials.

# In[4]:


#Extracting Data from Yahoo Finance and Adding them to Values table using date as key
end_date= "2020-09-30"
start_date = "2010-01-01"
date_range = pd.bdate_range(start=start_date,end=end_date)
values = pd.DataFrame({ 'Date': date_range})
values['Date']= pd.to_datetime(values['Date'])


# Once we have the date range in dataframe, we need to use ticker symbols to pull out data from the API. yahoofinancials returns the output in a JSON format. The following code loops over the the list of ticker symbols and extracts just the closing prices for all the historical dates and keeps them adding to the dataframe horizontally. Note I have used the merge function to mantain the sanctity of dates. Given these asset classes might have different regional and trading holidays, the date ranges are not bound to be the same. By merging, we will eventually have several NAs which we will frontfill later on.

# In[5]:


#Extracting Data from Yahoo Finance and Adding them to Values table using date as key
for i in ticker:
    raw_data = YahooFinancials(i)
    raw_data = raw_data.get_historical_price_data(start_date, end_date, "daily")
    df = pd.DataFrame(raw_data[i]['prices'])[['formatted_date','adjclose']]
    df.columns = ['Date1',i]
    df['Date1']= pd.to_datetime(df['Date1'])
    values = values.merge(df,how='left',left_on='Date',right_on='Date1')
    values = values.drop(labels='Date1',axis=1)


# In[6]:


#Renaming columns to represent instrument names rather than their ticker codes for ease of readability
names.insert(0,'Date')
values.columns = names
print(values.shape)
print(values.isna().sum())
values.tail()


# In[7]:


#Front filling the NaN values in the data set
values = values.fillna(method="ffill",axis=0)
values = values.fillna(method="bfill",axis=0)
values.isna().sum()


# In[8]:


# Co-ercing numeric type to all columns except Date
cols=values.columns.drop('Date')
values[cols] = values[cols].apply(pd.to_numeric,errors='coerce').round(decimals=1)
values.tail()


# In[9]:


values.to_csv("Class_Training Data_Values.csv")


# In[10]:


from pandas import DataFrame

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


# Besides just the lagged returns, we also see how far the current Gold price is from its moving average for with different window. This is a very commonly used metric in technical analysis where moving averages offer supports and resistances for asset prices. We use a combination of simple and exponential moving averages. We then add these moving averages to the existing feature space.

# In[11]:


#Calculating Moving averages for Gold
ma = DataFrame(values['Date'],columns=['Date'])
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


# In[12]:


#Merging Moving Average values to the feature space
print(data.shape)
data['Date']=pd.to_datetime(data['Date'],format='%Y-%b-%d')
data = pd.merge(left=data,right=ma,how='left',on='Date')
print(data.shape)
data.isna().sum()


# This wall all about features. Now we need to create targets, i.e what we want to predict. Since we are predicting returns, we need to pick a horizon for which we need to predict returns. I have chosen 14-day and 22-day horizons because other smaller horizons tend to be very volatile and lack and predictive power. One can however, experiment with other horizons as well.
# 
# 

# In[13]:


#Caluculating forward returns for Target
y = DataFrame(data=values['Date'])
print(y.shape)

y['T+14']=values["Gold"].pct_change(periods=-14)
y['T+22']=values["Gold"].pct_change(periods=-22)
print(y.shape)
y.isna().sum()


# In[14]:


# Removing NAs
print(data.shape)
data = data[data['Gold-T-250'].notna()]
y = y[y['T+22'].notna()]
print(data.shape)
print(y.shape)


# In[15]:


# Now we will merge the Target variables with the feature space to get a data whcih we can finally start modelling on.

from pandas import merge

#Adding Target Variables
data = merge(left=data,right=y,how='inner',on='Date',suffixes=(False,False))
print(data.shape)
data.isna().sum()


# In[16]:


import seaborn as sns
plt.style.use('dark_background')
sns.distplot(data['T+22']); plt.show()
sns.distplot(data['T+14']); plt.show()


# In[17]:


data.to_csv("Class_Training Data.csv",index=False)


# ## Creating Labels
# We will try to predict any adverse (negative) return in Gold beyond a threshold.This threshold can be defined based on the risk tolerance of the investor. Here, I have taken the threshold to be 15% lowest return observations in the data history. In effect I am training to the model to predict a fall equal to or worse than 394 worse days in past 10 years.

# In[18]:


import scipy.stats as st
#Select Threshold p (left tail probability)
p= 0.15
#Get z-Value
z = st.norm.ppf(p)
print(z)


# In[19]:


#Calculating Threshold (t) for each Y
t_14 = round((z*np.std(data["T+14"]))+np.mean(data["T+14"]),5)
t_22 = round((z*np.std(data["T+22"]))+np.mean(data["T+22"]),5)

print("t_14=",t_14)
print("t_22=",t_22)


# So We can see above that threshold for 14-day model is -0.037 or -3.7%. This means that Gold returns over 140day period has been lower than -3.7% only 15 out of 100 days. We will label them as Target outcomes. Similar for T+22 days and T+5 Days

# In[20]:


#Creating Labels
data['Y-14'] = (data['T+14']< t_14)*1
data['Y-22']= (data['T+22']< t_22)*1
print("Y-14", sum(data['Y-14']))
print("Y-22", sum(data['Y-22']))


# In[21]:


data.head()


# In[22]:


# Now that we have the labels to predict, we can delet the return columns

data = data.drop(['T+14','T+22'],axis=1)
data.head()


# # 22-Day Model

# In[23]:


get_ipython().system('pip install pycaret')
from pycaret.classification import *


# In[24]:


data_22 = data.drop(['Y-14'],axis=1)
data_22.head()


# In[25]:


s22 = setup(data=data_22, target='Y-22', session_id=11, silent=True);


# In[26]:


compare_models(turbo=False)


# In[27]:


mlp = create_model('mlp')


# In[28]:


lgbm = create_model('lightgbm')


# In[29]:


et = create_model('et')


# In[30]:


catb = create_model('catboost')


# ### Hyparameters tunning:

# In[31]:


mlp_tune = tune_model(mlp, n_iter=50,optimize='Recall')


# In[32]:


lgbm_tune = tune_model(lgbm, n_iter=50,optimize='Recall')


# In[33]:


et_tune = tune_model (et, n_iter=150)


# In[34]:


catb_tune = tune_model(catb, n_iter=150,optimize='Recall')


# ## Ensemble approach:
# ### Bagging & Boosting:
# Bagging serves as a good introduction to ensemble methods because it is relatively
# easy to understand and because it is relatively easy to demonstrate its variance
# reduction properties.

# In[35]:


mlp_bagged = ensemble_model(estimator=mlp, method='Bagging')


# In[36]:


et_boosting = ensemble_model(estimator=et, method='Boosting')


# In[37]:


catb_tune_boosted = ensemble_model(estimator=catb_tune, method= 'Bagging')


# In[38]:


lgbm_bagged = ensemble_model(lgbm, method='Bagging')


# In[39]:


lgbm_boosted = ensemble_model(lgbm, method='Boosting')


# ### Blending models:

# In[40]:


blend1 = blend_models(estimator_list=[mlp,lgbm,et])


# In[41]:


blend2 = blend_models(estimator_list=[lgbm,et], method='soft')


# ## Stacking Models

# In[42]:


#stack1 = create_stacknet(estimator_list=[[catb_tune,blend2],[mlp]], restack=False)


# In[43]:


stack2 = stack_models(estimator_list=[catb_tune,blend2], meta_model=mlp, restack=True)


# In[44]:


stack3 = stack_models(estimator_list=[catb_tune,blend2], meta_model=mlp, restack=False)


# In[45]:


#stack4 = create_stacknet(estimator_list=[[catb_tuned,blend2,mlp],[mlp]], restack=False)


# ## Evaluation

# In[46]:


get_ipython().system('pip install scikit-plot')


# In[47]:


import scikitplot 


# In[48]:


evaluate_model(mlp)


# In[49]:


evaluate_model(et)


# In[50]:


evaluate_model(lgbm)


# In[51]:


a = predict_model(stack3)


# In[52]:


print(a['Label'].sum())
print(a['Y-22'].sum())
print(a.shape)


# In[67]:


classifier_22 = finalize_model(stack3)
save_model(classifier_22,"22D Classifier")


# # 14- Day Model:

# In[54]:


data_14 = data.drop(['Y-22'],axis=1)
s14 = setup(data=data_14, target='Y-14', session_id=11, silent=True, ignore_features=['Date']);


# In[55]:


compare_models(sort='Recall', turbo= False)


# In[56]:


mlp14 = create_model('mlp')


# In[57]:


knn = create_model('knn')


# In[58]:


knn14_tune = tune_model(knn, n_iter=100)


# In[59]:


dt14 = create_model('dt')


# In[60]:


dt_boosted = ensemble_model(estimator=dt14, method='Boosting')


# In[61]:


blend14_1 = blend_models(estimator_list=[knn14_tune, dt14, mlp14])


# In[62]:


stack14_1 = stack_models(estimator_list = [knn14_tune, dt14, blend14_1], restack= False)


# In[63]:


stack14_2 = stack_models(estimator_list = [knn14_tune, dt14, mlp14], 
                         meta_model= blend14_1, restack= True)


# In[64]:


stack14_3 = stack_models(estimator_list = [knn14_tune, dt14, mlp14], 
                         meta_model= blend14_1,restack= False)


# In[66]:


classifier_14 = finalize_model(stack14_2)
save_model(classifier_14, "14D Classifier")


# In[ ]:




