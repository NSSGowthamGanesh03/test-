#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os




# In[2]:


df1= pd.read_csv("/Users/gowtham/Downloads/obdii-ds3/exp1_14drivers_14cars_dailyRoutes.csv", nrows=47515, low_memory=False)


# In[3]:


df1.shape


# In[4]:


df1.isnull().sum()


# In[5]:


df1.info()


# In[6]:


#to display all columns
pd.set_option('display.max_columns',None)


# In[7]:


#to see the top 5 rows
df1.head()


# In[8]:


#to see the value conts of a particular column
df1.ENGINE_POWER.value_counts()


# In[9]:


df1.AUTOMATIC.value_counts()


# In[10]:


#to rename the column name
df1.rename(columns = {'BAROMETRIC_PRESSURE(KPA)':'Baro_PRESSURE'}, inplace = True)


# In[11]:


df1.head()


# In[12]:


df1.Baro_PRESSURE.value_counts()


# In[13]:


# to drop rows of  null values present in the particular columns
df1.dropna(subset=['ENGINE_LOAD',],inplace=True)


# In[14]:


#total no of null values
df1.isnull().sum()


# In[15]:


#shape of the data frame
df1.shape


# In[16]:


# columns in the data frame
df1.columns


# In[17]:


list_del=['Baro_PRESSURE','FUEL_LEVEL',
      'AMBIENT_AIR_TEMP','MAF', 'LONG TERM FUEL TRIM BANK 2',
       'FUEL_TYPE', 'FUEL_PRESSURE', 'SHORT TERM FUEL TRIM BANK 2', 'SHORT TERM FUEL TRIM BANK 1',
       'ENGINE_RUNTIME', 'TROUBLE_CODES',
       'TIMING_ADVANCE', 'EQUIV_RATIO', 'MIN', 'HOURS', 'DAYS_OF_WEEK',
       'MONTHS', 'YEAR']


# In[18]:


# list of columns to be dropped
df1.drop(list_del, axis=1,inplace=True)


# In[19]:


df1.head(10)


# In[20]:


df1.shape


# In[21]:


df1.isnull().sum()


# In[22]:


df1.shape


# In[23]:


# for joining two columns i mean concacenate
#df1['Ca'] = df1[['VEHICLE_ID', 'ENGINE_LOAD']].apply(lambda x: ''.join(x), axis=1)


# In[24]:


df1.head()


# In[25]:


#import pandas_profiling
#df1.profile_report()


# In[26]:


# to see particular row together have null values
df1[df1['ENGINE_POWER'].isna()][['MARK', 'MODEL', 'CAR_YEAR', 'AUTOMATIC','VEHICLE_ID']]


# In[27]:


# to see which all cars are automatic
pd.set_option('display.max_rows',None)
df1[df1['AUTOMATIC']=='s'][['VEHICLE_ID']]


# In[28]:


df1.ENGINE_LOAD.unique()


# In[29]:


df1[df1['ENGINE_POWER']=='1'][['VEHICLE_ID']]


# In[30]:


list_del1= ['ENGINE_POWER','MARK', 'MODEL', 'CAR_YEAR', 'AUTOMATIC']
df1.drop(list_del1, axis=1,inplace=True)


# In[31]:


df1.head(10)


# In[32]:


df1.VEHICLE_ID.value_counts()


# In[33]:


df1.isnull().sum()


# In[34]:


df1.INTAKE_MANIFOLD_PRESSURE.unique()


# In[35]:


df1.shape


# In[36]:


df1.drop('DTC_NUMBER', axis=1,inplace=True)


# In[37]:


df1.isnull().sum()


# In[38]:


df1.ENGINE_COOLANT_TEMP.value_counts(ascending=True)


# In[39]:


df1.dtypes


# In[40]:


df1[df1['INTAKE_MANIFOLD_PRESSURE'].isna()][['VEHICLE_ID']]


# In[41]:


#import pandas_profiling
#df1.profile_report()


# In[42]:


df1.INTAKE_MANIFOLD_PRESSURE.value_counts(ascending=True)


# In[43]:


df1.ENGINE_LOAD.value_counts(sort=False)


# In[44]:


df1.ENGINE_COOLANT_TEMP.value_counts()


# In[45]:


df1[df1['ENGINE_COOLANT_TEMP'].isna()][['ENGINE_LOAD']]


# In[46]:


df1['ENGINE_LOAD']= df1['ENGINE_LOAD'].apply(lambda x: x.replace(',','.'))


# In[47]:


df1['ENGINE_LOAD']= df1['ENGINE_LOAD'].apply(lambda x: x.replace('%',''))


# In[48]:


df1.head()


# In[49]:


#import pandas_profiling
#df1.profile_report()


# In[50]:


df1['THROTTLE_POS'].fillna('13%', inplace=True)


# In[51]:


df1.isnull().sum()


# In[52]:


df1['THROTTLE_POS']= df1['THROTTLE_POS'].apply(lambda x: x.replace('%',''))


# In[53]:


df1.head()


# In[54]:


df1['THROTTLE_POS'].value_counts()


# In[55]:


df1.drop(['INTAKE_MANIFOLD_PRESSURE','SPEED'],axis=1,inplace=True)


# In[56]:


df1.head()


# In[57]:


df1.isnull().sum()


# In[58]:


df1.dtypes


# In[59]:


df1["ENGINE_COOLANT_TEMP"].fillna(df1["ENGINE_COOLANT_TEMP"].mean(), inplace=True)


# In[60]:


df1["ENGINE_RPM"].fillna(df1["ENGINE_RPM"].mean(), inplace=True)


# In[61]:


df1["AIR_INTAKE_TEMP"].fillna(df1["AIR_INTAKE_TEMP"].mean(), inplace=True)


# In[62]:


df1.isnull().sum()


# In[63]:


df1.dtypes


# In[64]:


df1.drop(['TIMESTAMP'],axis=1,inplace=True)


# In[65]:


for i in df1.index:
    if df1['ENGINE_LOAD'][i]< '50' :
        df1['ENGINE_LOAD'][i]= 0
    else:
        df1['ENGINE_LOAD'][i]= 1


# In[66]:


df1.head()


# In[67]:


df1.ENGINE_LOAD.value_counts()


# In[68]:


num_cols=["ENGINE_COOLANT_TEMP","ENGINE_RPM","AIR_INTAKE_TEMP","THROTTLE_POS"]


# In[69]:


cat_cols=df1.columns.difference(num_cols)


# In[70]:


df1[cat_cols] = df1[cat_cols].apply(lambda x: x.astype('category')) 
df1[num_cols] = df1[num_cols].apply(lambda x: x.astype('float'))


# In[71]:


df1.dtypes


# In[72]:


from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder() 


# In[73]:


df1['VEHICLE_ID'] = labelEncoder.fit_transform(df1['VEHICLE_ID'])


# In[74]:


#df1=pd.get_dummies(df1, columns=['VEHICLE_ID'])


# In[75]:


df1.VEHICLE_ID.value_counts()


# In[76]:


df1.head()


# In[77]:


df1.AIR_INTAKE_TEMP.sort_values()


# In[78]:


y=df1['ENGINE_LOAD']
x=df1.drop(['ENGINE_LOAD'],axis=1)


# In[79]:


""""def rescale_x(scaler_option, x_train):
    scale = None
    if scaler_option=='False':
        x_train_ = x_train
    elif scaler_option == "MinMaxScaler":
        scale = preprocessing.MinMaxScaler()
        x_train_ = scale.fit_transform(x_train)
    elif scaler_option == "MaxAbsScaler":
        scale = preprocessing.MaxAbsScaler()
        x_train_ = scale.fit_transform(x_train)
    elif scaler_option == "RobustScaler":
        scale = preprocessing.RobustScaler()
        x_train_ = scale.fit_transform(x_train)
    elif scaler_option == "QuantileTransformer":
        scale = preprocessing.QuantileTransformer()
        x_train_ = scale.fit_transform(x_train)
    elif scaler_option == "Normalizer":
        scale = preprocessing.Normalizer()
        x_train_ = scale.fit_transform(x_train)
    else:
        scale = preprocessing.StandardScaler()
        x_train_ = scale.fit_transform(x_train)
    return x_train_, scale""""


# In[80]:


x_train, x_val, y_train, y_val = train_test_split(x.values, y.values, test_size=0.20, random_state=10)


# In[81]:


x_train.shape


# In[82]:


x_val.shape


# In[83]:


y_train.shape


# In[84]:


#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(x_train.iloc[:,1:])
#scaler.fit(x_val.iloc[:,1:])
#x_train.iloc[:,1:] = scaler.transform(x_train.iloc[:,1:])
#x_val.iloc[:,1:] = scaler.transform(x_val.iloc[:,1:])


# In[85]:


y_train.head()


# In[86]:


x_train.head()


# In[87]:


x_val.head()


# In[88]:


from xgboost import  XGBClassifier
X_classifier = XGBClassifier(max_depth=5,
    learning_rate=0.1,
    n_estimators=200)
X_classifier.fit(x_train,y_train)

Y_train_xgb = X_classifier.predict(x_train)
Y_test_xgb =X_classifier.predict(x_val)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,Y_train_xgb))
print(accuracy_score(y_val,Y_test_xgb))


# In[89]:


pickle.dump(X_classifier, open('model.pkl','wb'))


# In[90]:


model = pickle.load(open('model.pkl','rb'))


# In[93]:


print(model.predict([[1,80,995,60,26]]))


# In[ ]:




