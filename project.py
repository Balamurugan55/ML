
# coding: utf-8

# In[2]:


import pandas as pd
data=pd.read_csv("train.csv",index_col='id', parse_dates=True)
data.head()


# In[3]:


data.corr()


# In[6]:


data['vendor_id']


# In[7]:


data.head()


# In[8]:


from math import *
def dis(x1,x2,y1,y2):
    x=pow((x2-x1),2)
    y=pow((y2-y1),2)
    return sqrt(x+y)


# In[38]:


data.pop('distance')


# In[ ]:



data['distance']=data.apply(lambda row:dis(row.pickup_longitude,row.dropoff_longitude,row.pickup_latitude,row.dropoff_latitude),axis=1)


# In[ ]:


data.insert(8,'distance',1)


# In[ ]:


data.head()


# In[ ]:


data.corr()


# In[ ]:


data.iloc[1,1][0:4]


# In[ ]:


data.insert(3,'month',1)


# In[ ]:


test.pop('distance')


# In[ ]:


data['month']=data.apply(lambda row:row.pickup_datetime[5:7],axis=1)


# In[ ]:


data.head()


# In[ ]:


data.head()


# In[ ]:


data['date']=data.apply(lambda row:row.pickup_datetime[8:10],axis=1)


# In[ ]:


data.insert(5,'hours',1)


# In[ ]:


data['hours']=data.apply(lambda row:row.pickup_datetime[11:13],axis=1)


# In[ ]:


data


# In[87]:


data.corr()


# In[88]:


col=['month','date','hours','distance']
x_train=data[col]
y_train=data['trip_duration']
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train, y_train)


# In[89]:


print(linreg.intercept_)
print(linreg.coef_)


# In[90]:


test=pd.read_csv("test.csv",index_col='id', parse_dates=True)


# In[91]:


test.head()


# In[103]:


test.insert(7,'distance',1)


# In[104]:


test['distance']=test.apply(lambda row:dis(row.pickup_longitude,row.dropoff_longitude,row.pickup_latitude,row.dropoff_latitude),axis=1)


# In[106]:


#test.insert(2,'month',1)
test['month']=test.apply(lambda row:row.pickup_datetime[5:7],axis=1)


# In[111]:


#test.insert(3,'date',1)
test['date']=test.apply(lambda row:row.pickup_datetime[8:10],axis=1)


# In[112]:


#test.insert(4,'hours',1)
test['hours']=test.apply(lambda row:row.pickup_datetime[11:13],axis=1)


# In[113]:


cols=['month','date','hours','distance']
x_test=test[cols]
test


# In[116]:


y_pred = linreg.predict(x_test)
y_pred


# In[119]:


test.insert(12,'trip_duration',1)
test["trip_duration"]=y_pred


# In[120]:


test.head()


# In[122]:


c=['trip_duration']
ans=test[c]
ans


# In[133]:


ans.to_csv('e:/machine/ans.csv')


# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

