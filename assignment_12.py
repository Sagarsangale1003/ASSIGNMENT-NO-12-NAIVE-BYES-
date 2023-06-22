#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# In[2]:


test=pd.read_csv(r"C:\Users\sagar\Desktop\sagar\sagar_assignment\Assignment12\SalaryData_Test.csv")
test.head()


# In[3]:


train=pd.read_csv(r"C:\Users\sagar\Desktop\sagar\sagar_assignment\Assignment12\SalaryData_Train.csv")
train.head()


# In[4]:


test.shape,train.shape


# In[5]:


test.info()


# In[6]:


train.info()


# In[8]:


test.columns
train.columns


# In[9]:


sns.pairplot(train)


# In[10]:


train.head(2)


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


labelencoder=LabelEncoder()


# In[13]:


text=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# In[14]:


for i in text:
    train[i]=labelencoder.fit_transform(train[i])
    test[i]=labelencoder.fit_transform(test[i])


# In[15]:


train_x=train.iloc[:,0:13]
train_y=train.iloc[:,13]


# In[16]:


train_x


# In[17]:


train_y


# In[18]:


test_x=test.iloc[:,0:13]
test_y=test.iloc[:,13]


# In[19]:


test_x


# In[20]:


test_y


# In[21]:


from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as NB


# In[22]:


mb=MB()
train_pred=mb.fit(train_x,train_y).predict(train_x)
train_pred


# In[23]:


test_pred=mb.fit(train_x,train_y).predict(test_x)
test_pred


# In[24]:


accuracy=np.mean(train_pred==train_y)
accuracy


# In[25]:


gb=NB()
train_pred_gb=gb.fit(train_x,train_y).predict(train_x)
train_pred_gb


# In[26]:


test_pred_gb=gb.fit(train_x,train_y).predict(test_x)
test_pred_gb


# In[27]:


np.mean(train_pred_gb==train_y)


# Accuarcy For Gaussian is best as compare to Multinomial (its number data)

# In[ ]:




