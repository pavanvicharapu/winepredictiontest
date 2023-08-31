#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('C:\\Users\\91994\\OneDrive\\Desktop\\winetest.csv')
df.head()


# In[9]:


df.info()


# In[12]:


df.head()

df.describe()
# In[13]:


df.isnull().sum()


# In[14]:


df.tail()


# In[15]:


df.describe()


# In[17]:


df.shape


# In[18]:


duplicate=df.duplicated()
print(duplicate.sum())
df[duplicate]


# In[19]:


print(df.quality.value_counts())


# In[20]:


sns.countplot(df['quality'])
plt.grid()
plt.show()


# In[21]:


df.corr()


# In[23]:


corr=df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True,cmap='coolwarm')


# In[24]:


target_name='quality'
y=df[target_name]
X=df.drop(target_name,axis=1)


# In[25]:


X.head()


# In[26]:


X.shape


# In[27]:


y.shape


# In[28]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_res=sc.fit_transform(X)


# In[29]:


X.head()


# In[30]:


x_res


# In[31]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif["vif"]=[variance_inflation_factor(x_res,i) for i in range(x_res.shape[1])]
vif["Features"]=X.columns
vif


# In[32]:


x_res.shape


# In[33]:


X1=X.drop(['residual sugar','density'],axis=1)
X1.shape


# In[36]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X1)
rescaledX=scaler.transform(X1)


# In[37]:


rescaledX.shape


# In[38]:


y.value_counts()


# In[39]:


sns.countplot(df['quality'])
plt.grid()
plt.show()


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(rescaledX,y,test_size=0.2,random_state=7)


# In[41]:


X_train.shape,y_train.shape


# In[42]:


X_test.shape,y_test.shape


# In[43]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_train_pred=dt.predict(X_train)
dt_test_pred=dt.predict(X_test)


# In[44]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[45]:


print('Train Accuracy:',accuracy_score(y_train,dt_train_pred)*100)


# In[46]:


print("Accuracy score",accuracy_score(y_test,dt_test_pred)*100)


# In[47]:


print(confusion_matrix(y_test,dt_test_pred))


# In[48]:


print(classification_report(y_test,dt_test_pred,digits=4))


# In[49]:


from sklearn.metrics import precision_score,recall_score,f1_score,classification_report,confusion_matrix
print("precision Score of macro is:",round(precision_score(y_test,dt_test_pred,average='macro')*100,2))
print("precision Score of micro is:",round(precision_score(y_test,dt_test_pred,average='micro')*100,2))
print("precision Score of weighted is:",round(precision_score(y_test,dt_test_pred,average='weighted')*100,2))


# In[50]:


print('f1_score of macro:',round(f1_score(y_test,dt_test_pred,average='macro')*100,2))
print('f1_score of micro:',round(f1_score(y_test,dt_test_pred,average='micro')*100,2))
print('f1_score of weighted:',round(f1_score(y_test,dt_test_pred,average='weighted')*100,2))


# In[51]:


df.head()


# In[59]:


input_data=(7.0,0.27,0.36,0.045,0.17,3.0,5.0,0.87,0.54,1.32)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=dt.predict(input_data_reshaped)
print(prediction)


# In[55]:


from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(random_state=1)
RFC.fit(X_train,y_train)
RFC_train_pred=RFC.predict(X_train)
RFC_train_pred=RFC.predict(X_test)


# In[60]:


print('Train Accuracy:',accuracy_score(y_train,RFC_train_pred)*100)


# In[61]:


print(confusion_matrix(y_test,RFC_test_pred))


# In[ ]:




