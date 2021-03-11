#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


# # Reading the data

# In[19]:


d=pd.read_csv('train.txt',sep=';',encoding='latin',header=None,names=['text','emotion'])


# In[20]:


d.head()


# In[21]:


d.shape


# # Visualizing the data

# In[24]:


d['emotion'].value_counts().plot(kind='bar',grid=True)


# In[25]:


d.head()


# In[26]:


d['text'].isnull().sum()


# In[28]:


d.loc[d['text']==' ']


# In[29]:


from sklearn.feature_extraction.text import CountVectorizer


# # Applying the CountVectorizer Method

# In[30]:


cv=CountVectorizer(max_features=1000,ngram_range=(1,1),max_df=0.3,min_df=2)


# In[32]:


x=d['text']
y=d['emotion']


# In[33]:


from sklearn.model_selection import train_test_split


# # Spliting the data

# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)


# In[50]:


x_train[0]


# In[36]:


x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)


# In[37]:


x_train


# # Model Building

# In[38]:


import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[40]:


from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# In[41]:


enc=LabelEncoder()


# In[42]:


y_train=enc.fit_transform(y_train)
y_test=enc.transform(y_test)


# In[43]:


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# In[44]:


y_train


# In[45]:


model=Sequential()
model.add(Dense(512,activation='relu',input_shape=(1000,)))
model.add(Dropout(0.3))
model.add(Dense(6,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# # Model Training

# In[46]:


model.fit(x_train,y_train,epochs=12,batch_size=32,validation_data=(x_test,y_test))


# In[47]:


results=pd.DataFrame(model.history.history)


# # Results

# In[48]:


results


# In[49]:


results.plot(figsize=(12,8))


# # Testing the model

# In[53]:


d['text'][2]


# In[54]:


input_string='im grabbing a minute to post i feel greedy wrong'


# In[56]:


text=cv.transform([input_string])


# In[57]:


text


# In[63]:


enc.inverse_transform(model.predict_classes(text))[0]


# In[ ]:




