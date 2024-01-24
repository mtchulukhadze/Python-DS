#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


df = sns.load_dataset('iris')
df.head()


# In[15]:


df['species'].value_counts()


# In[16]:


X = df[['sepal_length', 'sepal_width']]
y = df['species']
# df['species'].value_counts()
y = y.map({'setosa': 0, 'versicolor': 1, 'virginica': 2})


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 42)


# In[18]:


model = LogisticRegression()


# In[19]:


model.fit(X_train, y_train)


# In[20]:


y_pred = model.predict(X_test)


# In[31]:


accurancy = metrics.accuracy_score(y_test, y_pred)
classification = classification_report(y_test, y_pred)


# In[30]:


data = {'sepal_length': [5.5], 'sepal_width': [3.0]}
df = pd.DataFrame(data)
new_predictions = model.predict(df)

predicted_species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
predicted_species_name = [predicted_species[p] for p in new_predictions]
predicted_species_name


# In[ ]:




