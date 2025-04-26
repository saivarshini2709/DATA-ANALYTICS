#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries & functions
# 
# 
# 
# 

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


# ### Importing dataset

# In[4]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


dataset=pd.read_excel("/content/drive/My Drive/1_LiveProjects/Project1_Credit_Scoring/a_Dataset_CreditScoring.xlsx")


# ### Data preparation

# In[6]:


# shows count of rows and columns
dataset.shape


# In[7]:


#shows first few rows of the code
dataset.head()


# In[8]:


#dropping customer ID column from the dataset
dataset=dataset.drop('ID',axis=1)
dataset.shape


# In[9]:


# explore missing values
dataset.isna().sum()


# In[10]:


# filling missing values with mean
dataset=dataset.fillna(dataset.mean())


# In[11]:


# explore missing values post missing value fix
dataset.isna().sum()


# In[ ]:


# # count of good loans (0) and bad loans (1)
# dataset['TARGET'].value_counts()


# In[ ]:


# # data summary across 0 & 1
# dataset.groupby('TARGET').mean()


# ### Train Test Split

# In[12]:


y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:29].values


# In[14]:


# splitting dataset into training and test (in ratio 80:20)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0,
                                                    stratify=y)


# In[15]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[16]:


# Exporting Normalisation Coefficients for later use in prediction
import joblib
joblib.dump(sc, '/content/drive/My Drive/1_LiveProjects/Project1_Credit_Scoring/f2_Normalisation_CreditScoring')


# ### Risk Model building

# In[17]:


classifier =  LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[18]:


# Exporting Logistic Regression Classifier for later use in prediction

# import joblib
joblib.dump(classifier, '/content/drive/My Drive/1_LiveProjects/Project1_Credit_Scoring/f1_Classifier_CreditScoring')


# ### Model *performance*

# In[19]:


print(confusion_matrix(y_test,y_pred))


# In[20]:


print(accuracy_score(y_test, y_pred))


# ### Writing output file

# In[21]:


predictions = classifier.predict_proba(X_test)
predictions


# In[22]:


# writing model output file

df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns = ['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])

dfx=pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)

dfx.to_csv("/content/drive/My Drive/1_LiveProjects/Project1_Credit_Scoring/c1_Model_Prediction.xlsx", sep=',', encoding='UTF-8')

dfx.head()


# ### Coding ends here!
