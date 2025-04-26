#!/usr/bin/env python
# coding: utf-8

# # S SAI VARSHINI 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[2]:


Titanic = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:


Titanic.columns


# # Data Dictionary
# 
# survived: 0 = No, 1 = Yes
# pclass:Ticket Class 1=1st, 2=2nd, 3=3rd
# SibSp: # of Sibilings/Spouses aboard the titanic (0 mentions neither have have Spuose nor Sibilings)
# parch: # of parents/children aboard the titanic
# ticket: Ticket number
# Cabin: Cabin Number
# embarked: Port of Embarkation C= Cherboug, S= Southamptom, Q = Queenstown

# In[4]:


Titanic.head()


# In[5]:


Titanic.tail()


# In[6]:


Titanic.shape


# In[7]:


Titanic.info()


# In[8]:


Titanic.describe()


# In[9]:


Titanic.describe(include=['O'])


# # Data Analyze by pivoting features¶

# In[10]:


Titanic[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[11]:


Titanic[["Sex" , "Survived"]].groupby(["Sex"] , as_index = False).mean().sort_values(by="Survived" , ascending = False)


# In[12]:


Titanic[["Parch" , "Survived"]].groupby(["Parch"] , as_index = False) .mean().sort_values(by="Survived" , ascending = False)


# In[13]:


Titanic[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[14]:


Titanic[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# # Data Analyze by visualization Method

# In[15]:


Titanic.hist(bins=10,figsize=(9,7),grid=False)


# In[16]:


print("No of Passengers in original data:  " , str(len(Titanic.index)))


# In[17]:


sns.countplot(x="Survived" , data=Titanic , color="orange")


# In[18]:


sns.countplot(x="Survived", hue="Parch",data=Titanic)


# In[19]:


sns.countplot(x="Survived" ,hue="Sex" , data=Titanic )


# In[20]:


sns.countplot(x="Survived" ,hue="Pclass" , data=Titanic )


# In[21]:


sns.countplot(x="Survived" ,hue="Embarked" , data=Titanic )


# In[22]:


sns.countplot(x="Survived", hue="SibSp",data=Titanic)


# In[23]:


Titanic["Parch"].plot.hist( figsize=(5,4))


# In[24]:


Titanic["SibSp"].plot.hist()


# # Data Cleaning Or Filling The Missing Values

# In[25]:


Titanic.isnull()


# In[26]:


Titanic.isnull().sum()


# In[27]:


null_var = Titanic.isnull().sum()/Titanic.shape[0] *100
null_var


# In[28]:


drop_column = null_var[null_var >20].keys()
drop_column


# In[29]:


N_Titanic_datA = Titanic.drop(columns = drop_column)


# In[30]:


Titanic_copy = Titanic.copy()
Titanic_copy2 = Titanic.copy()
Titanic_Deep = Titanic_copy.copy()


# In[31]:


N_Titanic_datA.isnull().sum()/Titanic_Deep.shape[0] *100


# In[32]:


N_Titanic_datAA = N_Titanic_datA.dropna()


# In[33]:


Categorical_Values = N_Titanic_datAA.select_dtypes(include=["object"]).columns
Categorical_Values_test = test.select_dtypes(include=["object"]).columns


# In[34]:


Numarical_Values = N_Titanic_datAA.select_dtypes(include=['int64','float64']).columns
Numarical_Values_test = test.select_dtypes(include=['int64','float64']).columns


# In[35]:


test.shape


# In[36]:


sns.countplot(x="Survived", hue='Age',data=Titanic)


# In[37]:


def cat_var_dist(var):
    return pd.concat([Titanic_Deep[var].value_counts()/Titanic_Deep.shape[0] * 100, 
          N_Titanic_datAA[var].value_counts()/N_Titanic_datAA.shape[0] * 100], axis=1,
         keys=[var+'_org', var+'clean'])


# In[38]:


cat_var_dist("Ticket")


# In[39]:


Imputer_mean = SimpleImputer(strategy='mean')


# In[40]:


Imputer_mean.fit(Titanic_Deep[Numarical_Values])


# In[41]:


Imputer_mean.statistics_


# In[42]:


Imputer_mean.transform(Titanic_Deep[Numarical_Values])


# In[43]:


Titanic_Deep[Numarical_Values] = Imputer_mean.transform(Titanic_Deep[Numarical_Values])
nnnn = Titanic_Deep[Numarical_Values]


# In[44]:


Titanic_Deep[Numarical_Values].isnull().sum()


# In[45]:


Imputer_mean = SimpleImputer(strategy='most_frequent')


# In[46]:


Titanic_Deep[Categorical_Values] = Imputer_mean.fit_transform(Titanic_Deep[Categorical_Values])


# In[47]:


Titanic_Deep[Categorical_Values].isnull().sum()


# In[48]:


New_Titanic_datA = pd.concat([Titanic_Deep[Numarical_Values] , Titanic_Deep[Categorical_Values]] , axis=1)


# In[49]:


New_Titanic_datA.isnull().sum()


# In[50]:


skip_column = null_var[null_var >20].keys()
skip_column


# In[51]:


Nn_Titanic_datA = Titanic_copy.drop(columns = skip_column)


# In[52]:


Titanic_mean = Nn_Titanic_datA.fillna(Nn_Titanic_datA.mean())
Titanic_mean = Titanic_mean.dropna()


# In[53]:


print(Titanic_mean.isnull().sum())


# In[54]:


Titanic_median = Nn_Titanic_datA.fillna(Nn_Titanic_datA.median())
test_median =  test.fillna(test.median())
Titanic_median = Titanic_median.dropna()
Titanic_median.isnull().sum()


# In[55]:


print("*"*30 , "Data Cleaning Using Different Method" , "*"*30)
print("*"*30 , "Simple Row Delete Mehtod" , "*"*30)
print(N_Titanic_datAA.isnull().sum())
print("*"*30 , "SimpleImputer Method" , "*"*30)
print(New_Titanic_datA.isnull().sum())
print("*"*30 , "Median" , "*"*30)
print(Titanic_median.isnull().sum())
print("*"*30 , "Mean" , "*"*30)
print(Titanic_mean.isnull().sum())


# # Finding categorical feature, Training Testing, and Accuracy Using Three Different Methods

# In[56]:


N_Titanic_datAA.tail()


# In[57]:


sex = pd.get_dummies(N_Titanic_datAA["Sex"] , drop_first=True)
sexx =  pd.get_dummies(test_median["Sex"] , drop_first=True)


# In[58]:


pclass = pd.get_dummies(N_Titanic_datAA["Pclass"] , drop_first=True)
pclasss = pd.get_dummies(test_median["Pclass"] , drop_first=True)


# In[59]:


embarked = pd.get_dummies(N_Titanic_datAA["Embarked"] , drop_first=True)
embarkedd = pd.get_dummies(test_median["Embarked"] , drop_first=True)


# In[60]:


N_Titanic_datAA_copy = N_Titanic_datAA.copy()


# In[61]:


N_Titanic_datAA_copy.drop(['Embarked', 'Pclass' ,"Sex" , "Ticket" , "Name"], axis=1 , inplace=True)


# In[62]:


test_median.drop(['Embarked', 'Pclass' ,"Sex" , "Ticket" , "Name"], axis=1 , inplace=True)


# In[63]:


N_Titanic_datAA_copy = pd.concat([N_Titanic_datAA_copy ,sex ,pclass ,embarked] ,axis=1)
N_Titanic_datAA_copy.head()
test_median = pd.concat([test_median ,sexx ,pclasss ,embarkedd] ,axis=1)
test_median.head()


# In[64]:


test_median.drop(["Cabin"], axis=1 , inplace=True)


# In[65]:


test1= test_median.copy()


# In[66]:


test_median.head()


# # Training & Testing

# In[67]:


X = N_Titanic_datAA_copy.drop("Survived" , axis=1)
y = N_Titanic_datAA_copy["Survived"]


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[69]:


y.shape


# # Finding The Accuracy

# In[81]:


X_train = X_train.values  # Remove column names (converts DataFrame to ndarray)
knn.fit(X_train, y_train)


# In[82]:


import pandas as pd
import matplotlib.pyplot as plt
file_path = "train.csv"
df = pd.read_csv(file_path)
plt.hist(df["Survived"], bins=2, edgecolor="black", alpha=0.7)
plt.xticks([0, 1], ["Not Survived", "Survived"])
plt.xlabel("Survival Status")
plt.ylabel("Count")
plt.title("Histogram of Survived Column")
plt.show()


# In[83]:


survived_df = df[df["Survived"] == 1]
age_counts = survived_df["Age"].value_counts().dropna()
most_common_age = age_counts.idxmax()
most_common_age_count = age_counts.max()
print(f"The age that survived the most is {most_common_age} with {most_common_age_count} survivors.")


# In[84]:


plt.hist(df["Survived"], bins=4, color='green', edgecolor='black')
plt.show()


# In[85]:


df[df["Survived"] == 0]


# In[86]:


df[df["Age"].isnull()]["Pclass"].value_counts().idxmax()

