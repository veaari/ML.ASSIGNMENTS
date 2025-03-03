#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")
iris = pd.read_csv("iris.csv")
iris 


# In[2]:


iris.info()


# In[3]:


iris.describe()


# In[4]:


iris.isnull().sum()


# In[5]:


iris['variety'].value_counts()


# In[6]:


iris.head()


# In[7]:


print(iris.isnull().sum())  


# In[8]:


print(iris.columns)
from mlxtend.preprocessing import TransactionEncoder
transactions = iris.groupby(['variety']).apply(list)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
iris_encoded = pd.DataFrame(te_ary, columns=te.columns_)
iris_encoded


# In[9]:


print(iris.columns)


# In[10]:


iris.columns = iris.columns.str.strip()


# #### Observations
# - There are no null values
# - Found 4 numerical values
# - One Categorical value(Target)
# - Here the numerical columns are independent and Target is the
#    independent
# - There are 3 types of Flowers among the dataset
# - They are Setosa , Versicolor and Virginica
# - There are no duplicate rows
# - Equal amount flower quantity in dataset

# In[11]:


print(iris['petal.length'].mean().round(2))
print(iris['petal.width'].mean())
print(iris['sepal.length'].mean())
print(iris['sepal.width'].mean())


# In[12]:


duplicate_values=iris.duplicated().sum()
duplicate_values


# In[13]:


duplicated_rows = iris[iris.duplicated()]
duplicated_rows


# In[14]:


iris1 = iris.drop_duplicates()
print(iris1.head())


# In[15]:


labelencoder=LabelEncoder()
iris.iloc[:,-1]=labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[16]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
iris.iloc[:,-1]=labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[17]:


iris.info()


# ### Observations
# - for y column we can use lebel encoding 
# - for x columns we use one hot encoding and also exceptionally we can     use label encodong
# - the taget columns(variety ) is still object type . it needs to be        converted to numeric(int)

# In[18]:


iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[19]:


iris.head(4)


# In[27]:


X=iris.iloc[:,0:4]
Y=iris['variety']


# In[21]:


Y


# In[22]:


X


# In[23]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
print(x_train.head(23))


# In[24]:


x_train


# In[25]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', max_depth=None)
model.fit(x_train, y_train)


# In[28]:


import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(dpi=1200)
tree.plot_tree(model)
plt.show()


# In[30]:


fn = ['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)']
cn=['serosa','versecolor','virginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names =fn,class_names=cn,filled=True)


# In[31]:


preds = model.predict(x_test)
preds


# In[36]:


from sklearn.metrics import classification_report
print(classification_report(y_test, preds))


# In[ ]:




