#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import streamlit as st


# In[14]:


data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df


# In[15]:


x = df.drop(columns=['target'])
y = df['target']


# In[16]:


x_train, x_test, y_train, y_test, = train_test_split(x,y,test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)


# In[17]:


predictions = clf.predict(x_test)
accuracy = accuracy_score(y_test, predictions)


# In[22]:


data_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = clf.predict(data_input)
predicted_class = data.target_names[prediction[0]]

st.subheader("Prediction")
st.write(f"Predicted Class: **{predicted_class}**")

# if st.button("Show Decision Tree"):
    # tree_rules = export_text(clf, feature_names=data.feature_names)
    # st.text(tree_rules)


# In[21]:


import io
if st.button("Show Decision Tree Diagram"):
    fig, ax = plt.subplots(figsize=(12, 8))
    tree.plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True, ax=ax)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    st.image(buf, caption="Decision Tree Visualization", use_column_width=True)


# In[ ]:




