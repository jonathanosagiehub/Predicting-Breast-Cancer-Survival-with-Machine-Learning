#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Survival Prediction using Python

# ### I will start the task of breast cancer survival prediction by importing the necessary Python libraries and the dataset we need:

# In[17]:


import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv("BRCA.csv")
print(data.head())


# ### Let's check to see whether any of the columns in this dataset contain null values:

# In[2]:


print(data.isnull().sum())


# ### This dataset has some null values in each column, I will drop these null values:

# In[3]:


data = data.dropna()


# ### Now let’s have a look at the insights about the columns of this data:

# In[4]:


data.info()


# ### Breast cancer is mostly found in females, so let’s have a look at the Gender column to see how many females and males are there:

# In[5]:


print(data.Gender.value_counts())


# ### As expected, the proportion of females is more than males in the gender column. Now let’s have a look at the stage of tumour of the patients:

# In[6]:


# Tumour Stage
stage = data["Tumour_Stage"].value_counts()
transactions = stage.index
quantity = stage.values

figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Tumour Stages of Patients")
figure.show()


# ### So most of the patients are in the second stage. Now let’s have a look at the histology of breast cancer patients. (Histology is a description of a tumour based on how abnormal the cancer cells and tissue look under a microscope and how quickly cancer can grow and spread):

# In[7]:


# Histology
histology = data["Histology"].value_counts()
transactions = histology.index
quantity = histology.values
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Histology of Patients")
figure.show()


# ### Now let’s have a look at the values of ER status, PR status, and HER2 status of the patients:

# In[8]:


# ER status
print(data["ER status"].value_counts())
# PR status
print(data["PR status"].value_counts())
# HER2 status
print(data["HER2 status"].value_counts())


# ### Now let’s have a look at the type of surgeries done to the patients:

# In[9]:


# Surgery_type
surgery = data["Surgery_type"].value_counts()
transactions = surgery.index
quantity = surgery.values
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Type of Surgery of Patients")
figure.show()


# ### So we explored the data, the dataset has a lot of categorical features. To use this data to train a machine learning model, we need to transform the values of all the categorical columns. Here is how we can transform values of the categorical features:

# In[10]:


data["Tumour_Stage"] = data["Tumour_Stage"].map({"I": 1, "II": 2, "III": 3})
data["Histology"] = data["Histology"].map({"Infiltrating Ductal Carcinoma": 1, 
                                           "Infiltrating Lobular Carcinoma": 2, "Mucinous Carcinoma": 3})
data["ER status"] = data["ER status"].map({"Positive": 1})
data["PR status"] = data["PR status"].map({"Positive": 1})
data["HER2 status"] = data["HER2 status"].map({"Positive": 1, "Negative": 2})
data["Gender"] = data["Gender"].map({"MALE": 0, "FEMALE": 1})
data["Surgery_type"] = data["Surgery_type"].map({"Other": 1, "Modified Radical Mastectomy": 2, 
                                                 "Lumpectomy": 3, "Simple Mastectomy": 4})
print(data.head())


# # Breast Cancer Survival Prediction Model

# ### We can now move on to training a machine learning model to predict the survival of a breast cancer patient. Before training the model, we need to split the data into training and test set:

# In[13]:


# Splitting data
x = np.array(data[['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 
                   'Tumour_Stage', 'Histology', 'ER status', 'PR status', 
                   'HER2 status', 'Surgery_type']])
y = np.array(data[['Patient_Status']])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)


# ### Now here’s how we can train a machine learning model:

# In[18]:


model = SVC()
model.fit(xtrain, ytrain)


# ### Now let’s input all the features that we have used to train this machine learning model and predict whether a patient will survive from breast cancer or not:

# In[14]:


# Prediction
# features = [['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 'Tumour_Stage', 'Histology', 'ER status', 'PR status', 'HER2 status', 'Surgery_type']]
features = np.array([[36.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
print(model.predict(features))


# # Summary

# ### As the use of data in healthcare is very common today, we can use machine learning to predict whether a patient will survive a deadly disease like breast cancer or not.

# In[ ]:




