#!/usr/bin/env python
# coding: utf-8

# #  Dragon Real Estate Problem - Ml Project

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.head()


# In[5]:


# housing.drop([16:])


# In[6]:


housing.drop(housing.columns[[16,25]], axis = 1, inplace = True)


# In[7]:


housing.head()


# In[8]:


housing.drop(housing.iloc[:, 14:26], inplace = True, axis = 1)


# In[9]:


housing.drop(housing.iloc[:, 14:25], inplace = True, axis = 1)


# In[10]:


housing.head()


# In[11]:


housing.info()


# In[12]:


housing['CHAS'].value_counts()


# In[13]:


housing.describe()



# In[14]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


housing.hist(bins=50 , figsize=(20,15))


# # Train-Test splitting

# In[17]:


# for learning purpose
import numpy as np
def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices] , data.iloc[test_indices]    


# In[18]:


# train_set,test_set=split_train_test(housing,0.2)


# In[19]:


# print(f"Rows in train set:{len(train_set)}\nRows in train set:{len(test_set)}\n")


# In[20]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set:{len(train_set)}\nRows in train set:{len(test_set)}\n")


# In[21]:


#stratified suffling for better result of 'CHAS' becuase it is very important 

from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[22]:


strat_test_set['CHAS'].value_counts()


# In[23]:


strat_train_set['CHAS'].value_counts()


# # Looking for correlations

# In[24]:


corr_matrix=housing.corr()


# In[25]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[26]:


from pandas.plotting import scatter_matrix
attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[27]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# In[28]:


housing=strat_train_set.drop('MEDV',axis=1)
housing_lables=strat_train_set["MEDV"]


# # Scikit-learn Design
# 
# primilary three types of objects
# 1.Estimators - It estimates some parameter based on a dataset. Eg. Imputer. It has a fit method and transform method. fit method- fits the dataset and calculates internal parameters
# 
# 2. Transformers - transform method takes input and returns output based on the learnings from fit ( ) . It also has a convenience function called fit_transform ( ) which fits and then transforms . 
# 
# 3. Predictors - LinearRegression model is an example of predictor . fit ( ) and predict ( ) are two common functions . It also gives score ( ) function which will evaluate the predictions .

# # feature scaling
# Primarily , two types of feature scaling methods : 
# 1. Min - max scaling ( Normalization ) ( value - min ) / ( max - min ) Sklearn provides a class called MinMaxScaler for this 
# 2. Standardization
# ( value - mean ) / std Sklearn provides a class called StandardScaler

# # Creating Pipeline

# In[29]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])


# In[30]:


housing_num_tr=my_pipeline.fit_transform(housing)


# In[31]:


housing_num_tr


# # Selecting a desired model for real estate

# In[32]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_lables)


# In[33]:


some_data=housing.iloc[:5]


# In[34]:


some_lables=housing_lables.iloc[:5]


# In[35]:


prepared_data=my_pipeline.transform(some_data)


# In[36]:


model.predict(prepared_data)


# In[37]:


list(some_lables)


# # Evaluating Model

# In[38]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_lables,housing_predictions)
rmse=np.sqrt(mse)


# In[39]:


mse


# # Using better technique - Cross Validation

# In[40]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_lables,scoring='neg_mean_squared_error')
rmse_scores=np.sqrt(-scores)


# In[41]:


rmse_scores


# In[42]:


def print_scores(scores):
    print('scores :',scores)
    print('Mean :',scores.mean())
    print('Standard deviation :',scores.std())


# In[43]:


print_scores(rmse_scores)


# In[44]:


from joblib import dump
dump(model,'house.joblib')


# # Testing the model on test data
# 

# In[51]:


x_test=strat_test_set.drop("MEDV",axis=1)
y_test=strat_test_set["MEDV"].copy()
x_test_prepared=my_pipeline.transform(x_test)
final_predictions=model.predict(x_test_prepared)
final_mse=mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_predictions,list(y_test))



# In[46]:


final_rmse


# In[53]:


print(prepared_data[0])


# # Model Use

# In[54]:


from joblib import dump,load
import numpy as np
model=load('house.joblib')
input=np.array([[-0.43942006 , 3.12628155, -1.12165014 ,-0.27288841, -1.42262747, -0.24141041,
 -1.31238772 , 2.61111401, -1.0016859,  -0.5778192 , -0.97491834,  0.41164221,
 -0.86091034]])
model.predict(input)


# In[ ]:




