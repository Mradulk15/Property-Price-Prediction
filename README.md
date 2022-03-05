# Property-Price-Prediction
#!/usr/bin/env python
# coding: utf-8

# # Property Price Prediction System using Multiple Regression Models

# # Problem Statement

# # Objective
# ## 1. Predict the property price.
# ## 2. 

# # 
# 
# # STEP 1 : Importing the required Packages into our Python Environment

# In[275]:


#import required libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import statistics as sts
import math as mt


# # STEP 2 : Importing Data and EDA
# ## Importing the property prediction data and do some EDA on it
# ## First, Let’s import the data and have a look to see what kind of data we are dealing with:

# # 
# 
# ### Data Description:
# 
# ### Train.csv - 29451 rows x 12 columns
# ### Test.csv - 68720 rows x 11 columns
# 

# # Attributes Description:

# In[276]:


from IPython import display
display.Image("Attributes.png")


# In[277]:


#import Data
House_train_data = pd.read_csv('House Price train.csv')


# In[278]:


House_train_data


# In[279]:


#get some information about our Data-Set
House_train_data.shape


# In[280]:


House_test_data = pd.read_csv('House Price test.csv')


# In[281]:


House_test_data


# In[282]:


House_test_data.shape


# In[ ]:





# In[283]:


House_train_data.info()


# In[ ]:





# In[284]:


House_test_data.info()


# In[ ]:





# In[285]:


House_train_data.describe()


# In[286]:


House_train_data.describe(include = "object")


# In[ ]:





# In[287]:


print("Total no of unique values available in POSTED_BY =" ,House_train_data["POSTED_BY"].nunique())
print("Uniques Data  that are available in 1st feature are = ",House_train_data["POSTED_BY"].unique())
print("Number of entries of Unique value available in POSTED_BY =" ,House_train_data["POSTED_BY"].value_counts())
type(House_train_data["POSTED_BY"])


# In[ ]:





# In[288]:


House_train_data["POSTED_BY"] = House_train_data["POSTED_BY"].replace(to_replace=['Owner', 'Dealer','Builder'], value=[0,1,2])
House_test_data["POSTED_BY"]  = House_test_data["POSTED_BY"].replace(to_replace=['Owner', 'Dealer','Builder'], value=[0,1,2])


# In[289]:


print("Total no of unique value available in UNDER_CONSTRUCTION =" ,House_train_data["UNDER_CONSTRUCTION"].nunique())
print("Uniques Data that are available in UNDER_CONSTRUCTION are = ",House_train_data["UNDER_CONSTRUCTION"].unique())
print("Number of entries of Unique value available in UNDER_CONSTRUCTION =" ,House_train_data["UNDER_CONSTRUCTION"].value_counts())
type(House_train_data["UNDER_CONSTRUCTION"])


# In[290]:


print("Total no of unique value available in RERA =" ,House_train_data["RERA"].nunique())
print("Uniques Data that are available in RERA are = ",House_train_data["RERA"].unique())
print("Number of entries of Unique value available in RERA =" ,House_train_data["RERA"].value_counts())
type(House_train_data["RERA"])


# In[291]:


print("Total no of unique value available in BHK_NO.  =" ,House_train_data["BHK_NO."].nunique())
print("Uniques Data that are available in BHK_NO. are = ",House_train_data["BHK_NO."].unique())
print("Number of entries of Unique value available in BHK_NO. =" ,House_train_data["BHK_NO."].value_counts())
type(House_train_data["BHK_NO."])


# In[292]:


print("Total no of unique value available in BHK_OR_RK  =" ,House_train_data["BHK_OR_RK"].nunique())
print("Uniques Data that are available in BHK_OR_RK are = ",House_train_data["BHK_OR_RK"].unique())
print("Number of entries of Unique value available in BHK_OR_RK =" ,House_train_data["BHK_OR_RK"].value_counts())
type(House_train_data["BHK_OR_RK"])


# ## Convertion of the Unique data into binary type for the required calculations

# In[293]:


House_train_data["BHK_OR_RK"] = House_train_data["BHK_OR_RK"].replace(to_replace=('BHK', 'RK'),value=[1,0])


# In[294]:


print("Total no of unique value available in BHK_OR_RK  =" ,House_train_data["BHK_OR_RK"].nunique())
print("Uniques Data are that available in BHK_OR_RK are = ",House_train_data["BHK_OR_RK"].unique())
print("Number of entries of Unique value available in BHK_OR_RK =" ,House_train_data["BHK_OR_RK"].value_counts())
type(House_train_data["BHK_OR_RK"])


# In[295]:


print("Total no of unique value available in SQUARE_FT  =" ,House_train_data["SQUARE_FT"].nunique())
print("Uniques Data that are available in SQUARE_FT are = ",House_train_data["SQUARE_FT"].unique())
print("Number of entries of Unique value available in SQUARE_FT =" ,House_train_data["SQUARE_FT"].value_counts())
type(House_train_data["SQUARE_FT"])


# In[296]:


print("Total no of unique value available in READY_TO_MOVE  =" ,House_train_data["READY_TO_MOVE"].nunique())
print("Uniques Data that are available in READY_TO_MOVE are = ",House_train_data["READY_TO_MOVE"].unique())
print("Number of entries of Unique value available in READY_TO_MOVE =" ,House_train_data["READY_TO_MOVE"].value_counts())
type(House_train_data["READY_TO_MOVE"])


# In[297]:


print("Total no of unique value available in RESALE  =" ,House_train_data["RESALE"].nunique())
print("Uniques Data that are available in RESALE are = ",House_train_data["RESALE"].unique())
print("Number of entries of Unique value available in RESALE =" ,House_train_data["RESALE"].value_counts())


# In[298]:


print("Total no of unique value available in ADDRESS  =" ,House_train_data["ADDRESS"].nunique())
print("Uniques Data that are available in ADDRESS are = ",House_train_data["ADDRESS"].unique())
print("Number of entries of Unique value available in ADDRESS =" ,House_train_data["ADDRESS"].value_counts())
type(House_train_data["ADDRESS"])


# In[299]:


print("Total no of Unique value available in LONGITUDE  =" ,House_train_data["LONGITUDE"].nunique())
print("Uniques Data that are available in LONGITUDE are = ",House_train_data["LONGITUDE"].unique())
print("Number of entries of Unique value available in LONGITUDE =" ,House_train_data["LONGITUDE"].value_counts())
type(House_train_data["LONGITUDE"])


# In[300]:


print("Total no of Unique value available in LATITUDE  =" ,House_train_data["LATITUDE"].nunique())
print("Uniques Data that are available in LATITUDE are = ",House_train_data["LATITUDE"].unique())
print("Number of entries of Unique value available in LATITUDE =" ,House_train_data["LATITUDE"].value_counts())
type(House_train_data["LATITUDE"])


# In[301]:


print("Total no of Unique value available in TARGET(PRICE_IN_LACS)  =" ,House_train_data["TARGET(PRICE_IN_LACS)"].nunique())
print("Uniques Data that are available in TARGET(PRICE_IN_LACS) are = ",House_train_data["TARGET(PRICE_IN_LACS)"].unique())
print("Number of entries of Unique value available in TARGET(PRICE_IN_LACS) =" ,House_train_data["TARGET(PRICE_IN_LACS)"].value_counts())
type(House_train_data["TARGET(PRICE_IN_LACS)"])


# # DATA VISUALISATION

# In[302]:


plt.rcParams['axes.facecolor'] = "yellow"
sns.displot(House_train_data["TARGET(PRICE_IN_LACS)"])


# In[ ]:





# In[303]:


plt.figure(figsize=(12,8))
plt.rcParams['axes.facecolor'] = "gray"
sns.scatterplot(x="LONGITUDE", y="LATITUDE", data=House_train_data, hue="TARGET(PRICE_IN_LACS)",palette = 'rainbow')


# In[ ]:





# ### Renaming the last column for our assistance

# In[304]:


House_train_data = House_train_data.rename(columns = {'TARGET(PRICE_IN_LACS)' : 'TARGET'})


# In[ ]:





# #    Removing the Outliers from Test Data
# 
# ## Detecting outliers using quantile ranges
# 
# ## What Are Quantiles?
# ### Quantiles are very easy to understand. Let’s say we have a series of 20 numbers. We can sort the numbers from lowest to highest. We can then group these points into quantiles, which are identified by cut points in the sorted data that describe the point below which X% of data falls.
#  ### Note that quantiles are generally expressed as a fraction (from 0 to 1). They correspond exactly to percentiles, which range from 0 to 100.

# In[305]:


Quant1= House_train_data.quantile(.25)
Quant2= House_train_data.median()
Quant3= House_train_data.quantile(.85)
IQR = Quant3 - Quant1
QMIN = Quant1 - 1.5*IQR
QMAX = Quant3 + 1.5*IQR


# In[306]:


print("Quantile 1 = \n",Quant1)
print("\n\nQuantile 2 = \n ",Quant2)
print("\n\nQuantile 3 = \n",Quant3)
print("\n\nInter Quantile range = \n",IQR)
print("\n\nQ-MIN = \n",QMIN)
print("\n\nQ-MAX = \n",QMAX)


# ## -Quartiles, which divide the data into groups. The first quartile represents the data points that fall in the lowest 30%, the second quartile points fall between 30% and 50%, and so forth.
# 
# ## -Interquartile range, or IQR, which defines the range covered by 2nd and 3rd quartiles.

# In[ ]:





# In[307]:


House_train_data.head()


# In[ ]:





# In[308]:


House_train_data = House_train_data[(House_train_data.TARGET>QMIN.TARGET)&(House_train_data.TARGET<QMAX.TARGET)]


# In[309]:


House_train_data = House_train_data[(House_train_data.SQUARE_FT>QMIN.SQUARE_FT)&(House_train_data.SQUARE_FT<QMAX.SQUARE_FT)]


# In[310]:


House_train_data = House_train_data[(House_train_data.LATITUDE>QMIN.LATITUDE)&(House_train_data.LATITUDE<QMAX.LATITUDE)]


# In[311]:


House_train_data = House_train_data[(House_train_data.LONGITUDE>QMIN.LONGITUDE)&(House_train_data.LONGITUDE<QMAX.LATITUDE)]


# In[312]:


House_train_data.shape


# In[ ]:





# # Analysing the data through data Visualisation after the removal of outliers

# In[ ]:





# In[313]:


plt.rcParams['axes.facecolor'] = "pink"
sns.displot(House_train_data["TARGET"])


# In[ ]:





# In[314]:


plt.figure(figsize=(12,8))
plt.rcParams['axes.facecolor'] = "RosyBrown"
sns.scatterplot(x="LONGITUDE", y="LATITUDE", data=House_train_data, hue="TARGET",palette = 'rainbow')


# In[ ]:





# ## Distribution Plot

# In[315]:


sns.displot(House_train_data['SQUARE_FT'])


# In[ ]:





# # Heat Map
# ## Heatmaps are very useful to find relations between two variables in a dataset. Heatmap can be easily produced using the ‘heatmap’ function provided by the seaborn package in python.

# In[316]:


plt.figure(figsize = (10,6))
sns.heatmap(House_train_data.corr(), annot = True ,cmap='twilight')
plt.title(' Heat for correlation \n')
plt.show


# ### - From the Heatmap graph we can clearly see that there is a strong negative relationship between 'READY_TO_MOVE' and 'UNDER_CONSTRUCTION' which leads to multicollinearity. So we will remove one of them.
# ###  1. Since we have location based on the Longitude and Lattitude i.e why ADDRESS will be a wastaged attribute so now we can remove ADDRESS attribute too.
# ###  2. From the Heatmap we can clearly see that attribute 'BHK_OR_RK' isn't affecting our Target price so there is not significant to take it as a parameter, so we will remove this also.

# # Now we will remove "READY_TO_MOVE","ADDRESS" and "BHK_OR_RK" from both Train and test data.

# In[317]:


House_train_data = House_train_data.drop(['READY_TO_MOVE'],axis = 1, inplace=False)
House_train_data = House_train_data.drop(['ADDRESS'],axis = 1, inplace=False)
House_train_data = House_train_data.drop(['BHK_OR_RK'],axis = 1, inplace=False)

House_test_data = House_test_data.drop(['READY_TO_MOVE'],axis = 1, inplace=False)
House_test_data = House_test_data.drop(['ADDRESS'],axis = 1, inplace=False)
House_test_data = House_test_data.drop(['BHK_OR_RK'],axis = 1, inplace=False)


# In[318]:


House_train_data.head()


# In[319]:


House_test_data.head()


# In[320]:


House_train_data.describe()


# In[321]:


House_test_data.describe()


# In[ ]:





# # MODELING

# ## Applying Linear Regression Model on the House Price Train data File

# In[ ]:





# In[322]:


X_train = House_train_data.drop('TARGET', axis = 1).values
Y_train = House_train_data['TARGET'].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train , Y_test = train_test_split(X_train, Y_train ,test_size=0.1, random_state=0)


# In[323]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[324]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[325]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()


# In[326]:


LR.fit(X_train,Y_train)


# In[327]:


print("The intercept value =", LR.intercept_)


# In[328]:


print("The coeffiecients are Theat 0 ,Theta 1, Theta 2 , Theta 3 =", LR.coef_)


# In[329]:


Predict = LR.predict(X_test)


# In[330]:


Predict


# In[331]:


from sklearn.metrics import r2_score


# In[332]:


r2_score(Y_test,Predict)


# In[333]:


from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error,r2_score


# In[334]:


Y_predict = LR.predict(X_test).reshape(X_test.shape[0])


predict_df = pd.DataFrame({ 'Actual_value': Y_test, 'Predicted value': Y_predict})
print(predict_df)


# ## MAE

# In[335]:


mean_absolute_error(y_true=predict_df['Actual_value'], y_pred=predict_df['Predicted value'])


# ## MSE

# In[336]:


print(mean_squared_error(y_true=predict_df['Actual_value'], y_pred=predict_df['Predicted value']))


# ## RMSE

# In[337]:


print(np.sqrt(mean_squared_error(y_true=predict_df['Actual_value'], y_pred=predict_df['Predicted value'])))


# In[338]:


from sklearn.metrics import explained_variance_score
explained_variance_score(Y_test,Predict)


# In[339]:


R_Squared = r2_score(Y_test,Predict)
R_Squared


# # Now Applying Linear Regression Model on the House Price Test data File

# In[340]:


House_test_data1 = House_test_data


# In[341]:


Quant11 = House_test_data1.quantile(.35)
Quant21 = House_test_data1.median()
Quant31 = House_test_data1.quantile(.60)
IQR1 = Q31 - Q11
QMIN1 = Quant11 - 1.5*IQR1
QMAX1 = Quant31 + 1.5*IQR1


# In[342]:


House_test_data1 = House_test_data1[(House_test_data1.SQUARE_FT>QMIN1.SQUARE_FT)&(House_test_data1.SQUARE_FT<QMAX1.SQUARE_FT)]
House_test_data1.shape


# In[343]:


House_test_data1 = House_test_data1[(House_test_data1.LONGITUDE>QMIN1.LONGITUDE)&(House_test_data1.LONGITUDE<QMAX1.LONGITUDE)]
House_test_data1.shape


# In[344]:


House_test_data1 = House_test_data1[(House_test_data1.LATITUDE>QMIN1.LATITUDE)&(House_test_data1.LATITUDE<QMAX1.LATITUDE)]
House_test_data1.shape


# In[345]:


X2_test = House_test_data1.values
X2_test


# In[346]:


X2_test = scaler.transform(X2_test)


# In[347]:


Pred1 = LR.predict(X2_test)


# In[348]:


Pred1


# In[349]:


Testdata = pd.DataFrame({'Target':Pred1})
Testdata


# In[350]:


Testdata.plot()


# In[ ]:





# # Applying next model i.e Decision Tree on the House Price Train data File

# In[ ]:





# In[351]:


#Importing Decision tree
from sklearn.tree import DecisionTreeRegressor


# In[352]:


decision_tree_model= DecisionTreeRegressor()


# In[353]:


decision_tree_model.fit(X_train,Y_train)
y_prediction_tree=decision_tree_model.predict(X_train)


# In[354]:


y_prediction_tree_test=decision_tree_model.predict(X_test)
y_prediction_tree_test


# ## RMSE and R2_square

# In[355]:


print("RMSE:",np.sqrt(mean_squared_error(Y_train,y_prediction_tree)))
print("R_square:",r2_score(Y_train,y_prediction_tree))


# In[ ]:





# # Now Applying Decision Tree Model on the House Price Test data File

# In[356]:


Pred2 = decision_tree_model.predict(X2_test)


# In[357]:


Pred2


# In[358]:


Testdata1 = pd.DataFrame({'Target1':Pred2})
Testdata1


# In[359]:


Testdata1.plot()


# In[ ]:





# # Model Evaluation

# In[360]:


print("Linear Regression - R_Square: ",r2_score(Y_test,Predict))
print("Decison Tree - R_Square:",r2_score(Y_train,y_prediction_tree))


# In[361]:


print("Linear Regression - RMSE",np.sqrt(mean_squared_error(y_true=predict_df['Actual_value'], y_pred=predict_df['Predicted value'])))
print("Decison Tree - RMSE:",np.sqrt(mean_squared_error(Y_train,y_prediction_tree)))


# In[ ]:





# # Conclusion

# In[ ]:




