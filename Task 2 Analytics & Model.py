# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:52:30 2021

@author: Aben George
"""


# =============================================================================
# # DATA @ ANZ TASKS
# 
# 1) Using the same transaction dataset, identify the annual salary for each customer
# 
# 
# 2) Explore correlations between annual salary and various customer attributes (e.g. age)
#  those that you construct or derive yourself (e.g. those relating to purchasing behaviour)
# 
# 
# 3) Build a simple regression model to predict the annual salary for each customer 
# 
# 4)How accurate is your model? 
# 
# 5) build a decision-tree based model to predict salary.
# 
# 6)Does it perform better? How would you accurately test the performance of this model?
# #-----------------------------------------------------------------------------------------
# =============================================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import math
import numpy as np
from matplotlib import pyplot as plt
#Warning Issue Solution
pd.options.mode.chained_assignment = None  # default='warn'

from haversine import haversine, Unit


df = pd.read_csv('anztran.csv')
#--------------------------------------------------------------------------

                                #1) Annual Salary

# Take Values only from Tax Dec = pay/salary

df_annual_salary = df[df["txn_description"]=="PAY/SALARY"].groupby("customer_id").sum()

    # Add it back into df
#df['annual salary'] = df_annual_salary['amount'] |  WONT WORK since df_annual_salary was extracted using groupby
           

# Add values in list and for loop with cust id

annual_salary = []

for customer_id in df['customer_id']:
    annual_salary.append(int(df_annual_salary.loc[customer_id]["amount"]))
    
    # . loc for referencing based on row (unique customer id) and column (amount)
    
df['annual salary'] = annual_salary

#-----------------------------------------------------------------------
                            #2) Annual Salary Correlations



from scipy.stats import pearsonr


# =============================================================================
        # Distance between merchant and customer

#Column split 

cust_latlon = df["long_lat"].str.split(" ", n = 1, expand = True)

df["cust_lat"]= cust_latlon[0]
df["cust_lat"] = df["cust_lat"].astype(float)

df["cust_lon"]= cust_latlon[1]
df["cust_lon"] = df["cust_lon"].astype(float)

merc_latlon = df["merchant_long_lat"].str.split(" ", n = 1, expand = True)

df["merc_lat"]= merc_latlon[0]
df["merc_lat"] = df["merc_lat"].astype(float)

df["merc_lon"]= merc_latlon[1]
df["merc_lon"] = df["merc_lon"].astype(float)

# Deal with missing nan long values in merchant
# Dropping rows containing missing values

df = df.dropna(subset=['merc_lon'])

#            Calculating distance b/w customer and merchant

import math


def distance1(s_lat, s_lng, e_lat, e_lng):
    
   
    R = 6373.0
    
    s_lat = s_lat*np.pi/180.0                      
    s_lng = np.deg2rad(s_lng)     
    e_lat = np.deg2rad(e_lat)                       
    e_lng = np.deg2rad(e_lng)  
    
    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2
    
    return 2 * R * np.arcsin(np.sqrt(d)) 

#print(distance1(merc_revoved["cust_lat"], merc_revoved["cust_lon"], merc_revoved["merc_lat"], merc_revoved["merc_lon"]))

df['DistanceFromMercKM']=distance1(df["cust_lat"], df["cust_lon"], df["merc_lat"], df["merc_lon"])


#       Finding Count of POS/SALES POS by customer for purchase behaviour


txn = df.groupby('customer_id')['txn_description'].count() 

df_txn1 = df[df["txn_description"]==("POS"or"SALES-POS")].groupby("customer_id").count()

txn1 = []

for customer_id in df['customer_id']:
     txn1.append(int(df_txn1.loc[customer_id]["txn_description"])) #WORKS!!
     

df['SalesPosCount'] = txn1



# Correlation of Annual Salary with Categorical Variables - Gender, Card Present


#Card present = Authorised
df = df.loc[df['status'] == 'authorized']

# Gender Dummy

gender_dummies = pd.get_dummies(df.gender, prefix="gender")
df = pd.concat([df,gender_dummies],axis='columns')
df.drop('gender',axis='columns',inplace=True)




# =============================================================================

#Checking correlations using .corr
corr = df.corr()

#print(corr)
# =============================================================================


                # Correlation Values with Annual Salary


#Age = 	0.026

#Amount = 0.091

#Balance = 0.253

#Distance From Merchant = 0.129

#Gender = 0.054

#Card Present = -0.018

#'SalesPosCount' = -0.191


# =============================================================================
                #3) Modelling For Annual Salary

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


# Modelling 1 - Without Training Data

#Simple regression model to predict the annual salary for each customer 

# Taking First index with annual salary = 14191

reg1 = linear_model.LinearRegression()

reg1.fit(df[['DistanceFromMercKM', 'age', 'balance', 'gender_F', 'amount']],df['annual salary'])

print(reg1.predict(np.array([5.192,26,35.39,1,16.25]).reshape(1,-1)))


reg2 = linear_model.LinearRegression()

reg2.fit(df[['DistanceFromMercKM','balance', 'gender_F', 'amount']],df['annual salary'])

print("Without Age = ",reg2.predict(np.array([5.192,35.39,1,16.25]).reshape(1,-1)))


reg3 = linear_model.LinearRegression()

reg3.fit(df[['DistanceFromMercKM','balance', 'amount']],df['annual salary'])

print("Without Age,Gender = ",reg3.predict(np.array([5.192,35.39,16.25]).reshape(1,-1)))


reg4 = linear_model.LinearRegression()

reg4.fit(df[['DistanceFromMercKM','balance']],df['annual salary'])

print("Without Age,Gender, Amount = ",reg4.predict(np.array([5.192,35.39]).reshape(1,-1)))


# =============================================================================

# Modelling 2 - With Train Test Split + Adding SalesPosCount Variable




x = df[['DistanceFromMercKM', 'age', 'gender_F','card_present_flag']]

y = df['annual salary']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2) 

tts = linear_model.LinearRegression()
tts.fit(x_train,y_train)

print("Pred using Linear Train-Test split = ",tts.predict(x_test))

test_predx = tts.predict(x_test)


print("Score without SalesPosCount  = ", tts.score(x_test,y_test))



a1 = df[['DistanceFromMercKM', 'age', 'gender_F','card_present_flag','SalesPosCount']]

b1 = df['annual salary']

from sklearn.model_selection import train_test_split

a1_train, a1_test, b1_train, b1_test = train_test_split(a1,b1,test_size = 0.2) 

tts1 = linear_model.LinearRegression()
tts1.fit(a1_train,b1_train)

print("Pred using Linear Train-Test split = ",tts1.predict(a1_test))

test_predx1 = tts1.predict(a1_test)


print("Score WITH SalesPosCount score  = ", tts1.score(a1_test,b1_test))



# Final model Score = 50%



# =============================================================================

                    #Linear Regression Conclusion

# With the right custom variables model accuracy can be increased from 17% to 50%

# However data inconsistencies may hinder further predictions



# =============================================================================



# Decision Tree Model  | Index 0 Annual Salary = 14191

input_df = df[['DistanceFromMercKM', 'age', 'gender_F','card_present_flag']]

target_df = df['annual salary']

tree_model = tree.DecisionTreeClassifier()

tree_model.fit(input_df,target_df)

print("TreeModel1= ", tree_model.predict(np.array([5,26,1,1]).reshape(1,-1)))

print("TreeModel1Score= ",tree_model.score(input_df, target_df))

# TreeModel1Score=  0.9996112478942595


# =============================================================================

# Decision Tree Model Regressor
# - Adding Merchant State

merchantstate_dummies = pd.get_dummies(df['merchant_state'], prefix="merchant_state")

x1 = df[['DistanceFromMercKM', 'age', 'gender_F','card_present_flag','SalesPosCount']]
x1 = pd.concat([x1,merchantstate_dummies],axis='columns')

y1 = df['annual salary']



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,test_size = 0.2) 

dtr = tree.DecisionTreeRegressor()

dtr.fit(x1_train,y1_train)

print("Pred using Decision Train-Test split = ",dtr.predict(x1_test))

dtr_predx = dtr.predict(x1_test)


print("Decision Train-Test model score = ", dtr.score(x1_test,y1_test))


#Decision Train-Test model score =  0.9996353194211346

# =============================================================================
                            #Decision Tree Conclusion

# Decision Tree Regressor is more accurate (99%) and can be used segment customers
#  based on income brackets

# =============================================================================

