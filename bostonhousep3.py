import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

#exploring to know about dataset 
df=pd.read_csv(r"D:\HousingData.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())


# Dealing with null value 
print(df.isnull().sum())

print(df)

cols=['CRIM' , 'ZN' , 'INDUS' , 'CHAS' , 'AGE' , 'LSTAT']

for col in cols:
    df[col] = df[col].fillna(df[col].mean())

print(df.isnull().sum())

df['price']=df['MEDV']
df=df.drop(columns='MEDV' )
print(df.head())



sns.heatmap(df.corr() , annot=True)
plt.show()

X=df.drop(columns='price')
Y=df['price']

from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test =train_test_split(X,Y, test_size=0.2 , random_state=42)
print(X_train.shape , X_test.shape , Y_train.shape ,Y_test.shape)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_trainsca=scaler.fit_transform(X_train)
X_testsca=scaler.transform(X_test)

# AT first we apply linear regression 

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_trainsca , Y_train)

X_pre=model.predict(X_trainsca)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("the r2 score of trained data :" , r2_score(Y_train , X_pre))
print("the mean square error of trained data :" , mean_squared_error(Y_train , X_pre))
print("the mean square error of trained data :" , mean_absolute_error(Y_train , X_pre))

Xt_pre=model.predict( X_testsca)

print("the r2 score of test data :" , r2_score(Y_test , Xt_pre))
print("the mean square error of test data :" , mean_squared_error(Y_test, Xt_pre))
print("the mean square error of test data :" , mean_absolute_error(Y_test , Xt_pre))


# now we apply XGBOOST Regressor to improve model performance 

from xgboost import XGBRegressor

model=XGBRegressor()

model.fit(X_train , Y_train)

X_trainx=model.predict(X_train)

print("the r2 score of trained data :" , r2_score(Y_train , X_trainx))
print("the mean square error of trained data :" , mean_squared_error(Y_train , X_trainx))
print("the mean square error of trained data :" , mean_absolute_error(Y_train , X_trainx))

x_testx=model.predict(X_test)

print("the r2 score of trained data :" , r2_score(Y_test , x_testx))
print("the mean square error of trained data :" , mean_squared_error(Y_test , x_testx))
print("the mean square error of trained data :" , mean_absolute_error(Y_test , x_testx))








