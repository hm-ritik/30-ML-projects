print(" Diabeties Prediction ")
" diabeties prediction project 7 "

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


def loaddata(path):
    df=pd.read_csv(path)

    return df

path=r"D:\diabetes (3).csv"
df=loaddata(path)

def eda(df):
    print(df.head())
    print(df.info())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

eda(df)    

plt.subplot(2,2,1)
sns.histplot(data=df , x='Age' , hue='Outcome' ,bins=30, kde=True)
plt.xlabel("The age group of people")
plt.ylabel(" OUTCOME")
plt.title("")

plt.subplot(2,2,2)
sns.histplot(data=df , x='Glucose' , hue='Outcome' ,bins=30, kde=True)
plt.xlabel("Glucose")
plt.ylabel(" OUTCOME")
plt.title("")

plt.subplot(2,2,3)
sns.histplot(data=df , x='BloodPressure' , hue='Outcome' ,bins=30, kde=True)
plt.xlabel("Blood Pressure")
plt.ylabel(" OUTCOME")
plt.title("")

plt.subplot(2,2,4)
sns.histplot(data=df , x='BMI' , hue='Outcome' ,bins=30, kde=True)
plt.xlabel("BMI")
plt.ylabel(" OUTCOME")
plt.title("")
#plt.show()




X=df.drop(columns='Outcome', axis=1)
Y=df["Outcome"]

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(X , Y , random_state=42 , test_size=0.3)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_standar=scaler.fit_transform(x_train)
Xt_standar=scaler.transform(x_test)


#checking for imbalanced data 

print(df['Outcome'].value_counts())

" there are 500 0 and only 268 1 , sign of imbalanced data"
from imblearn.over_sampling import SMOTE

sm=SMOTE()

Xsampled , Ysampled=sm.fit_resample(X_standar , y_train)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(max_iter=1000,  
    random_state=42,  
    solver='liblinear')
model.fit(Xsampled , Ysampled)

from sklearn.metrics import accuracy_score 

xtrain_predict=model.predict(Xsampled)
print("The accuracy score of train data:" , accuracy_score(Ysampled , xtrain_predict))

xtest_predict=model.predict(Xt_standar)
print("The accuracy score of Test data :" , accuracy_score(y_test , xtest_predict))





