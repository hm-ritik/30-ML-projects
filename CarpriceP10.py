import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Car Price Prediction")
print("---------------------")

df=pd.read_csv(r"D:\CarPrice_Assignment.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df)

" the dataset is very vast and need to preprocess the data carefully"
"there are 26 columns and 205 rows "
"there is no null value in dataset "

#  dataVisualization 
plt.figure(figsize=(10,12))


plt.subplot(2,2,1)
plt.xlabel("Fuel type")
plt.ylabel("Count")
sns.countplot(df , x='fueltype') 

plt.subplot(2,2,2)
plt.xlabel("Carbody")
plt.ylabel("Count")
sns.countplot(df , x='carbody')


plt.subplot(2,2,3)
plt.xlabel("Engine Type")
plt.ylabel(" Count ")
sns.countplot(df,x='enginetype')

plt.subplot(2,2,4)
plt.xlabel('Aspiration')
plt.ylabel("Count")
sns.countplot(df , x='aspiration')

plt.show()


plt.figure(figsize=(10,12))

plt.subplot(2,2,1)
plt.xlabel("")
plt.ylabel("")
sns.histplot(df , x='horsepower' , kde=True , bins= 30)


plt.subplot(2,2,2)
plt.xlabel("")
plt.ylabel("")
sns.histplot(df , x='peakrpm' , kde=True , bins= 30)



plt.subplot(2,2,3)
plt.xlabel("")
plt.ylabel("")
sns.histplot(df , x='carheight' , kde=True , bins= 30)


plt.subplot(2,2,4)
plt.xlabel("")
plt.ylabel("")
sns.histplot(df , x='carlength' , kde=True , bins= 30)
plt.show() 


df=df.drop(columns=['car_ID' , 'symboling' , 'CarName'] , axis=1 )


df=pd.get_dummies(data= df ,columns=['fueltype','aspiration','doornumber','carbody','drivewheel', 'enginetype', 'cylindernumber','drivewheel', 'fuelsystem' , 'enginelocation'] ,drop_first=True)
print(df.head())

X=df.drop(columns='price' , axis=1)
Y=df['price']

scaler=StandardScaler()

X_train , X_test , Y_train , Y_test = train_test_split(X,Y , random_state=42 , test_size=0.2)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train_scaled , Y_train)

from sklearn.metrics import accuracy_score , r2_score

x_train_predict=model.predict(X_train_scaled)
print("The accuracy of train data" ,  r2_score(Y_train , x_train_predict) )

x_test_predict=model.predict(X_test_scaled)
print("The accuracy of test data" , r2_score(Y_test , x_test_predict))







