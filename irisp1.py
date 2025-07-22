import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

iris=sns.load_dataset('iris')
print(iris.head())
print(iris.shape)
print(iris.info())
print(iris.describe())
print(iris.isnull().sum())
print(iris['species'].value_counts())


#do labelencoding for species 
from sklearn.preprocessing import LabelEncoder 
label=LabelEncoder()
iris['target']=label.fit_transform(iris['species'])
iris_data=iris.drop(columns='species' , axis=1)
print(iris_data.head())
print(iris_data['target'].value_counts())

X=iris_data.drop(columns='target' , axis=1)
Y=iris_data['target']

#train test  split 
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test=train_test_split(X,Y ,random_state=42 , test_size=0.2 , stratify=Y)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
Xtrain=scaler.fit_transform(X_train)
Xtest=scaler.transform(X_test)

print(Xtrain.std())
#model evaluation 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model=LogisticRegression()

model.fit(Xtrain , Y_train)

#calculate accuracy rate 

xpre=model.predict(Xtrain)
print("the accuracy of train data" , accuracy_score(Y_train , xpre))

xtpre=model.predict(Xtest)
print("the accuracy score of test data" , accuracy_score(Y_test , xtpre))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(Xtrain, Y_train)
print("KNN Test Accuracy:", accuracy_score(Y_test, knn.predict(Xtest)))
print("KNN Train Accuracy:", accuracy_score(Y_train , knn.predict(Xtrain)))
