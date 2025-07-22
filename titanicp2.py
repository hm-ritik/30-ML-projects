import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


df=sns.load_dataset('titanic')
print(df.head())
print(df.shape)

titanic_df=df.drop(columns=['who' , 'alive' , 'embark_town' , 'adult_male' , 'class' , 'deck'] ,  axis=1)
print(titanic_df.shape)
print(titanic_df.head())
print(titanic_df.shape)
print(titanic_df.info())
print(titanic_df.describe())
print(titanic_df.isnull().sum())

#there are 117 null datapoint in age fill it with mean
t_df=titanic_df.copy()
t_df['age'] = titanic_df['age'].fillna(titanic_df['age'].mean())
t_df['embarked']=titanic_df['embarked'].fillna('S')

print(t_df.isnull().sum())

print(t_df['survived'].value_counts())
print(t_df['age'].value_counts())
print(t_df['sex'].value_counts())


# now we do plotting using matplotlib and seaborn 
sns.countplot(data=t_df , x='survived')
plt.title("Data of passenger survived or Died ")
plt.show()

sns.countplot(data=t_df ,x='pclass' )
plt.title("Data of passenger class")
plt.show()

sns.countplot(data=t_df , x='embarked' )
plt.title("City of passenger")
plt.show()

sns.countplot(data=t_df , x='sex' )
plt.title("Gender of passenger")
plt.show()

sns.histplot(data=t_df , x='fare' , kde=True , bins=20)
plt.title("Pricing of fare")

plt.show()

sns.countplot(data=t_df , x='survived' , hue='sex')
plt.title("the survival according to gender ")
plt.show()

sns.countplot(data=t_df , x='pclass' , hue='sex')
plt.title("the survival according to gender ")
plt.show()

sns.histplot(data=t_df , x='age' , hue='survived' , bins=15  , kde=True)
plt.title("the Survival acc to Age group")
plt.show()

t_dff=t_df.drop(columns=['embarked' , 'alone' , 'sex' ], axis=1)
sns.heatmap(t_dff.corr() , annot=True , linewidths=0.5)
plt.title("Correlation of dataset")
plt.show()

#here i think we are done with visualization of dataset 

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
 
t_df['sexe']=label.fit_transform(t_df['sex'])
t_df['embarkedn']=label.fit_transform(t_df['embarked'].fillna('S'))
t_df['alonen']=label.fit_transform(t_df['alone'])

t_df=t_df.drop(columns=['sex' , 'embarked' , 'alone' ])
print(t_df.head())

from sklearn.model_selection import train_test_split

X=t_df.drop(columns='survived' , axis=1)
Y=t_df['survived']

X_train , X_test , Y_train , Y_test = train_test_split( X , Y , test_size=0.2 , random_state=42 , stratify=Y)

print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_trainscaled=scaler.fit_transform(X_train)
X_testscaled=scaler.transform(X_test)

#fitting and model evaluation 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model=LogisticRegression()
model.fit(X_trainscaled , Y_train)

xpre=model.predict(X_trainscaled)
print("the accuracy score of training data" , accuracy_score(Y_train , xpre))

xtpre=model.predict(X_testscaled)
print("the acuracy score of test data" , accuracy_score(Y_test , xtpre))