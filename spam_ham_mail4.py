import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

" we are now going to deal with dataset "

df=pd.read_csv(r"D:\spam.csv" , encoding='latin1')
print(df.head()) 
print(df.info())
print(df.describe())
print(df.shape)

print(df.isnull().sum())

df=df.drop(columns=[  'Unnamed: 2' ,'Unnamed: 3' , 'Unnamed: 4'] , axis=1)
print(df.head()) 

from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['Mail_cate']=label.fit_transform(df['v1'])
df=df.drop(columns='v1')

df['Message']=df['v2']
df=df.drop(columns='v2')
print(df.head())

X=df['Message']
Y=df['Mail_cate']

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test=train_test_split(X ,Y , test_size=0.2 , random_state=42)
print(x_train.shape)
print(y_train.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

TF=TfidfVectorizer(stop_words='english' , lowercase=True)

x_train_tfidf=TF.fit_transform(x_train)
x_test_tffidf=TF.transform(x_test)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = sm.fit_resample(x_train_tfidf, y_train)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

model=LogisticRegression()

model.fit(x_train_balanced,y_train_balanced)

train_predict=model.predict(x_train_balanced)
print("the accuracy of training data :" , accuracy_score(y_train_balanced , train_predict))

test_predict=model.predict(x_test_tffidf)
print("the accuracy score of test data :" , accuracy_score(y_test , test_predict))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_balanced , train_predict))
print(confusion_matrix(y_test, test_predict))
