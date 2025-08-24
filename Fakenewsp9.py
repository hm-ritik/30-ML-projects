"""Fake news Detector a ml model which detect which news is fake or which one is correct """
"Concepts we use are NLP , TF-IDf , BERT"
print("Fake News Detector")
print("--------------------")

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv(r"D:\fake_real_news.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

df=df.drop(columns='date' , axis=1)

df['news'] = df['title'] + " " + df['text'] + " " + df['subject']

df=df.drop(columns=['title' , 'subject' , 'text'] , axis=1)

print(df.head)

X=df['news']
Y=df['target']

from sklearn.model_selection import train_test_split

X_train  , X_test , Y_train ,   Y_test = train_test_split (X, Y , test_size=0.2 , random_state= 41)

Vectorizer= TfidfVectorizer( max_features=5000)


X_trainv=Vectorizer.fit_transform(X_train)
X_testv=Vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(max_iter=1000)

model.fit(X_trainv , Y_train)

from sklearn.metrics import accuracy_score
train_predict=model.predict(X_trainv)
print("The accuracy Score of training Data" , accuracy_score(Y_train , train_predict))

test_predict=model.predict(X_testv)
print("The accuracy of test data is" , accuracy_score(Y_test , test_predict))
