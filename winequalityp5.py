" wine quality prediction project nuumber 5"
"work on muliclassification problem"

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

df=pd.read_csv(r"D:\winequality-red.csv")
print(df.shape)
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())


" plotting of data using maplotlib and seaborn "
print(df['quality'].value_counts())
sns.countplot(data=df , x='quality')
plt.title("Quality classification of Wine")
plt.show()


plt.figure(figsize=(12, 10))  

plt.subplot(2,2,1)
sns.barplot(data=df , y='alcohol' , x='quality' )
plt.title("Quality vs Alcohol")


plt.subplot(2,2,2)
plt.title("Quality vs PH")
sns.barplot(data=df , y='pH' , x='quality' )

plt.subplot(2,2,3)
plt.title("Quality vs density ")
sns.barplot(data=df , y='density' , x='quality' )

plt.subplot(2,2,4)
plt.title("Quality vs fixed acidity")
sns.barplot(data=df , y='fixed acidity' , x='quality' )


plt.tight_layout()  
plt.show()


def simplify_quality(value):
    if value <= 4:
        return 0  
    elif value == 5 or value<=6:
        return 1 
    else:
        return 2  

df['quality_label'] = df['quality'].apply(simplify_quality)



X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']
print(y.value_counts())

from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from imblearn.over_sampling import SMOTE

sm=SMOTE()

X_train_resampled , Y_train_resampled = sm.fit_resample(X_train_scaled , Y_train)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,        
    max_depth=None,          
    random_state=42,
    class_weight='balanced'   
)

model.fit(X_train_resampled, Y_train_resampled)

from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))







