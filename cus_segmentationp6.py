""" customer segmentation analysis using kmeans clustring"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def loaddata(path):
    path=r"D:\Mall_Customers.csv"
    df=pd.read_csv(path)
    return df

path=r"D:\Mall_Customers.csv"
df=loaddata(path)



def eda(df):
    print(df.head())
    print(df.info())
    print(df.isnull().sum()) 


eda(df)    


X=df[['Annual Income (k$)' , 'Spending Score (1-100)']]

print(X)

" now WSCC (within cluster square of sum ) "

WSCC=[]

for i in range (1,11):
    kmean=KMeans(n_clusters=i , random_state=42 ,  init='k-means++')
    kmean.fit(X)
    WSCC.append(kmean.inertia_)

plt.plot(range(1,11) , WSCC)
plt.title("No of Clusters For Optimal Solution(Elbow Method) ")
plt.xlabel('Range of cluster')
plt.ylabel('WSCC')
plt.show()    


"the optimal number of cluster is 5 because bend of wcss is slow down after 5 "


kmean=KMeans(n_clusters=5 , init='k-means++' , random_state=42)
predict=kmean.fit_predict(X)

print(predict)


" now visualize the clusters"

# Visualizing the clusters
plt.scatter(X.iloc[predict == 0, 0], X.iloc[predict == 0, 1], s=100, label='Cluster 1')
plt.scatter(X.iloc[predict == 1, 0], X.iloc[predict == 1, 1], s=100, label='Cluster 2')
plt.scatter(X.iloc[predict == 2, 0], X.iloc[predict == 2, 1], s=100, label='Cluster 3')
plt.scatter(X.iloc[predict == 3, 0], X.iloc[predict == 3, 1], s=100, label='Cluster 4')
plt.scatter(X.iloc[predict == 4, 0], X.iloc[predict == 4, 1], s=100, label='Cluster 5')

# Centroids
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], s=300, c='black', label='Centroids', marker='X')

plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()



    



























