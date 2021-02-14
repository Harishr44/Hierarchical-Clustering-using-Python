# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:35:27 2020

@author: Harish
"""

import pandas as pd
import matplotlib.pyplot as plt 
airline=pd.read_excel("EastWestAirlines.xlsx")
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

df_norm = norm_func(airline.iloc[:,1:])
df_norm.describe()

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
z = linkage(df_norm, method="single",metric="euclidean")


plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=8.,  
)

plt.show()

from sklearn.cluster import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=10,linkage='single',affinity = "euclidean").fit(df_norm) 
h_complete.labels_


cluster_labels=pd.Series(h_complete.labels_)
cluster_labels.value_counts()
airline['clust']=cluster_labels
airline = airline.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
a1_means=pd.DataFrame(airline.groupby(airline.clust).mean())
airline.to_csv("Airline_h.csv",index=False) 
