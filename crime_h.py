# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:25:09 2020

@author: Harish
"""

import pandas as pd
import matplotlib.pyplot as plt 
crime= pd.read_csv("crime_data.csv")
def norm_func(i):
    x=(i-i.mean())/(i.std())
    return(x)
df_norm=norm_func(crime.iloc[:,1:])

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
z = linkage(df_norm, method="single",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=8.,  
)

from sklearn.cluster import	AgglomerativeClustering 
h_complete	= AgglomerativeClustering(n_clusters=5,linkage='single',affinity = "euclidean").fit(df_norm) 

h_complete.labels_

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels.value_counts()
crime['clust']=cluster_labels
crime = crime.iloc[:,[5,0,1,2,3,4]]

crime.groupby(crime.clust).mean()

# Zero cluster cities show less average value for murder assault rape than other cluster cities, and as i calculated
# the linkages by using single method which shows the simmilar group of cities in one cluster so zero cluster cities have safer environment than other cities
