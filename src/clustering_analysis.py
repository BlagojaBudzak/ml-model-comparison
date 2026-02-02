#%% md
# # Imports
#%%
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
#%% md
# # Load Dataset
#%%
dataset = pd.read_csv("../data/Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values  # Annual Income & Spending Score
#%% md
# # Hierarchical Clustering
#%% md
# ## Dendrogram
#%%
plt.figure(figsize=(8,5))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
#%% md
# ## Fit Hierarchical Clustering
#%%
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Plot clusters
plt.figure(figsize=(8,5))
colors = ['red','blue','green','cyan','magenta']
for i in range(5):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
plt.title('Hierarchical Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
#%% md
# # K-Means Clustering
#%% md
# ## Elbow Method
#%%
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#%% md
# ## Fit K-Means with 5 clusters
#%%
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plot clusters
plt.figure(figsize=(8,5))
for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centroids', edgecolor='black')
plt.title('K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
#%% md
# # Cluster Statistics
#%%
print("Cluster Statistics (K-Means):")
for i in range(5):
    cluster_points = X[y_kmeans == i]
    print(f"Cluster {i+1}:")
    print(f"  Number of customers: {len(cluster_points)}")
    print(f"  Mean Income: {cluster_points[:,0].mean():.2f}")
    print(f"  Mean Spending Score: {cluster_points[:,1].mean():.2f}\n")
#%% md
# # Observations
# 
# - Cluster 1: Low income, low spending → Budget-conscious customers
# - Cluster 2: High income, low spending → Potential savers
# - Cluster 3: Medium income, medium spending → Average customers
# - Cluster 4: Low income, high spending → Carefree spenders
# - Cluster 5: High income, high spending → Premium customers
# 
# 