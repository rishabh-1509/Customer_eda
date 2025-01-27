import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load Data
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

# Data Exploration and Preprocessing
# Merge data on customer ID (assuming "CustomerID" is the common key)
data = pd.merge(customers, transactions, on='CustomerID', how='inner')

# Feature Engineering
# Example: Total spend, number of transactions, average transaction value
data['TotalSpend'] = data.groupby('CustomerID')['Price'].transform('sum')
data['NumTransactions'] = data.groupby('CustomerID')['Price'].transform('count')
data['AvgTransactionValue'] = data['TotalSpend'] / data['NumTransactions']

# Drop duplicate CustomerID rows and keep features for clustering
clustering_data = data[['CustomerID', 'TotalSpend', 'NumTransactions', 'AvgTransactionValue']].drop_duplicates()

# Standardize Features
scaler = StandardScaler()
X = scaler.fit_transform(clustering_data[['TotalSpend', 'NumTransactions', 'AvgTransactionValue']])

# Clustering
# Choosing a range for the number of clusters
cluster_range = range(2, 11)
db_scores = []
silhouette_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    # Calculate metrics
    db_index = davies_bouldin_score(X, labels)
    silhouette_avg = silhouette_score(X, labels)

    db_scores.append(db_index)
    silhouette_scores.append(silhouette_avg)

# Plot DB Index and Silhouette Scores
plt.figure(figsize=(12, 6))
plt.plot(cluster_range, db_scores, label='DB Index', marker='o')
plt.plot(cluster_range, silhouette_scores, label='Silhouette Score', marker='x')
plt.title('DB Index and Silhouette Scores for Different Cluster Counts')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()

# Select the optimal number of clusters (manually or based on DB Index)
optimal_k = cluster_range[np.argmin(db_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X)

# Add cluster labels to data
clustering_data['Cluster'] = labels

# Visualize Clusters using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
clustering_data['PCA1'] = pca_result[:, 0]
clustering_data['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(data=clustering_data, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title(f'Customer Segments (K={optimal_k})')
plt.show()

# Report Results
print(f'Optimal Number of Clusters: {optimal_k}')
print(f'Davies-Bouldin Index: {min(db_scores)}')

# Save results
clustering_data.to_csv('Customer_Segmentation_Results.csv', index=False)
