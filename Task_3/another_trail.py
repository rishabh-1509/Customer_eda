import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and preprocess data
customers_df = pd.read_csv("Customers.csv")
transactions_df = pd.read_csv("Transactions.csv")

# Aggregate transaction data
transactions_agg = transactions_df.groupby("CustomerID").agg(
    TotalAmount=("Price", "sum"),
    AvgAmount=("Price", "mean"),
    TransactionCount=("TransactionID", "count")
).reset_index()

# Merge with customer data
data = pd.merge(customers_df, transactions_agg, on="CustomerID", how="inner")

# Select relevant features for clustering
features = data[["TotalAmount", "AvgAmount", "TransactionCount"]]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 2: Apply clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Set number of clusters to 5 (tune this)
clusters = kmeans.fit_predict(scaled_features)

# Assign clusters to the data
data["Cluster"] = clusters

# Step 3: Evaluate clustering
db_index = davies_bouldin_score(scaled_features, clusters)
silhouette_avg = silhouette_score(scaled_features, clusters)

print(f"Davies-Bouldin Index: {db_index}")
print(f"Silhouette Score: {silhouette_avg}")

# Step 4: Visualize clusters
# Reduce dimensions for visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=reduced_features[:, 0],
    y=reduced_features[:, 1],
    hue=data["Cluster"],
    palette="viridis",
    s=50
)
plt.title("Customer Segmentation Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()
