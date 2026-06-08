"""
Step 2: Customer Segmentation (Unsupervised Learning)
Author: Student
Course: Retail Analytics Project

This script standardizes the RFM features and segments customers using:
1. K-Means clustering (with Elbow Method & Silhouette Score to select K).
2. DBSCAN clustering (for comparison and outlier detection).
3. Dynamic, robust cluster labeling to assign meaningful segment names:
   - Champions
   - Loyal Customers
   - New Customers
   - At-Risk Customers
   - Lost Customers
4. PCA (Principal Component Analysis) to visualize clusters in 2D.
5. Saves the models, scalers, and plots.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------------------------------------------
# 1. Load Data
# ----------------------------------------------------
print("--- Step 1: Loading customer feature data ---")
data_path = "data/processed/customer_features.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing {data_path}. Please run 1_data_prep.py first.")

df = pd.read_csv(data_path)
print(f"Loaded dataset with shape: {df.shape}")

# Select RFM features for clustering
rfm_cols = ['recency', 'frequency', 'monetary']
X_rfm = df[rfm_cols]

# ----------------------------------------------------
# 2. Standardize Features
# ----------------------------------------------------
print("\n--- Step 2: Scaling RFM features ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rfm)
print("Standardization complete. Mean of scaled features ~0, Variance ~1.")

# ----------------------------------------------------
# 3. K-Means Evaluation (Elbow & Silhouette)
# ----------------------------------------------------
print("\n--- Step 3: Evaluating K-Means (K=2 to 8) ---")
inertias = []
silhouette_scores = []
k_range = range(2, 9)

# To avoid massive memory consumption and slow computation on 96k records,
# we compute the Silhouette Score on a random sample of 10,000 customers.
sample_size = 10000
np.random.seed(42)
sample_indices = np.random.choice(X_scaled.shape[0], size=sample_size, replace=False)
X_scaled_sample = X_scaled[sample_indices]

for k in k_range:
    print(f"Fitting K-Means for K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    
    # Calculate silhouette score on the sample
    labels_sample = kmeans.predict(X_scaled_sample)
    score = silhouette_score(X_scaled_sample, labels_sample)
    silhouette_scores.append(score)
    print(f"  K={k} -> Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {score:.4f}")

# Save the Elbow & Silhouette plots
plt.figure(figsize=(12, 5))

# Elbow Curve
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, marker='o', color='#1f77b4', linewidth=2)
plt.title('Elbow Method (Inertia)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.grid(True, linestyle='--', alpha=0.6)

# Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', color='#ff7f0e', linewidth=2)
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plot_path = "reports/elbow_silhouette.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved Elbow & Silhouette evaluation plot to: {plot_path}")

# ----------------------------------------------------
# 4. Fit Optimal K-Means Model (K=5)
# ----------------------------------------------------
# A choice of K=5 clusters is typically optimal for retail customer segmentation.
optimal_k = 5
print(f"\n--- Step 4: Fitting final K-Means model with K={optimal_k} ---")
kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans_opt.fit_predict(X_scaled)

# ----------------------------------------------------
# 5. Compare with DBSCAN
# ----------------------------------------------------
print("\n--- Step 5: Comparing with DBSCAN for outlier detection (on sample) ---")
# Since DBSCAN can be extremely slow and memory intensive on 96k rows (O(N^2) complexity),
# we fit DBSCAN on the same random 10,000 customer sample to check for outliers/noise.
dbscan = DBSCAN(eps=0.5, min_samples=15, n_jobs=-1)
dbscan_labels = dbscan.fit_predict(X_scaled_sample)

n_outliers = np.sum(dbscan_labels == -1)
n_db_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"DBSCAN on 10k sample found {n_db_clusters} clusters and {n_outliers} outlier/noise customers ({n_outliers / len(X_scaled_sample) * 100:.2f}% of sample).")
print("Note: DBSCAN helps identify unusual outlier customers, but K-Means is better for assigning a clear profile to every customer.")

# ----------------------------------------------------
# 6. Dynamic and Robust Cluster Labeling
# ----------------------------------------------------
print("\n--- Step 6: Mapping cluster IDs to human-readable labels ---")
# To make this process robust and independent of random cluster index initialization,
# we compute cluster centroids and label them by sorting:
centroids = df.groupby('cluster')[rfm_cols].mean()
print("Cluster Centroids (Mean raw values):")
print(centroids)

# We will assign labels based on sorting characteristics:
# 1. 'Champions': highest average monetary spend
# 2. 'Lost Customers': highest average recency among remaining clusters
# 3. 'At-Risk Customers': highest average recency among remaining clusters
# 4. 'New Customers': lowest average recency among remaining clusters
# 5. 'Loyal Customers': the last remaining cluster

cluster_mapping = {}
available_clusters = list(range(optimal_k))

# 1. Champions (Highest Monetary)
champions_cluster = centroids.loc[available_clusters, 'monetary'].idxmax()
cluster_mapping[champions_cluster] = "Champions"
available_clusters.remove(champions_cluster)

# 2. Lost Customers (Highest Recency from the remaining)
lost_cluster = centroids.loc[available_clusters, 'recency'].idxmax()
cluster_mapping[lost_cluster] = "Lost Customers"
available_clusters.remove(lost_cluster)

# 3. At-Risk Customers (Next Highest Recency)
at_risk_cluster = centroids.loc[available_clusters, 'recency'].idxmax()
cluster_mapping[at_risk_cluster] = "At-Risk Customers"
available_clusters.remove(at_risk_cluster)

# 4. New Customers (Lowest Recency among remaining)
new_cluster = centroids.loc[available_clusters, 'recency'].idxmin()
cluster_mapping[new_cluster] = "New Customers"
available_clusters.remove(new_cluster)

# 5. Loyal Customers (The final remaining cluster)
loyal_cluster = available_clusters[0]
cluster_mapping[loyal_cluster] = "Loyal Customers"

print("\nFinal Cluster Index Mapping:")
for cid, label in cluster_mapping.items():
    print(f"  Cluster {cid} -> {label}")

# Map cluster labels in DataFrame
df['segment'] = df['cluster'].map(cluster_mapping)

# ----------------------------------------------------
# 7. PCA 2D Cluster Visualization
# ----------------------------------------------------
print("\n--- Step 7: Visualizing clusters using PCA ---")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]

plt.figure(figsize=(10, 8))
# Use a curated, premium color palette for segments
colors = {
    'Champions': '#2ca02c',       # Vibrant Green
    'Loyal Customers': '#1f77b4',  # Sleek Blue
    'New Customers': '#ff7f0e',    # Warm Orange
    'At-Risk Customers': '#d62728',# Cautionary Red
    'Lost Customers': '#7f7f7f'    # Muted Grey
}

sns.scatterplot(
    data=df, x='pca_1', y='pca_2', hue='segment', 
    palette=colors, alpha=0.6, s=15, edgecolor=None
)
plt.title('Customer Segments PCA 2D Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Segments', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

pca_plot_path = "reports/pca_clusters.png"
plt.savefig(pca_plot_path, dpi=150)
plt.close()
print(f"Saved PCA 2D Cluster visualization to: {pca_plot_path}")

# ----------------------------------------------------
# 8. Save Models & Output Dataset
# ----------------------------------------------------
print("\n--- Step 8: Saving models and output data ---")
os.makedirs("models", exist_ok=True)

# Save scaler and kmeans model
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/kmeans.pkl", "wb") as f:
    pickle.dump(kmeans_opt, f)

# Save cluster mapping dictionary
with open("models/cluster_mapping.pkl", "wb") as f:
    pickle.dump(cluster_mapping, f)

# Save labeled features CSV
df.to_csv("data/processed/segmented_customers.csv", index=False)
print("Saved scaler to models/scaler.pkl")
print("Saved K-Means to models/kmeans.pkl")
print("Saved cluster mapping dictionary to models/cluster_mapping.pkl")
print("Saved segmented dataset to data/processed/segmented_customers.csv")
print("\nCustomer segmentation stage finished successfully!")
