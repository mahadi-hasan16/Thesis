import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# Load your dataset
# Assuming df is your DataFrame with the features
# Make sure to replace 'your_data.csv' with your actual dataset file path
df = pd.read_csv('./dataset2.csv')

# Select features for clustering
features = df[['Answer_Correctness', 'Math_MeanNN', 'Math_SDNN', 'Math_RMSSD', 'Math_SDSD', 
               'Math_NN50', 'Math_pNN50', 'Stroop_MeanNN', 'Stroop_SDNN', 'Stroop_RMSSD', 
               'Stroop_SDSD', 'Stroop_NN50', 'Stroop_pNN50', 'IQ_MeanNN', 'IQ_SDNN', 'IQ_RMSSD', 
               'IQ_SDSD', 'IQ_NN50', 'IQ_pNN50']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, indices_train, indices_test = train_test_split(features_scaled, df.index, test_size=0.2, random_state=42)

# Apply K-Means clustering on the training set
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_train = kmeans.fit_predict(X_train)

# Create a DataFrame for the training set with correct index
df_train = df.loc[indices_train].copy()
df_train['cluster'] = cluster_train

# Evaluate the clustering using silhouette score on the testing set
cluster_test = kmeans.predict(X_test)

# Create a DataFrame for the testing set with correct index
df_test = df.loc[indices_test].copy()
df_test['cluster'] = cluster_test

# Analyze cluster characteristics on the training set
cluster_means = df_train.groupby('cluster').mean()
print("\nCluster Means (Training Set):")
print(cluster_means)

# Visualize the clusters using PCA on the testing set (2D plot)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(X_test)
df_test['pca1'] = features_pca[:, 0]
df_test['pca2'] = features_pca[:, 1]

# Plot clusters in 2D space
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for cluster in df_test['cluster'].unique():
    plt.scatter(df_test[df_test['cluster'] == cluster]['pca1'], df_test[df_test['cluster'] == cluster]['pca2'], label=f'Cluster {cluster}')

plt.title('K-Means Clustering (Testing Set)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
