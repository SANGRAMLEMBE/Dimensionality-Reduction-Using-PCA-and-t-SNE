# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
print(df.head())

# Create a preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # Add more preprocessing steps here if needed
])

# Apply the preprocessing pipeline
X_scaled = preprocessing_pipeline.fit_transform(X)

# Define the t-SNE model
tsne = TSNE(random_state=42)

# Define the parameter grid
param_grid = {
    'perplexity': [5, 30, 50],
    'n_iter': [300, 500, 1000],
    'learning_rate': [10, 100, 500]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(tsne, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_scaled)

# Best parameters
print(f"Best t-SNE parameters: {grid_search.best_params_}")

# Apply PCA
pca = PCA(n_components=3)  # 3D PCA
X_pca = pca.fit_transform(X_scaled)

# Apply t-SNE with best parameters
tsne_best = TSNE(**grid_search.best_params_, random_state=42)
X_tsne = tsne_best.fit_transform(X_scaled)

# Apply t-SNE with 3 components
tsne_best = TSNE(n_components=3, **grid_search.best_params_, random_state=42)
X_tsne = tsne_best.fit_transform(X_scaled)

# 3D t-SNE Visualization
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(122, projection='3d')
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, cmap='viridis')
legend2 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend2)
ax.set_title('3D t-SNE')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')

plt.show()

# 2D t-SNE Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('2D t-SNE: Iris Dataset')
plt.colorbar(label='Target Labels')
plt.show()

# Silhouette Score
sil_pca = silhouette_score(X_pca, y)
sil_tsne = silhouette_score(X_tsne, y)

# Calinski-Harabasz Index
ch_pca = calinski_harabasz_score(X_pca, y)
ch_tsne = calinski_harabasz_score(X_tsne, y)

print(f'Silhouette Score (PCA): {sil_pca}')
print(f'Silhouette Score (t-SNE): {sil_tsne}')
print(f'Calinski-Harabasz Index (PCA): {ch_pca}')
print(f'Calinski-Harabasz Index (t-SNE): {ch_tsne}')

from sklearn.cluster import MiniBatchKMeans

# Use MiniBatchKMeans for larger datasets
batch_size = 100  # Adjust based on dataset size
mini_tsne = TSNE(n_components=2, method='exact', random_state=42)
mini_kmeans = MiniBatchKMeans(batch_size=batch_size)

X_mini_tsne = mini_tsne.fit_transform(X_scaled)

from sklearn.base import BaseEstimator, TransformerMixin

class DimensionalityReductionPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, method='pca', n_components=2, **kwargs):
        self.method = method
        self.n_components = n_components
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y=None):
        if self.method == 'pca':
            self.model = PCA(n_components=self.n_components, **self.kwargs)
            self.model.fit(X)
        elif self.method == 'tsne':
            self.model = TSNE(n_components=self.n_components, **self.kwargs)
            # t-SNE will fit and transform in the same step
            X_reduced = self.model.fit_transform(X)
            return X_reduced
        return self

    def transform(self, X, y=None):
        if self.method == 'pca':
            return self.model.transform(X)
        elif self.method == 'tsne':
            # Since t-SNE does not support transform, we return the result of fit_transform
            X_reduced = self.model.fit_transform(X)
            return X_reduced

# Example Usage
dr_pipeline = DimensionalityReductionPipeline(method='tsne', n_components=2, perplexity=30, n_iter=1000)
X_reduced = dr_pipeline.fit(X_scaled)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Model training
model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X_reduced, y, cv=5)
print(f'Cross-validation accuracy: {np.mean(scores)}')
