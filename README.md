# Dimensionality-Reduction-Using-PCA-and-t-SNE
# Project Overview
This project demonstrates the application of dimensionality reduction techniques, specifically Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE), on high-dimensional datasets. The project aims to improve computational efficiency, enhance data visualization, and reduce noise in the dataset while maintaining the underlying structure of the data.

Key Features: Implementation of PCA and t-SNE for dimensionality reduction. 2D and 3D visualization of high-dimensional data. Hyperparameter tuning for t-SNE using GridSearchCV. Quantitative evaluation of the clustering quality using the Silhouette Score. A reusable and scalable pipeline for applying dimensionality reduction to new datasets.
# Installation
Prerequisites: Python 3.10 or higher pip (Python package manager)
# Project Details
1. Principal Component Analysis (PCA): PCA is used to transform the data into a set of orthogonal components that explain the maximum variance in the data. The project visualizes these components and evaluates the clustering quality after reduction.

2. t-Distributed Stochastic Neighbor Embedding (t-SNE): t-SNE is a nonlinear dimensionality reduction technique, particularly effective for visualizing data. The project includes hyperparameter tuning for t-SNE and compares its performance against PCA


3. Evaluation Metrics: Silhouette Score: Used to evaluate the quality of the clusters formed after dimensionality reduction. The score ranges from -1 to 1, where higher scores indicate better-defined clusters.
