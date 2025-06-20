import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

'''
PCA is a preprocessing algorithm used to reduce dataset dimensionlity (features). It extracts the most important component in the dataset that
highly influence the variance.
Note that PCA is an unsupervised algorithm that doesn't consider target label values, it just focuses on feature variance maximization.
'''
class PCA:
    '''
    n_components allows to control the trade-off between dimensionality reduction and information retention.
    n_components [0:1] preserves at least the determined values of data variance.
    n_components [1 <= n] preserves the exact number of features requested
    '''
    def __init__(self, n_components = 0.90):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None # Variance explained by each component
        self.explained_variance_ratio = None # Percentage of variance explained by each component
        

    def fit(self, X: np.ndarray):
        _, n_features = X.shape
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        covariance = np.cov(X, rowvar=False)

        eigen_values, eigen_vectors = np.linalg.eig(covariance)
        idxs = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idxs] #each value represents a feature
        eigen_vectors = eigen_vectors[:, idxs] #each vector represents an instance of X

        # Determine the number of components to keep
        if self.n_components is None:
            self.n_components = n_features
        elif isinstance(self.n_components, int):
            self.n_components = min(self.n_components, n_features)
        elif 0 < self.n_components < 1:
            cumulative_variance_ratio = np.cumsum(eigen_values) / np.sum(eigen_values)
            self.n_components = np.argmax(cumulative_variance_ratio >= self.n_components) + 1
        else:
            raise ValueError("Invalid value for n_components.")
        
        self.components = eigen_vectors[:, :self.n_components]
        self.explained_variance = eigen_values[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / np.sum(eigen_values)

        print(f"explained variance per comp. idxs{idxs[:self.n_components]}: {self.explained_variance}")
        print(f"explained variance ratio per comp. idxs{idxs[:self.n_components]}: {self.explained_variance_ratio}")

    
    def transform(self, X):
        if self.components is None or self.mean is None:
            raise ValueError("PCA model has not been fitted yet. Call fit() first.")
        
        centered_X = X - self.mean
        transformed_X = np.dot(centered_X, self.components)
        return transformed_X

    #approximated reconstruction
    def inverse_transform(self, X_transformed):
        if self.components is None or self.mean is None:
            raise ValueError("PCA model has not been fitted yet. Call fit() first.")
        
        reconstructed_X_centered = np.dot(X_transformed, self.components.T)
        reconstructed_X = reconstructed_X_centered + self.mean
        return reconstructed_X