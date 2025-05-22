import numpy as np
from dtaidistance import dtw

class KNeighborsTimeSeries:
    """K-nearest neighbors classifier using DTW distance."""
    
    def __init__(self, n_neighbors=1, metric='dtw'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        
    def fit(self, X):
        """Store training data."""
        self.X_train = X
        return self
        
    def kneighbors(self, X):
        """Find k-nearest neighbors."""
        n_samples = len(X)
        distances = np.zeros((n_samples, len(self.X_train)))
        
        # Calculate DTW distances
        for i in range(n_samples):
            for j in range(len(self.X_train)):
                x1 = X[i].flatten()
                x2 = self.X_train[j].flatten()
                distances[i, j] = dtw.distance(x1, x2)
                
        # Get indices of k nearest neighbors
        indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        return distances[np.arange(len(distances))[:, None], indices], indices