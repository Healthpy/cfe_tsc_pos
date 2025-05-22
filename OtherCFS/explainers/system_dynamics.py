import numpy as np
from scipy.integrate import odeint
from sklearn.cluster import KMeans
from .base import BaseCounterfactual

class SDC(BaseCounterfactual):
    """System Dynamics-based Counterfactual explainer for time series data."""
    
    def __init__(self, model, data_name=None, n_segments=10, learning_rate=0.01, 
                 max_iter=100, feedback_strength=0.25):
        """
        Initialize the SDC explainer.
        
        Args:
            model: The model to explain
            data_name: Name of the dataset
            n_segments: Number of key segments to modify
            learning_rate: Learning rate for gradient-based optimization
            max_iter: Maximum number of iterations
            feedback_strength: Strength of feedback loops (0-1)
        """
        super().__init__(model, data_name)
        self.n_segments = n_segments
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.feedback_strength = feedback_strength
        
    def _identify_key_points(self, x):
        """Identify key turning points in the time series."""
        # Handle both univariate and multivariate cases
        if x.shape[0] == 1:  # Univariate
            dx = np.gradient(x[0])
            ddx = np.gradient(dx)
            threshold = np.std(ddx) * 1.5
            key_points = np.where(np.abs(ddx) > threshold)[0]
        else:  # Multivariate
            # Compute gradients for each feature
            dx = np.gradient(x, axis=1)
            ddx = np.gradient(dx, axis=1)
            # Combine information from all features
            combined_ddx = np.sum(np.abs(ddx), axis=0)
            threshold = np.std(combined_ddx) * 1.5
            key_points = np.where(combined_ddx > threshold)[0]
        
        # Use K-means to cluster if too many points
        if len(key_points) > self.n_segments:
            print(f"length of key_points: {len(key_points)}")
            kmeans = KMeans(n_clusters=self.n_segments, n_init=10, random_state=42)
            if x.shape[0] == 1:
                points_2d = np.column_stack((key_points, x[0, key_points]))
            else:
                points_2d = np.column_stack((key_points, np.mean(x[:, key_points], axis=0)))
            kmeans.fit(points_2d)
            key_points = [int(p[0]) for p in kmeans.cluster_centers_]
            
        return np.sort(key_points)

    def _system_dynamics(self, state, t, params, n_features=1):
        """Define system dynamics using differential equations."""
        # Extract current state and parameters
        x = state[:n_features]  # Position states
        dx = state[n_features:]  # Velocity states
        k, b, c = params
        
        # System equations for each feature
        dxdt = dx
        ddxdt = np.array([-k*x[i] - b*dx[i] + c for i in range(n_features)])
        
        return np.concatenate([dxdt, ddxdt])

    def _apply_dynamics(self, x, key_points, target_values):
        """Apply system dynamics to generate counterfactual at key points."""
        n_features = x.shape[0]
        
        # Initialize parameters
        k = 1.0  # spring constant
        b = self.feedback_strength  # damping coefficient
        c = 0.1  # external force
        
        # Initial conditions with proper shapes
        state0 = np.concatenate([x[:, 0], np.zeros(n_features)])
        t = np.linspace(0, x.shape[1], x.shape[1])
        
        # Solve ODE for each feature
        solution = odeint(self._system_dynamics, state0, t, args=([k, b, c], n_features))
        
        # Adjust solution towards target values at key points
        cf = x.copy()
        for i, idx in enumerate(key_points):
            if n_features > 1:
                target = target_values[i]  # Shape: (n_features,)
                for f in range(n_features):
                    cf[f, idx] = target[f]
                    if idx < x.shape[1]-1:
                        segment = solution[idx:idx+2, f]
                        scale = target[f]/segment[0] if segment[0] != 0 else 1.0
                        cf[f, idx:idx+2] = segment * scale
            else:
                target = target_values[i]  # Scalar value
                cf[0, idx] = target
                if idx < x.shape[1]-1:
                    segment = solution[idx:idx+2, 0]
                    scale = target/segment[0] if segment[0] != 0 else 1.0
                    cf[0, idx:idx+2] = segment * scale
                
        return cf

    def _optimize_counterfactual(self, x, target_class, key_points):
        """Optimize counterfactual using gradient descent."""
        cf = x.copy()
        n_features = x.shape[0]
        
        # Initialize target values with proper shapes
        if n_features > 1:
            target_values = np.array([x[:, kp] for kp in key_points])  # Shape: (n_key_points, n_features)
        else:
            target_values = x[0, key_points]  # Shape: (n_key_points,)
        
        for _ in range(self.max_iter):
            new_cf = self._apply_dynamics(cf, key_points, target_values)
            
            if self._is_valid_cf(new_cf, x, target_class):
                return new_cf
                
            # Get probability for target class
            probs = self.predict_proba(new_cf.reshape(1, *new_cf.shape))
            
            # Calculate gradient and ensure proper shape
            grad = (probs[0][target_class] - 1.0) * self.learning_rate
            
            # Apply gradient with proper broadcasting
            if n_features > 1:
                target_values = target_values + grad  # Broadcasting over all features
            else:
                target_values = target_values + grad
            
        return None

    def generate(self, x, target_class=None, X_train=None, y_train=None):
        """
        Generate counterfactual using system dynamics approach.
        
        Args:
            x: Input time series
            target_class: Target class for counterfactual
            X_train: Training data (not used in this method)
            y_train: Training labels (not used in this method)
            
        Returns:
            Counterfactual explanation if found, None otherwise
        """
        # Ensure proper shape
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        # Get target class if not specified
        if target_class is None:
            target_class = self._get_target_class(x)
            
        # Identify key points for modification
        key_points = self._identify_key_points(x)
        
        # Generate counterfactual using system dynamics
        cf = self._optimize_counterfactual(x, target_class, key_points)
        
        if cf is not None:
            return cf.reshape(1, cf.shape[0], cf.shape[1])
        return None
