import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseCounterfactual

class DynamicSystemCF(BaseCounterfactual):
    """Dynamic Systems-based Counterfactual Generator."""
    
    def __init__(self, model, data_name=None, n_steps=100, learning_rate=0.01, 
                 max_iter=100, control_weight=0.1, state_weight=1.0):
        super().__init__(model, data_name)
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.control_weight = control_weight
        self.state_weight = state_weight
        
    def _system_dynamics(self, state, control):
        """
        Define the state-space dynamics: x_{t+1} = f(x_t, u_t)
        
        Args:
            state: Current state x_t
            control: Control input u_t
        Returns:
            Next state x_{t+1}
        """
        # Ensure proper shapes and handle batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(control.shape) == 1:
            control = control.unsqueeze(0)
            
        # Create dynamics matrices of appropriate size
        state_dim = state.size(-1)
        A = torch.eye(state_dim, device=state.device)  # State transition
        B = 0.1 * torch.eye(state_dim, device=state.device)  # Control input scaled
        
        # Simple linear dynamics with proper broadcasting
        next_state = state + torch.matmul(control.T, B)
        
        return next_state.squeeze(0) if len(state.shape) == 2 else next_state
    
    def _rollout_trajectory(self, initial_state, controls):
        """
        Roll out the trajectory given initial state and control sequence.
        
        Args:
            initial_state: Starting state
            controls: Sequence of control inputs
        Returns:
            Full state trajectory
        """
        # Initialize trajectory storage
        trajectory = [initial_state]
        current_state = initial_state
        
        # Ensure controls have proper shape
        if not isinstance(controls, torch.Tensor):
            controls = torch.tensor(controls, dtype=torch.float32)
            
        # Roll out dynamics
        for control in controls:
            next_state = self._system_dynamics(current_state, control)
            trajectory.append(next_state)
            current_state = next_state
            
        return torch.stack(trajectory)
    
    def _compute_cost(self, trajectory, controls, original, target_class):
        """
        Compute cost function combining classification loss and control effort.
        
        Args:
            trajectory: Generated trajectory
            controls: Control inputs used
            original: Original trajectory
            target_class: Target classification
        Returns:
            Total cost
        """
        # Classification loss
        logits = self.model(trajectory.unsqueeze(0))
        class_loss = nn.CrossEntropyLoss()(logits, torch.tensor([target_class]))
        
        # Control effort (L2 norm of controls)
        control_cost = torch.norm(controls)
        
        # State deviation cost
        state_cost = torch.norm(trajectory - original)
        
        return (class_loss + 
                self.control_weight * control_cost + 
                self.state_weight * state_cost)
    
    def generate(self, x, target_class=None, X_train=None, y_train=None):
        """
        Generate counterfactual using dynamic systems optimization.
        
        Args:
            x: Input trajectory
            target_class: Target class
            X_train: Training data (not used)
            y_train: Training labels (not used)
        Returns:
            Counterfactual trajectory
        """
        # Ensure proper shape
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        if target_class is None:
            target_class = self._get_target_class(x)
        
        # Convert to tensor and get dimensions
        x_tensor = torch.tensor(x, dtype=torch.float32)
        n_features = x_tensor.shape[0]
        n_timesteps = x_tensor.shape[1]
        
        # Initialize smaller control sequence
        controls = torch.zeros((n_timesteps-1, n_features), 
                             dtype=torch.float32, requires_grad=True)
        
        # Scale controls to prevent large perturbations
        controls.data *= 0.1
        
        # Setup optimizer
        optimizer = optim.Adam([controls], lr=self.learning_rate)
        
        # Optimization loop
        for i in range(self.max_iter):
            optimizer.zero_grad()
            
            # Generate trajectory
            trajectory = self._rollout_trajectory(x_tensor[0], controls)
            
            # Compute cost
            cost = self._compute_cost(trajectory, controls, x_tensor, target_class)
            
            # Backpropagate and update
            cost.backward()
            optimizer.step()
            
            # Check if valid counterfactual found
            if i % 10 == 0:
                with torch.no_grad():
                    cf = trajectory.numpy()
                    if self._is_valid_cf(cf, x, target_class):
                        return cf.reshape(1, cf.shape[0], cf.shape[1])
        
        return None
