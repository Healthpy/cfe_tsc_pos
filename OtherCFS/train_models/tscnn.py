import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import accuracy_score
import os
import torch.nn.functional as F
import pandas as pd
import json

class TimeSeriesCNN(nn.Module):
    def __init__(self, n_features, n_classes, seq_length):
        super(TimeSeriesCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size after pooling
        self.flat_size = 128 * (seq_length // 4)
        
        self.fc1 = nn.Linear(self.flat_size, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        # x shape: (batch_size, n_features, seq_length)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class BayesianLSTM(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=64, n_layers=2, dropout=0.1):
        super(BayesianLSTM, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Bayesian linear layers
        self.fc_mu = nn.Linear(hidden_size, n_classes)
        self.fc_var = nn.Linear(hidden_size, n_classes)
        
    def forward(self, x, n_samples=1):
        # Run LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last time step output
        last_output = lstm_out[:, -1, :]
        
        # Get mean and log variance
        mu = self.fc_mu(last_output)
        log_var = self.fc_var(last_output)
        
        # Sample using reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        # Generate multiple samples if requested
        samples = []
        for _ in range(n_samples):
            sample = mu + eps * std
            samples.append(F.softmax(sample, dim=1))
            
        return torch.stack(samples)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class LSTMFCN(nn.Module):
    def __init__(self, n_features, n_classes, max_seq_len=None, lstm_units=128):
        super(LSTMFCN, self).__init__()
        
        self.num_classes = n_classes
        self.num_features = n_features
        self.max_seq_len = max_seq_len
        self.num_lstm_out = lstm_units
        
        # LSTM
        self.lstm = nn.LSTM(input_size=self.num_features, 
                           hidden_size=self.num_lstm_out,
                           num_layers=1,
                           batch_first=True)
        
        # FCN - now directly using input shape (batch, feature, seq_length)
        self.conv1 = nn.Conv1d(self.num_features,128, kernel_size=8)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3)
        
        # Batch Norm
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
        # SE layers
        self.se1 = SELayer(128)
        self.se2 = SELayer(256)
        
        # Dropout
        self.lstm_drop = nn.Dropout(0.8)
        self.fc_drop = nn.Dropout(0.3)
        
        # Final classifier
        self.fc = nn.Linear(128 + self.num_lstm_out, self.num_classes)
        
    def forward(self, x):
        # x shape: (batch, feature, seq_length)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Store the original shape for FCN branch
        x_orig = x  # Already in (batch, feature, seq_length)
        
        # For LSTM branch, transpose to (batch, seq_length, feature)
        x1 = x.transpose(1, 2)
        x1, (ht, ct) = self.lstm(x1)
        x1 = x1[:, -1, :]  # Take last time step
        x1 = self.lstm_drop(x1)
        
        # FCN branch - input already in correct shape (batch, feature, seq_length)
        x2 = self.fc_drop(F.relu(self.bn1(self.conv1(x_orig))))
        x2 = self.se1(x2)
        
        x2 = self.fc_drop(F.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        
        x2 = self.fc_drop(F.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)  # Global average pooling
        
        # Concatenate and classify
        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)
        
        return x_out

    def predict_proba(self, x):
        """Get probability predictions."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs.cpu().numpy()


def build_tf_model(n_features, n_classes, seq_length):
    inputs = keras.Input(shape=(seq_length, n_features))
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


class ModelWrapper:
    def __init__(self, model=None, framework='pytorch', n_features=None, n_classes=None, seq_length=None, model_type='lstmfcn'):
        self.framework = framework
        self.model_type = model_type
        
        if model is not None:
            self.model = model
        elif n_features is not None and n_classes is not None and seq_length is not None:
            if framework == 'pytorch':
                if model_type == 'lstmfcn':
                    self.model = LSTMFCN(n_features, n_classes)
                elif model_type == 'cnn':
                    self.model = TimeSeriesCNN(n_features, n_classes, seq_length)
                elif model_type == 'bayesian':
                    self.model = BayesianLSTM(n_features, n_classes)
        else:
            self.model = None
            
        self.best_acc = 0.0
        self.model_path = os.path.join('saved_models', framework, model_type)
        os.makedirs(self.model_path, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training = False  # Add training flag
        self.results_path = os.path.join('results', framework, model_type)
        os.makedirs(self.results_path, exist_ok=True)

    def get_model_config(self):
        """Get model configuration for saving"""
        if self.model_type == 'cnn':
            return {
                'model_state_dict': self.model.state_dict(),
                'n_features': self.model.conv1.in_channels,
                'n_classes': self.model.fc2.out_features,
                'seq_length': self.model.flat_size // 128 * 4
            }
        elif self.model_type == 'bayesian':
            return {
                'model_state_dict': self.model.state_dict(),
                'n_features': self.model.n_features,
                'n_classes': self.model.fc_mu.out_features,
                'hidden_size': self.model.hidden_size,
                'n_layers': self.model.n_layers
            }
        elif self.model_type == 'lstmfcn':
            return {
                'model_state_dict': self.model.state_dict(),
                'n_features': self.model.num_features,
                'n_classes': self.model.num_classes,
                'lstm_units': self.model.num_lstm_out,
                'dropout': self.model.lstm_drop.p
            }
        return None

    def load_model_from_config(self, config):
        """Create and load model from configuration"""
        if self.model_type == 'cnn':
            self.model = TimeSeriesCNN(
                config['n_features'],
                config['n_classes'],
                config['seq_length']
            )
        elif self.model_type == 'bayesian':
            self.model = BayesianLSTM(
                config['n_features'],
                config['n_classes'],
                hidden_size=config.get('hidden_size', 64),
                n_layers=config.get('n_layers', 2)
            )
        elif self.model_type == 'lstmfcn':
            self.model = LSTMFCN(
                config['n_features'],
                config['n_classes'],
                lstm_units=config.get('lstm_units', 128)
            )
            self.model.load_state_dict(config['model_state_dict'])

    def save_model(self, dataset_name):
        """Save the model to disk with framework and model type in filename"""
        base_path = f'{self.model_path}/{dataset_name}'
        if self.framework == 'pytorch':
            config = self.get_model_config()
            if config:
                model_file = f'{base_path}_model.pth'
                torch.save(config, model_file)
                print(f"Saved {self.framework} {self.model_type} model to {model_file}")
    
    def load_model(self, dataset_name):
        """Load the model from disk using framework and model type specific path"""
        base_path = f'{self.model_path}/{dataset_name}'
        model_file = f'{base_path}_model.pth'
        
        if os.path.exists(model_file):
            config = torch.load(model_file)
            self.load_model_from_config(config)
            print(f"Loaded {self.framework} {self.model_type} model from {model_file}")
            return True
        return False

    def save_training_results(self, dataset_name, epoch_metrics):
        results_file = os.path.join(self.results_path, f'{dataset_name}_training.csv')
        pd.DataFrame(epoch_metrics).to_csv(results_file, index=False)
        print(f"Saved training results to {results_file}")

    def save_prediction_results(self, dataset_name, y_true, y_pred, probas=None):
        results_file = os.path.join(self.results_path, f'{dataset_name}_predictions.csv')
        results = {'true_labels': y_true, 'predicted_labels': y_pred}
        if probas is not None:
            for i in range(probas.shape[1]):
                results[f'class_{i}_prob'] = probas[:, i]
        pd.DataFrame(results).to_csv(results_file, index=False)
        print(f"Saved prediction results to {results_file}")

    def train_model(self, X_train, y_train, dataset_name, batch_size=32, epochs=100):
        """
        Train a CNN model on time series data.
        
        Args:
            X_train: Training data with shape (n_samples, seq_length, n_features)
            y_train: Training labels
            batch_size: Batch size
            epochs: Number of epochs
            
        Returns:
            Self (for method chaining)
        """
        if self.load_model(dataset_name):
            print(f"Loaded pretrained model for {dataset_name}")
            return self

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Save model configuration
        config = {
            'n_features': X_train.shape[1] if len(X_train.shape) > 2 else 1,
            'n_classes': len(np.unique(y_train)),
            'seq_length': X_train.shape[2] if len(X_train.shape) > 2 else X_train.shape[1],
            'framework': self.framework,
            'model_type': self.model_type
        }
        
        config_path = f'{self.model_path}/{dataset_name}_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Training loop
        best_acc = 0.0
        epoch_metrics = []
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Evaluate accuracy
            self.model.eval()
            current_acc = self.accuracy_score(X_train, y_train)
            
            # Save if better
            if current_acc > best_acc:
                best_acc = current_acc
                self.save_model(dataset_name)
            
            metrics = {
                'epoch': epoch + 1,
                'loss': running_loss/len(train_loader),
                'accuracy': current_acc
            }
            epoch_metrics.append(metrics)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {metrics["loss"]:.4f}, Acc: {metrics["accuracy"]:.4f}')
        
        self.save_training_results(dataset_name, epoch_metrics)
        return self
    
    def predict(self, X, y_true=None, dataset_name=None):
        """
        Make predictions on new data.
        
        Args:
            X: Input data with shape (n_samples, seq_length, n_features)
            
        Returns:
            Predicted class labels
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
            
        self.model.eval()
        with torch.no_grad():
            if len(X.shape) == 2:
                X = X.unsqueeze(0)  # Add batch dimension
                
            outputs = self.model(X)
            probas = F.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = torch.max(outputs, 1)
            predictions = predicted.cpu().numpy()
            
            if dataset_name is not None and y_true is not None:
                self.save_prediction_results(dataset_name, y_true, predictions, probas)
            
            return predictions
    
    def accuracy_score(self, X, y_true):
        """
        Calculate accuracy score for predictions.
        
        Args:
            X: Input data
            y_true: True labels
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)
    
    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
            
        self.model.eval()
        with torch.no_grad():
            if len(X.shape) == 2:
                X = X.unsqueeze(0)  # Add batch dimension
                
            outputs = self.model(X)
            return F.softmax(outputs, dim=1).cpu().numpy()

    def predict_with_uncertainty(self, X, n_samples=10):
        """Make predictions with uncertainty estimates."""
        if not isinstance(self.model, BayesianLSTM):
            raise ValueError("Model must be BayesianLSTM for uncertainty estimation")
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            samples = self.model(X_tensor, n_samples=n_samples)
            
            # Mean prediction across samples
            mean_probs = samples.mean(dim=0)
            
            # Standard deviation across samples
            std_probs = samples.std(dim=0)
            
            return mean_probs.numpy(), std_probs.numpy()
        
    def train(self, mode=True):
        """Set training mode."""
        self.training = mode
        if hasattr(self.model, 'train'):
            self.model.train(mode)
        return self
        
    def eval(self):
        """Set evaluation mode."""
        self.training = False
        if hasattr(self.model, 'eval'):
            self.model.eval()
        return self
        
    def to(self, device):
        """Move model to device."""
        self.device = device
        if hasattr(self.model, 'to'):
            self.model.to(device)
        return self

    def __call__(self, x):
        """Make model wrapper compatible with torchattacks."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # For all PyTorch models, use proper gradient handling
        if self.training:
            outputs = self.model(x)
        else:
            with torch.enable_grad():
                outputs = self.model(x)
        
        return outputs  # LSTMFCN already returns softmax probabilities

    def forward(self, x):
        """Required for torchattacks compatibility."""
        return self.__call__(x)

def create_model(n_features, n_classes, seq_length, framework='pytorch', model_type='lstmfcn'):
    """
    Factory function to create a model wrapper.
    
    Args:
        n_features: Number of features in the input data
        n_classes: Number of output classes
        seq_length: Length of the time series sequence
        framework: 'pytorch' or 'tensorflow'
        
    Returns:
        Initialized ModelWrapper instance
    """
    return ModelWrapper(
        framework=framework,
        n_features=n_features,
        n_classes=n_classes,
        seq_length=seq_length,
        model_type=model_type
    )

def load_model_from_config(config_path):
    """Load a model from its configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set default values for missing parameters
    default_config = {
        'n_features': 1,
        'n_classes': 2,
        'seq_length': None,
        'framework': 'pytorch',
        'model_type': 'lstmfcn'
    }
    
    # Update defaults with provided config
    for key in default_config:
        if key not in config:
            config[key] = default_config[key]
    
    return create_model(
        n_features=config['n_features'],
        n_classes=config['n_classes'],
        seq_length=config['seq_length'],
        framework=config['framework'],
        model_type=config['model_type']
    )
