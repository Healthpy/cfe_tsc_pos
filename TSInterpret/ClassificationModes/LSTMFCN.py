import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        
        self.return_features = False
        self.num_classes = n_classes
        self.num_features = n_features
        self.max_seq_len = max_seq_len
        self.num_lstm_out = lstm_units
        
        # LSTM
        self.lstm = nn.LSTM(input_size=self.num_features, 
                           hidden_size=self.num_lstm_out,
                           num_layers=2,
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

    def get_last_conv_layer(self):
        """Return the last convolutional layer for CAM."""
        return self.conv3
        
    def forward(self, x):
        # x shape: (batch, feature, seq_length)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Store the original shape for FCN branch
        x_orig = x  # Already in (batch, feature, seq_length)
        
        # For LSTM branch, transpose to (batch, seq_length, feature)
        x1 = x.transpose(1, 2)
        lstm_out, (h, c) = self.lstm(x1)
        x1 = lstm_out[:, -1, :]  # Take last time step
        x1 = self.lstm_drop(x1)
        
        # FCN branch - input already in correct shape (batch, feature, seq_length)
        x2 = self.fc_drop(F.relu(self.bn1(self.conv1(x_orig))))
        x2 = self.se1(x2)
        
        x2 = self.fc_drop(F.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        
        x2 = self.fc_drop(F.relu(self.bn3(self.conv3(x2))))

        # Save feature map and return if requested
        self.cam_feature_map = x2
        if self.return_features:
            return x2
        
        x2 = torch.mean(x2, 2)  # Global average pooling
        
        # Concatenate and classify
        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)
        
        return x_out
    
    def forward_features(self):
        """Enable feature map return mode."""
        self.return_features = True
        
    def forward_normal(self):
        """Disable feature map return mode."""
        self.return_features = False

    def predict_proba(self, x):
        """Get probability predictions."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs.cpu().numpy()