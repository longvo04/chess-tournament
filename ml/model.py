"""
Neural Network Models cho Chess Position Evaluation
Hỗ trợ cả PyTorch và model đơn giản (chỉ cần NumPy)
"""
import numpy as np
import os

# Kiểm tra PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[Info] PyTorch không có sẵn. Sử dụng SimpleModel.")


if TORCH_AVAILABLE:
    
    class ChessNet(nn.Module):
        """
        Fully Connected Neural Network cho chess evaluation
        Input: 773 features
        Output: 1 (win probability for white: 0-1)
        """
        
        def __init__(self, input_size: int = 773, 
                     hidden_sizes: list = None,
                     dropout: float = 0.3):
            super(ChessNet, self).__init__()
            
            if hidden_sizes is None:
                hidden_sizes = [2048, 1024, 512, 256, 128]  # Deeper network
            
            layers = []
            prev_size = input_size
            
            for i, hidden_size in enumerate(hidden_sizes):
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.LeakyReLU(0.1))  # LeakyReLU thay vì ReLU
                if i < len(hidden_sizes) - 1:
                    layers.append(nn.Dropout(dropout))
                prev_size = hidden_size
            
            self.hidden = nn.Sequential(*layers)
            self.output = nn.Linear(prev_size, 1)
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            x = self.hidden(x)
            x = torch.sigmoid(self.output(x))
            return x
    
    
    class ChessResNet(nn.Module):
        """
        ResNet-style architecture với skip connections
        Tốt hơn cho training deep networks
        """
        
        def __init__(self, input_size: int = 773):
            super(ChessResNet, self).__init__()
            
            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
            
            # Residual blocks
            self.res1 = self._make_res_block(512, 512)
            self.res2 = self._make_res_block(512, 256)
            self.res3 = self._make_res_block(256, 128)
            
            # Skip connections for dimension changes
            self.skip2 = nn.Linear(512, 256)
            self.skip3 = nn.Linear(256, 128)
            
            # Output
            self.output = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def _make_res_block(self, in_features, out_features):
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(out_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        
        def forward(self, x):
            # Input projection
            x = self.input_proj(x)
            
            # Res block 1 (512 -> 512)
            identity = x
            x = self.res1(x)
            x = F.relu(x + identity)
            
            # Res block 2 (512 -> 256)
            identity = self.skip2(x)
            x = self.res2(x)
            x = F.relu(x + identity)
            
            # Res block 3 (256 -> 128)
            identity = self.skip3(x)
            x = self.res3(x)
            x = F.relu(x + identity)
            
            # Output
            x = self.output(x)
            return x


class SimpleMLModel:
    """
    Model đơn giản sử dụng Logistic Regression + Feature Engineering
    Không cần PyTorch - chạy được trên mọi máy
    """
    
    def __init__(self, input_size: int = 773):
        self.input_size = input_size
        self.weights = None
        self.bias = 0.0
        
        # Regularization
        self.l2_lambda = 0.0001
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, 
            learning_rate: float = 0.01,
            batch_size: int = 256,
            verbose: bool = True) -> dict:
        """
        Training với mini-batch gradient descent
        """
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features).astype(np.float32) * 0.01
        self.bias = 0.0
        
        history = {'loss': [], 'accuracy': []}
        
        # Learning rate decay
        initial_lr = learning_rate
        
        for epoch in range(epochs):
            # Decay learning rate
            lr = initial_lr / (1 + epoch * 0.01)
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                linear = np.dot(X_batch, self.weights) + self.bias
                predictions = self._sigmoid(linear)
                
                # Loss (Binary Cross Entropy + L2 regularization)
                eps = 1e-7
                bce_loss = -np.mean(
                    y_batch * np.log(predictions + eps) + 
                    (1 - y_batch) * np.log(1 - predictions + eps)
                )
                l2_loss = self.l2_lambda * np.sum(self.weights ** 2)
                batch_loss = bce_loss + l2_loss
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward
                error = predictions - y_batch
                dw = (1/len(X_batch)) * np.dot(X_batch.T, error)
                dw += 2 * self.l2_lambda * self.weights  # L2 regularization
                db = (1/len(X_batch)) * np.sum(error)
                
                # Update with momentum (simplified)
                self.weights -= lr * dw
                self.bias -= lr * db
            
            avg_loss = epoch_loss / n_batches
            
            # Accuracy on full dataset (every 10 epochs)
            if epoch % 10 == 0 or epoch == epochs - 1:
                predictions = self._sigmoid(np.dot(X, self.weights) + self.bias)
                accuracy = np.mean((predictions > 0.5) == (y > 0.5))
                
                history['loss'].append(avg_loss)
                history['accuracy'].append(accuracy)
                
                if verbose:
                    print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}, LR={lr:.6f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict win probability"""
        if self.weights is None:
            raise ValueError("Model chưa được train!")
        linear = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear)
    
    def save(self, path: str):
        """Lưu model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, 
                 weights=self.weights, 
                 bias=np.array([self.bias]),
                 input_size=np.array([self.input_size]))
        print(f"SimpleModel saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        data = np.load(path)
        self.weights = data['weights']
        self.bias = float(data['bias'][0])
        self.input_size = int(data['input_size'][0])


def save_pytorch_model(model, path: str, optimizer=None, epoch=None, loss=None):
    """Lưu PyTorch model với metadata"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'input_size': 773,
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, path)
    print(f"PyTorch model saved to {path}")


def load_pytorch_model(path: str, device='cpu'):
    """Load PyTorch model"""
    checkpoint = torch.load(path, map_location=device)
    
    class_name = checkpoint.get('model_class', 'ChessNet')
    input_size = checkpoint.get('input_size', 773)
    
    if class_name == 'ChessResNet':
        model = ChessResNet(input_size=input_size)
    else:
        model = ChessNet(input_size=input_size)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint
