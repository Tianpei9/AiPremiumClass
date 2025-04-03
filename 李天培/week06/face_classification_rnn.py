import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the Olivetti faces dataset
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# Reshape the data for RNN input (batch_size, sequence_length, input_size)
X = X.reshape(-1, 64, 64)  # Each image is treated as a sequence of 64 time steps with 64 features

# Convert to PyTorch tensors
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define different RNN models
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, model_name, train_loader, test_loader, criterion, optimizer, num_epochs, writer):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100 * correct / total
        
        # Log metrics to TensorBoard
        writer.add_scalar(f'{model_name}/Train Loss', train_loss, epoch)
        writer.add_scalar(f'{model_name}/Validation Loss', val_loss, epoch)
        writer.add_scalar(f'{model_name}/Train Accuracy', train_acc, epoch)
        writer.add_scalar(f'{model_name}/Validation Accuracy', val_acc, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

def main():
    # Hyperparameters
    input_size = 64
    hidden_size = 128
    num_classes = 40
    num_epochs = 50
    learning_rate = 0.001

    # Initialize TensorBoard writer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f'runs/face_classification/{current_time}'
    writer = SummaryWriter(log_dir)

    # Initialize models
    models = {
        'SimpleRNN': SimpleRNN(input_size, hidden_size, num_classes),
        'LSTM': LSTMModel(input_size, hidden_size, num_classes),
        'GRU': GRUModel(input_size, hidden_size, num_classes)
    }

    # Train each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_model(model, model_name, train_loader, test_loader, criterion, optimizer, num_epochs, writer)

    writer.close()

if __name__ == "__main__":
    main() 