import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
import datetime

class WeatherDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_days=1):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.data = data
        
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_days + 1
    
    def __getitem__(self, idx):
        X = self.data[idx:idx + self.sequence_length]
        if self.prediction_days == 1:
            y = self.data[idx + self.sequence_length]
        else:
            y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_days]
        return torch.FloatTensor(X), torch.FloatTensor(y)

class WeatherRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(WeatherRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(file_path, sequence_length=30):
    # Load and preprocess data
    df = pd.read_csv(file_path)
    
    # Extract maximum temperature
    temp_max = df['MaxTemp'].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler()
    temp_max_scaled = scaler.fit_transform(temp_max)
    
    return temp_max_scaled, scaler

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, writer, prediction_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        
        # Log metrics to TensorBoard
        writer.add_scalar(f'{prediction_type}/Train Loss', train_loss, epoch)
        writer.add_scalar(f'{prediction_type}/Validation Loss', val_loss, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

def main():
    # Hyperparameters
    sequence_length = 30
    hidden_size = 64
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    # Initialize TensorBoard writer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f'runs/weather_prediction/{current_time}'
    writer = SummaryWriter(log_dir)
    
    # Prepare data
    data, scaler = prepare_data('weatherww2.csv', sequence_length)
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Create datasets and dataloaders for 1-day prediction
    train_dataset_1day = WeatherDataset(train_data, sequence_length, prediction_days=1)
    val_dataset_1day = WeatherDataset(val_data, sequence_length, prediction_days=1)
    
    train_loader_1day = DataLoader(train_dataset_1day, batch_size=batch_size, shuffle=True)
    val_loader_1day = DataLoader(val_dataset_1day, batch_size=batch_size)
    
    # Create datasets and dataloaders for 5-day prediction
    train_dataset_5day = WeatherDataset(train_data, sequence_length, prediction_days=5)
    val_dataset_5day = WeatherDataset(val_data, sequence_length, prediction_days=5)
    
    train_loader_5day = DataLoader(train_dataset_5day, batch_size=batch_size, shuffle=True)
    val_loader_5day = DataLoader(val_dataset_5day, batch_size=batch_size)
    
    # Train 1-day prediction model
    print("\nTraining 1-day prediction model...")
    model_1day = WeatherRNN(input_size=1, hidden_size=hidden_size, output_size=1)
    optimizer_1day = optim.Adam(model_1day.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_model(model_1day, train_loader_1day, val_loader_1day, criterion, 
                optimizer_1day, num_epochs, writer, "1-day prediction")
    
    # Train 5-day prediction model
    print("\nTraining 5-day prediction model...")
    model_5day = WeatherRNN(input_size=1, hidden_size=hidden_size, output_size=5)
    optimizer_5day = optim.Adam(model_5day.parameters(), lr=learning_rate)
    
    train_model(model_5day, train_loader_5day, val_loader_5day, criterion, 
                optimizer_5day, num_epochs, writer, "5-day prediction")
    
    writer.close()

if __name__ == "__main__":
    main() 