import scipy.fft as fft
import scipy
import os
import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Perform Fourier transform as a feature extractor for each .wav file
def create_fft(filename):
    s_rate, data = scipy.io.wavfile.read(filename)
    fft_features = abs(fft.fft(data)[:1000])
    return fft_features

# Custom PyTorch Dataset
class GenreDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load and preprocess data
def load_data(path, genres, max_per_genre=30):
    features = []
    labels = []
    for label, genre in enumerate(genres):
        genre_path = join(path, genre)
        files = [f for f in listdir(genre_path) if isfile(join(genre_path, f))][:max_per_genre]
        for file in files:
            file_path = join(genre_path, file)
            fft_features = create_fft(file_path)
            features.append(fft_features)
            labels.append(label)
    return np.array(features), np.array(labels)

# Normalize features
def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Define Logistic Regression model in PyTorch
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.float(), labels.long()
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Testing function
def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.float(), labels.long()
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Main function
def main():
    path = "Data/wav/genres_original"
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    # Load and prepare data
    features, labels = load_data(path, genres)
    features = normalize_features(features)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Convert data into PyTorch Dataset and DataLoader
    train_dataset = GenreDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = GenreDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model, loss function, and optimizer
    input_size = features.shape[1]
    num_classes = len(genres)
    model = LogisticRegressionModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train and test model
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    test_model(model, test_loader)

if __name__ == "__main__":
    main()
