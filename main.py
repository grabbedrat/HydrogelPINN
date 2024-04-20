import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the JSON data
with open('hydrogels.json') as file:
    data = json.load(file)

# Extract features
X = []
for sample in data:
    features = [
        ', '.join(sample['Hydrogel Materials']),
        ', '.join(sample['Molecule']),
        sample['Release Media'],
        sample['Volume of Release Media (mL)'],
        sample['Sampled Volume Refreshed'],
        sample['Enzyme'],
        sample['pH'],
        sample['Temperature (°C)'],
        sample['Agitation (rpm)']
    ]
    X.append(features)

# Convert data to numpy array
X = np.array(X)

# Preprocess the features
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[:, :-3]).toarray()

# Convert numerical features to float
numerical_features = []
for i in range(X.shape[0]):
    volume_values = X[i, -3].split(', ')
    volume_floats = []
    for volume in volume_values:
        try:
            volume_floats.append(float(volume))
        except ValueError:
            pass
    volume = np.mean(volume_floats) if volume_floats else np.nan
    
    pH_values = X[i, -2].split(', ')
    pH_floats = []
    for pH in pH_values:
        try:
            pH_floats.append(float(pH))
        except ValueError:
            pass
    pH = np.mean(pH_floats) if pH_floats else np.nan
    
    temperature = np.mean([float(t.split('±')[0].strip()) for t in X[i, -1].split(', ') if '±' in t])
    numerical_features.append([volume, pH, temperature])

numerical_features = np.array(numerical_features)

# Handle missing values
numerical_features = np.where(np.isnan(numerical_features), 0, numerical_features)

# Combine encoded categorical features and numerical features
X_preprocessed = np.concatenate((X_encoded, numerical_features), axis=1)

# Split the data into training and testing sets
X_train, X_test = train_test_split(X_preprocessed, test_size=0.2, random_state=42)

# Convert preprocessed features to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Build the model
model = NeuralNetwork(input_dim=X_train.shape[1])

# Define the optimizer
optimizer = optim.Adam(model.parameters())

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Training loop
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    # Forward pass
    output = model(X_train)
    
    # Compute the loss (Mean Squared Error)
    loss = criterion(output, torch.zeros_like(output))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    train_output = model(X_train)
    test_output = model(X_test)
    
    # Compute evaluation metrics
    train_loss = criterion(train_output, torch.zeros_like(train_output))
    test_loss = criterion(test_output, torch.zeros_like(test_output))
    
    print(f'Training Loss: {train_loss.item():.4f}')
    print(f'Testing Loss: {test_loss.item():.4f}')
    
    # Compute mean squared error
    train_mse = mean_squared_error(np.zeros_like(train_output.numpy()), train_output.numpy())
    test_mse = mean_squared_error(np.zeros_like(test_output.numpy()), test_output.numpy())
    
    print(f'Training MSE: {train_mse:.4f}')
    print(f'Testing MSE: {test_mse:.4f}')