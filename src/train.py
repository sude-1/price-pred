import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_prep import PriceDataset
from model import PriceLSTM
import joblib
import os
import matplotlib.pyplot as plt

# Parametreler
file_path = "data/aapl.csv"
window_size = 10
batch_size = 16
num_epochs = 20
learning_rate = 0.001

# Dataset ve DataLoader
dataset = PriceDataset(file_path, window_size=window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
input_size = 1
hidden_size = 64
num_layers = 2
model = PriceLSTM(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []

# Eğitim Döngüsü
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs = inputs.unsqueeze(-1)  # (batch, seq_len, 1)
        targets = targets.unsqueeze(-1)  # (batch, 1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Model ve scaler kaydet
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/price_lstm.pth")
joblib.dump(dataset.scaler, "models/scaler.pkl")
print("Model and scaler has been saved.")

# Loss grafiği kaydet
os.makedirs("results", exist_ok=True)
plt.figure()
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss Over Epochs")
plt.savefig("results/training_loss.png")
plt.close()
print("Training loss graph saved into 'results/training_loss.png' file.")
