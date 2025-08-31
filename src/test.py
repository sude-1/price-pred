import torch
import matplotlib.pyplot as plt
from data_prep import PriceDataset
from model import PricePredictor

# Hiperparametreler
window_size = 5

# Dataset yükle
dataset = PriceDataset("data/aapl.csv", window_size=window_size)

# Model yükle
model = PricePredictor(input_size=1, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load("outputs/model.pth"))
model.eval()

# Tüm veriyi kullanarak tahmin yap
predictions = []
targets = []

with torch.no_grad():
    for i in range(len(dataset)):
        x, y = dataset[i]
        x = x.unsqueeze(0)  # batch dimension ekle
        pred = model(x).item()
        predictions.append(pred)
        targets.append(y.item())

# Sonuçları çiz
plt.figure(figsize=(12, 6))
plt.plot(targets, label="Gerçek Fiyat", color="blue")
plt.plot(predictions, label="Tahmin", color="red")
plt.legend()
plt.title("Apple AAPL Hisse Fiyatı - Gerçek vs Tahmin")
plt.xlabel("Gün")
plt.ylabel("Fiyat (Normalize)")
plt.show()
