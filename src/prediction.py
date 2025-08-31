import torch
from data_prep import PriceDataset
from model import PriceLSTM
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def predict_next_day():
    # Veri setini yükle
    dataset = PriceDataset("data/aapl.csv", window_size=10)

    # Model ve scaler yükle
    input_size = 1
    hidden_size = 64
    num_layers = 2
    model = PriceLSTM(input_size, hidden_size, num_layers, output_size=1)
    model.load_state_dict(torch.load("models/price_lstm.pth"))
    model.eval()

    scaler = joblib.load("models/scaler.pkl")

    # Son pencereyi al
    last_window = dataset.prices[-dataset.window_size:]
    last_window_tensor = torch.tensor(last_window.reshape(1, dataset.window_size, 1), dtype=torch.float32)

    # 1 Günlük Tahmin
    with torch.no_grad():
        pred = model(last_window_tensor).item()
    pred_price = scaler.inverse_transform(np.array(pred).reshape(-1, 1))[0][0]

    print(f"Estimated next day price: {pred_price:.2f} USD")

    # ----------------------------------------------------
    # Multi-step forecast (ör: 7 gün ileri)
    future_days = 7
    future_preds = []
    for _ in range(future_days):
        with torch.no_grad():
            pred = model(last_window_tensor).item()
        pred_price = scaler.inverse_transform(np.array(pred).reshape(-1, 1))[0][0]
        future_preds.append(pred_price)

        # pencereyi güncelle
        new_value = np.array(pred).reshape(1, 1, 1)
        last_window_tensor = torch.cat(
            [last_window_tensor[:, 1:, :], torch.tensor(new_value, dtype=torch.float32)],
            dim=1
        )

    print("Future predictions:", future_preds)

    # ----------------------------------------------------
    # Gerçek fiyatlar ve tahminleri grafikle göster
    df = pd.read_csv("data/aapl.csv")
    dates = pd.to_datetime(df.iloc[-30:, 0])  # son 30 gün tarihleri
    actual_prices = dataset.scaler.inverse_transform(dataset.prices[-30:].reshape(-1, 1)).flatten()

    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual_prices, label="Real Prices")

    # 1 günlük tahmin noktası
    plt.scatter(dates.iloc[-1] + pd.Timedelta(days=1), pred_price,
                color="red", marker="x", s=100, label="1-Day Prediction")

    # 7 günlük tahminleri ekle
    future_dates = pd.date_range(start=dates.iloc[-1] + pd.Timedelta(days=1), periods=future_days, freq="D")
    plt.plot(future_dates, future_preds, linestyle="--", color="orange", label="7-Day Prediction")

    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("AAPL Price Prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/future_predictions.png")
    plt.show()

    # ----------------------------------------------------
    # Tahminleri CSV'ye kaydet
    results = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": future_preds
    })
    results.to_csv("results/future_predictions.csv", index=False)
    print("Predictions saved into 'results/' folder.")


if __name__ == "__main__":
    predict_next_day()
