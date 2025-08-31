import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_prep import PriceDataset
from model import PriceLSTM
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def evaluate_model():

    dataset = PriceDataset("data/aapl.csv", window_size=10)

    X, y = dataset.X, dataset.y


    # Train/test split (%80 train - %20 test)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Modeli yükle

    model = PriceLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)

    model.load_state_dict(torch.load("models/price_lstm.pth"))
    model.eval()

    # Tahmin yap
    with torch.no_grad():
        X_test = X_test.unsqueeze(-1)  # (batch, seq_len, 1)
        predictions = model(X_test).squeeze().numpy()

    y_test = y_test.numpy()

    # Skaler yükle ve orijinal değerlere çevir
    scaler = joblib.load("models/scaler.pkl")
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Metrikler
    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, predictions_rescaled)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"MAE: {mae/np.mean(y_test_rescaled):.4f}")
    print(f"RMSE: {rmse/np.mean(y_test_rescaled):.4f}")
    print(f"R^2: {r2:.4f}")

    # Sonuçları kaydet
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    metrics_df = pd.DataFrame([{
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }])

    metrics_df.to_csv(os.path.join(results_dir, "evaluation_metrics.csv"), index=False)
    print("📊 Evaluation metrics saved to results/evaluation_metrics.csv")


    

    # Gerçek ve tahminleri karşılaştır
    plt.figure(figsize=(12,6))
    plt.plot(y_test, label="Real Prices", color="blue")
    plt.plot(predictions, label="Predictions", color="orange")

    plt.title("Real vs Predicted Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    # results klasörüne kaydet
    plt.savefig("results/evaluation_plot.png")
    plt.close()

    print("Real vs Prediction Graph has been saved as 'results/evaluation_plot.png'.")


if __name__ == "__main__":
    evaluate_model()
