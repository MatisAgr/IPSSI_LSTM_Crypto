# ===============================================
# BTC 1-min -> 1h LSTM Forecast
# ===============================================
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Utilitaires

# MinMaxScaler (sklearn or fallback)
try:
    from sklearn.preprocessing import MinMaxScaler
except Exception:
    print("[INFO] scikit-learn not available, using lightweight MinMaxScaler fallback.")
    class MinMaxScaler:
        def __init__(self, feature_range=(0, 1)):
            self.feature_range = feature_range
            self.data_min_ = None
            self.data_max_ = None
            self.scale_ = None
            self.min_ = None
        def fit(self, X):
            X = np.asarray(X, dtype=np.float32).reshape(-1, 1)
            self.data_min_ = np.nanmin(X, axis=0)
            self.data_max_ = np.nanmax(X, axis=0)
            denom = np.where((self.data_max_ - self.data_min_) == 0.0,
                             1.0, (self.data_max_ - self.data_min_))
            fr0, fr1 = self.feature_range
            self.scale_ = (fr1 - fr0) / denom
            self.min_ = fr0 - self.data_min_ * self.scale_
            return self
        def transform(self, X):
            X = np.asarray(X, dtype=np.float32).reshape(-1, 1)
            return X * self.scale_ + self.min_
        def fit_transform(self, X):
            return self.fit(X).transform(X)
        def inverse_transform(self, X):
            X = np.asarray(X, dtype=np.float32).reshape(-1, 1)
            return (X - self.min_) / self.scale_

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load Kaggle dataset (btcusd_1-min_data.csv)
print("Downloading/loading Kaggle dataset...")
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
csv_path = Path(path) / "btcusd_1-min_data.csv"
print("Using CSV:", csv_path)
df_raw = pd.read_csv(csv_path)

# Identify timestamp column
time_col = None
for c in ["Timestamp", "TimeStamp"]:
    if c in df_raw.columns:
        time_col = c
        break
if time_col is None:
    raise ValueError("No timestamp column found.")

# Parse timestamp to UTC
if pd.api.types.is_numeric_dtype(df_raw[time_col]):
    df_raw[time_col] = pd.to_datetime(df_raw[time_col], unit="s", utc=True)
else:
    df_raw[time_col] = pd.to_datetime(df_raw[time_col], errors="coerce", utc=True)

# Ensure numeric
for col in ["Open", "High", "Low", "Close", "Volume"]:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

# Build time-indexed DataFrame
btc = (df_raw.dropna(subset=[time_col])
             .sort_values(time_col)
             .set_index(time_col))
print("Loaded rows (1-min):", len(btc))

# Resample to 1h
btc_1h = btc.resample("1h").agg({
    "Open": "first",
    "High": "max",
    "Low":  "min",
    "Close":"last",
    "Volume":"sum"
}).dropna()
print("btc_1h shape:", btc_1h.shape)
print("Time range:", btc_1h.index.min(), "->", btc_1h.index.max())

# Optional: limit to last N hours to speed up training (set to None to use all)
N_LAST = None  # e.g., 40000 for ~4.5 years
if N_LAST is not None and len(btc_1h) > N_LAST:
    btc_1h = btc_1h.iloc[-N_LAST:]
    print("Trimmed to last", len(btc_1h), "hours.")

# Prepare target series (Close) and temporal split
close = btc_1h["Close"].astype(np.float32).values.reshape(-1, 1)
n = len(close)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)
train = close[:train_end]
val   = close[train_end:val_end]
test  = close[val_end:]

print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

# Scaling (fit on train only)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
val_scaled   = scaler.transform(val)
test_scaled  = scaler.transform(test)

# Sequence generators
LOOK_BACK   = 72     # 72 hours (~3 days of context)
BATCH_SIZE  = 128    # batch per epoque
EPOCHS      = 10     # epoque
HORIZON     = 1      # predict next step

train_gen = TimeseriesGenerator(train_scaled, train_scaled, length=LOOK_BACK, batch_size=BATCH_SIZE)
val_gen   = TimeseriesGenerator(val_scaled,   val_scaled,   length=LOOK_BACK, batch_size=BATCH_SIZE)
test_gen  = TimeseriesGenerator(test_scaled,  test_scaled,  length=LOOK_BACK, batch_size=1)

print(f"Windows: train={len(train_gen)}, val={len(val_gen)}, test={len(test_gen)}")

# LSTM model (simple + regularization to reduce overfitting)
model = Sequential([
    LSTM(64, input_shape=(LOOK_BACK, 1)),           # single LSTM is often enough for baseline
    Dropout(0.2),                                   # dropout to regularize
    Dense(16, activation="relu", kernel_regularizer=l2(1e-4)),
    Dense(1)                                        # linear output for regression
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1),
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks
)

# Evaluate and predict on TEST only
test_loss = model.evaluate(test_gen, verbose=0)
print(f"Test MSE (scaled): {test_loss:.6f}")

pred_scaled = model.predict(test_gen, verbose=0)          # shape (len(test_gen), 1)
pred = scaler.inverse_transform(pred_scaled).ravel()       # back to USD
y_true = scaler.inverse_transform(test_scaled[LOOK_BACK:]).ravel()

# Metrics
rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))
mae  = float(np.mean(np.abs(pred - y_true)))
print(f"RMSE: {rmse:.2f} USD | MAE: {mae:.2f} USD")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_true, label="True Close Value")
plt.plot(pred,  label="Predicted Value", alpha=0.8)
plt.title("BTC Close 1h — Test set: true vs predicted (split, 10 epochs)")
plt.xlabel("Time steps (hours in test segment)")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# Commentaire et difficultés rencontrés
# - V1 : Overfitting Claire -> La courbe prédite exactement superposée aux valeurs réelles 
# - V2 : Over Fiting Réduit -> Je veux désormais voir ce que donne le modèle sur les 62 derniers jours

# Entraînement additionnel sur les 62 DERNIERS JOURS (62*24h)
SLICE_DAYS = 62
subset_hours = SLICE_DAYS * 24

if len(btc_1h) >= subset_hours:
    btc_1h_62 = btc_1h.iloc[-subset_hours:].copy()
    print(f"[62D] Subset shape: {btc_1h_62.shape}, range: {btc_1h_62.index.min()} -> {btc_1h_62.index.max()}")

    # Série cible (Close) sur 62 jours
    close_62 = btc_1h_62["Close"].astype(np.float32).values.reshape(-1, 1)
    n62 = len(close_62)
    train_end_62 = int(0.70 * n62)
    val_end_62   = int(0.85 * n62)

    train_62 = close_62[:train_end_62]
    val_62   = close_62[train_end_62:val_end_62]
    test_62  = close_62[val_end_62:]

    print(f"[62D] Split: train={len(train_62)}, val={len(val_62)}, test={len(test_62)}")

    # Scaling (fit sur train_62 uniquement)
    scaler_62 = MinMaxScaler(feature_range=(0, 1))
    train_scaled_62 = scaler_62.fit_transform(train_62)
    val_scaled_62   = scaler_62.transform(val_62)
    test_scaled_62  = scaler_62.transform(test_62)

    # Générateurs
    train_gen_62 = TimeseriesGenerator(train_scaled_62, train_scaled_62, length=LOOK_BACK, batch_size=BATCH_SIZE)
    val_gen_62   = TimeseriesGenerator(val_scaled_62,   val_scaled_62,   length=LOOK_BACK, batch_size=BATCH_SIZE)
    test_gen_62  = TimeseriesGenerator(test_scaled_62,  test_scaled_62,  length=LOOK_BACK, batch_size=1)

    print(f"[62D] Windows: train={len(train_gen_62)}, val={len(val_gen_62)}, test={len(test_gen_62)}")

    # Modèle spécifique 62 jours (j'ai récupéré l'architecture précédente avec
    #                               une couche CNN pour améliorer mes résultats)
    tf.random.set_seed(42); np.random.seed(42)
    model_62 = Sequential([
        # Conv1D capte les motifs locaux (fenêtre de 5 heures)
        Conv1D(filters=256, kernel_size=5, padding="causal", activation="relu",input_shape=(LOOK_BACK, 1)),
        LSTM(64),
        Dropout(0.2),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-4)),
        Dense(1)
    ])
    model_62.compile(optimizer="adam", loss="mean_squared_error")

    callbacks_62 = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1),
    ]

    history_62 = model_62.fit(
        train_gen_62,
        validation_data=val_gen_62,
        epochs=EPOCHS,         # 10 époques toujours
        verbose=1,
        callbacks=callbacks_62
    )

    # Évaluation + prédiction (test 62j)
    test_loss_62 = model_62.evaluate(test_gen_62, verbose=0)
    print(f"[62D] Test MSE (scaled): {test_loss_62:.6f}")

    pred_scaled_62 = model_62.predict(test_gen_62, verbose=0)
    pred_62 = scaler_62.inverse_transform(pred_scaled_62).ravel()
    y_true_62 = scaler_62.inverse_transform(test_scaled_62[LOOK_BACK:]).ravel()

    # Métriques 62 jours
    rmse_62 = float(np.sqrt(np.mean((pred_62 - y_true_62) ** 2)))
    mae_62  = float(np.mean(np.abs(pred_62 - y_true_62)))
    print(f"[62D] RMSE: {rmse_62:.2f} USD | MAE: {mae_62:.2f} USD")

    # Plot 62 jours
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_62, label="True Close (test, 62j)")
    plt.plot(pred_62,  label="Predicted (test, 62j)", alpha=0.8)
    plt.title("BTC Close 1h — Test set (LAST 62 days): true vs predicted (10 epochs)")
    plt.xlabel("Time steps (hours in 62-day test segment)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print(f"[62D] Not enough data to take last {SLICE_DAYS} days. Available hours: {len(btc_1h)}")

# - V3 : Le CNN a considérablement améliorer les patternes de ma courbe de prédiction
# Difficultés rencontrés : a un moment donné j'ai voulu rajouté une couche de normalisation
# Tout comme je n'ai pas su paramétré le nombre de mes filtres : les résultats étaient dégeulasse avec 32
# Puis beaucoup meilleur avec 64 | 128 filtres.