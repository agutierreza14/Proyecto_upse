import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple

# === Configuración ===
PATH_DATA = "caracteristica_telco.csv"   
TIME_COL = "datetime"
GROUP_COL = "CellID"
TARGET_COL = "y_international"

# Variables numericas
BASE_FEATS = [
    "smsin","smsout","callin","callout","internet",
    "total_sms","total_call","total_traffic",
    "ratio_sms_out_in","ratio_call_out_in","ratio_sms_call",
    "internet_per_sms","internet_per_call",
    "share_sms","share_call","share_data",
    "any_sms","any_call","any_data",
    "weekend_sms","weekend_call","weekend_data",
    "night_sms","night_call","night_data",
    "hour","dayofweek","is_weekend","is_night","month",
    "log1p_total_sms","log1p_total_call","log1p_internet","log1p_total_traffic"
]

WINDOW = 4   # longitud de ventana T
STEP   = 1   # stride entre ventanas

def build_sequences(df: pd.DataFrame,
                    feature_cols: List[str],
                    time_col: str,
                    group_col: str,
                    target_col: str,
                    window: int,
                    step: int) -> Tuple[np.ndarray, np.ndarray]:
    
    X_list, y_list = [], []

    # Ordenamos por grupo y tiempo
    df = df.sort_values([group_col, time_col]).copy()

    # Por si datetime viene como string
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])

    # Solo columnas disponibles
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Trabajar por CellID
    for _, g in df.groupby(group_col):
        g = g.dropna(subset=[target_col])[feature_cols + [target_col, time_col]].reset_index(drop=True)

        # Necesitamos al menos window+1 puntos para sacar 1 ventana
        if len(g) < window + 1:
            continue

        feats = g[feature_cols].values.astype(float)
        target = g[target_col].values.astype(int)

        # Ventanas deslizantes
        for end in range(window, len(g), step):
            X_list.append(feats[end-window:end, :]) 
            y_list.append(target[end])               

    if not X_list:
        raise RuntimeError("No se construyeron secuencias: revisa WINDOW/STEP o la densidad temporal.")

    X = np.stack(X_list, axis=0)  # (N, T, F)
    y = np.array(y_list)          # (N,)
    return X, y

def standardize_by_train(X_train: np.ndarray, X_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  
    # X: (N, T, F)
    mu  = X_train.mean(axis=(0,1), keepdims=True)      # (1,1,F)
    std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
    X_train_s = (X_train - mu) / std
    X_other_s = (X_other - mu) / std
    return X_train_s, X_other_s

if __name__ == "__main__":
    df = pd.read_csv(PATH_DATA, low_memory=False)

    # Asegurar columnas clave
    needed = [TIME_COL, GROUP_COL, TARGET_COL]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: {c}")

    # Construir secuencias
    X, y = build_sequences(
        df, BASE_FEATS, TIME_COL, GROUP_COL, TARGET_COL,
        window=WINDOW, step=STEP
    )
    print("X shape:", X.shape, " y shape:", y.shape)  # (N, T, F) y (N,)

    # Split estratificado en base a y (mantener proporción de clases)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Standardize por estadísticas de train
    X_tr_s, X_te_s = standardize_by_train(X_tr, X_te)

    # Guardar para el modelo
    np.save("X_train_cnn.npy", X_tr_s)
    np.save("y_train_cnn.npy", y_tr)
    np.save("X_test_cnn.npy",  X_te_s)
    np.save("y_test_cnn.npy",  y_te)

    print("Guardado: X_train_cnn.npy, y_train_cnn.npy, X_test_cnn.npy, y_test_cnn.npy")