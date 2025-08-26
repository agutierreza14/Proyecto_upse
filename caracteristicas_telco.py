# caracteristicas_telco.py

import pandas as pd
import numpy as np
from typing import Tuple, List

TRAFFIC_COLS = ["smsin", "smsout", "callin", "callout", "internet"]
REQUIRED_COLS = ["datetime", "countrycode", "CellID"] + TRAFFIC_COLS

def cargar_transformado(path_csv: str) -> pd.DataFrame:
    """
    Carga el CSV de transformacion
    """
    df = pd.read_csv(path_csv, low_memory=False)
    return df

def validar_columnas(df: pd.DataFrame, required: List[str] = REQUIRED_COLS) -> None:
    """
    Verifica que existan las columnas requeridas.
    """
    faltantes = [c for c in required if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

def agregar_totales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Totales de cada modalidad y total combinado.
    """
    df = df.copy()
    df["total_sms"] = df["smsin"] + df["smsout"]
    df["total_call"] = df["callin"] + df["callout"]
    df["total_traffic"] = df["total_sms"] + df["total_call"] + df["internet"]
    return df

def agregar_ratios(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """
    Ratios robustos (evita división por 0 con eps).
    """
    df = df.copy()
    df["ratio_sms_out_in"] = df["smsout"] / (df["smsin"] + eps)
    df["ratio_call_out_in"] = df["callout"] / (df["callin"] + eps)
    df["ratio_sms_call"] = df["total_sms"] / (df["total_call"] + eps)
    df["internet_per_sms"] = df["internet"] / (df["total_sms"] + eps)
    df["internet_per_call"] = df["internet"] / (df["total_call"] + eps)
    # Opcional: limitar ratios extremos para robustez (winsorizar)
    for c in ["ratio_sms_out_in","ratio_call_out_in","ratio_sms_call","internet_per_sms","internet_per_call"]:
        df[c] = df[c].clip(upper=1e6)
    return df

def agregar_participaciones(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """
    Participación (share) de cada modalidad sobre el total.
    """
    df = df.copy()
    denom = df["total_traffic"] + eps
    df["share_sms"] = df["total_sms"] / denom
    df["share_call"] = df["total_call"] / denom
    df["share_data"] = df["internet"] / denom
    return df

def agregar_flags_interacciones(df: pd.DataFrame) -> pd.DataFrame:
   

    df = df.copy()
    # Flags de actividad
    df["any_sms"] = (df["total_sms"] > 0).astype(int)
    df["any_call"] = (df["total_call"] > 0).astype(int)
    df["any_data"] = (df["internet"] > 0).astype(int)

    # Interacciones con tiempo (si existen columnas temporales)
    if "is_weekend" in df.columns:
        df["weekend_sms"] = df["total_sms"] * df["is_weekend"]
        df["weekend_call"] = df["total_call"] * df["is_weekend"]
        df["weekend_data"] = df["internet"] * df["is_weekend"]
    if "is_night" in df.columns:
        df["night_sms"] = df["total_sms"] * df["is_night"]
        df["night_call"] = df["total_call"] * df["is_night"]
        df["night_data"] = df["internet"] * df["is_night"]
    return df

def crear_objetivos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables objetivo para aprendizaje supervisado:
      - y_country (multiclase): countrycode entero
      - y_international (binario): 1 si countrycode != 0; 0 si countrycode == 0
    """
    df = df.copy()
    df["y_country"] = pd.to_numeric(df["countrycode"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["y_country"]).copy()  # nos quedamos con filas etiquetadas
    df["y_international"] = (df["y_country"] != 0).astype(int)
    return df

def extraer_caracteristicas(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    - Totales
    - Ratios
    - Participaciones
    - Flags de actividad
    - Interacciones con tiempo (si existen)
    - Objetivos (y_country, y_international)
    Devuelve df_features y un resumen.
    """
    validar_columnas(df, REQUIRED_COLS)
    df = agregar_totales(df)
    df = agregar_ratios(df)
    df = agregar_participaciones(df)
    df = agregar_flags_interacciones(df)
    df = crear_objetivos(df)

    resumen = {
        "filas_con_objetivo": len(df),
        "n_celdas_unicas": int(df["CellID"].nunique(dropna=True)),
        "n_paises": int(df["y_country"].nunique(dropna=True)),
        "balance_binario": df["y_international"].value_counts().to_dict(),
        "ejemplo_filas": df.head(3).to_dict(orient="records"),
        "features_creadas": [
            "total_sms","total_call","total_traffic",
            "ratio_sms_out_in","ratio_call_out_in","ratio_sms_call",
            "internet_per_sms","internet_per_call",
            "share_sms","share_call","share_data",
            "any_sms","any_call","any_data",
            "weekend_*","night_*"
        ]
    }
    return df, resumen

def guardar(df: pd.DataFrame, path_out: str) -> None:
    df.to_csv(path_out, index=False)

if __name__ == "__main__":
    # === Configuracion de rutas ===
    
    PATH_IN = "transformacion_telco.csv"          
    PATH_OUT = "caracteristica_telco.csv"         

    
    df_in = cargar_transformado(PATH_IN)
    df_feat, info = extraer_caracteristicas(df_in)
    guardar(df_feat, PATH_OUT)

    print("EXTRACCIÓN DE CARACTERÍSTICAS")
    for k, v in info.items():
        print(f"{k}: {v}")
    print(f"\nArchivo con features guardado en: {PATH_OUT}")
