# transformacion_telco.py

import pandas as pd
import numpy as np
from typing import Tuple, List

TRAFFIC_COLS = ["smsin", "smsout", "callin", "callout", "internet"]

def cargar_limpio(path_csv: str) -> pd.DataFrame:
    """
    Carga el CSV limpio
    """
    df = pd.read_csv(path_csv, low_memory=False)
    return df

def asegurar_datetime(df: pd.DataFrame, col_fecha: str = "datetime") -> pd.DataFrame:
    """
    Asegurar que la columna datetime sea tipo datetime"""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[col_fecha]):
        df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors="coerce")

   
    df = df.dropna(subset=[col_fecha])
    return df

def derivar_variables_tiempo(df: pd.DataFrame, col_fecha: str = "datetime") -> pd.DataFrame:
    """
    
    """
    df = df.copy()
    dt = df[col_fecha].dt
    df["hour"] = dt.hour
    df["dayofweek"] = dt.dayofweek           # 0 = Lunes
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["month"] = dt.month
    df["is_night"] = df["hour"].isin([0,1,2,3,4,5,22,23]).astype(int)
    return df

def transformar_log1p(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Aplica log1p (log(1+x)) a columnas numéricas de tráfico para reducir asimetría.
    
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            # Asegurar que no haya valores negativos 
            if (df[c] < 0).any():
                raise ValueError(f"La columna {c} contiene valores negativos, no es válida para log1p.")
            df[f"log1p_{c}"] = np.log1p(df[c].astype(float))
    return df

def transformar(
    df: pd.DataFrame,
    col_fecha: str = "datetime",
    cols_trafico: List[str] = TRAFFIC_COLS,
) -> Tuple[pd.DataFrame, dict]:
    """

    - Deriva variables de tiempo
    - Aplica log1p a tráfico
    Retorna df_transformado 
    """
    df = asegurar_datetime(df, col_fecha=col_fecha)
    df = derivar_variables_tiempo(df, col_fecha=col_fecha)
    df = transformar_log1p(df, cols_trafico)

    resumen = {
        "filas_transformadas": len(df),
        "cols_agregadas_tiempo": ["hour", "dayofweek", "is_weekend", "month", "is_night"],
        "cols_agregadas_log1p": [f"log1p_{c}" for c in cols_trafico if c in df.columns],
        "ejemplo_filas": df.head(3).to_dict(orient="records"),
    }
    return df, resumen

def guardar(df: pd.DataFrame, path_out: str) -> None:
    df.to_csv(path_out, index=False)

if __name__ == "__main__":
    # === Configura tus rutas ===
    
    PATH_IN = "limpieza_telco_supervised.csv"           
    PATH_OUT = "transformacion_telco.csv"               

   
    df_clean = cargar_limpio(PATH_IN)
    df_transf, info = transformar(
        df_clean,
        col_fecha="datetime",
        cols_trafico=TRAFFIC_COLS,
    )
    guardar(df_transf, PATH_OUT)

    print("== TRANSFORMACIÓN COMPLETA")
    for k, v in info.items():
        print(f"{k}: {v}")
    print(f"\nArchivo transformado guardado en: {PATH_OUT}")
