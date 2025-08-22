#LIMPIEZA — Dataset de tráfico telco (SMS, llamadas, internet)

import pandas as pd
import numpy as np
from typing import Tuple
def cargar_crudo(path_csv: str) -> pd.DataFrame:
    """
    Carga el CSV crudo 
    """
    df = pd.read_csv(path_csv, low_memory=False)
    return df
def parsear_fecha(df: pd.DataFrame, col_fecha: str = "datetime") -> pd.DataFrame:
    """
    la columna de fecha tiene formato día/mes/año.
    Si falla, marca NaT (errors='coerce').
    """
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors="coerce")
    return df

def forzar_numerico(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Convierte a numérico (float) las columnas indicadas; valores no convertibles -> NaN.
    """
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def limpieza_basica(
    df: pd.DataFrame,
    col_fecha: str = "datetime",
    col_etiqueta: str = "countrycode",
    cols_trafico: list = ["smsin", "smsout", "callin", "callout", "internet"],
    imputar_trafico_con: float = 0.0,
) -> Tuple[pd.DataFrame, dict]:
     

    df = df.copy()

    # 1) Quitar filas sin fecha válida
    antes = len(df)
    df = df.dropna(subset=[col_fecha])
    despues_fecha = len(df)

    # 2) Quitar filas sin etiqueta (necesario para supervisado)
    df = df.dropna(subset=[col_etiqueta])
    despues_label = len(df)

    # 3) Imputación en columnas de tráfico
    df[cols_trafico] = df[cols_trafico].fillna(imputar_trafico_con)

    # 4) Resumen de limpieza
    resumen = {
        "filas_antes": antes,
        "filas_despues_fecha": despues_fecha,
        "filas_despues_label": despues_label,
        "imputacion_trafico_valor": imputar_trafico_con,
        "n_celdas_unicas": df["CellID"].nunique(dropna=True) if "CellID" in df.columns else None,
        "n_paises": df[col_etiqueta].nunique(dropna=True),
        "faltantes_trafico_post_imputacion": {
            c: int(df[c].isna().sum()) for c in cols_trafico
        },
    }

    return df, resumen

def guardar(df: pd.DataFrame, path_out: str) -> None:
    df.to_csv(path_out, index=False)

if __name__ == "__main__":
    # === Configura tus rutas ===
        

    PATH_IN = "datos_sms_llamadas_trafico_.csv" # ruta al CSV crudo
    PATH_OUT = "limpieza_telco_supervised.csv" # salida limpia (solo limpieza)

    # === Pipeline de limpieza ===
    df_raw = cargar_crudo(PATH_IN)

    # Forzar tipos numéricos de columnas clave (si alguna no existe, ajústala)
    cols_numericas = ["CellID", "countrycode", "smsin", "smsout", "callin", "callout", "internet"]
    df_raw = forzar_numerico(df_raw, cols_numericas)

    # Parseo de fecha
    df_raw = parsear_fecha(df_raw, col_fecha="datetime")

    # Limpieza solicitada (sin features ni transformaciones avanzadas)
    df_clean, info = limpieza_basica(
        df_raw,
        col_fecha="datetime",
        col_etiqueta="countrycode",
        cols_trafico=["smsin", "smsout", "callin", "callout", "internet"],
        imputar_trafico_con=0.0,
    )

    # Guardar
    guardar(df_clean, PATH_OUT)

    # Reporte por consola
    print("== LIMPIEZA COMPLETADA ==")
    for k, v in info.items():
        print(f"{k}: {v}")
    print(f"\nArchivo limpio guardado en: {PATH_OUT}")