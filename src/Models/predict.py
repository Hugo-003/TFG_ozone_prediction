# predict.py
import joblib
import pandas as pd
from pathlib import Path
from src.features.Build_features import build_all_features


def cargar_modelo(ruta='models/modelo_final_o3.pkl'):
    """Cargar un modelo previamente entrenado."""
    modelo = joblib.load(ruta)
    print(f"[INFO] Modelo cargado desde {ruta}")
    return modelo


def predecir(modelo, df_nuevos):
    """
    Aplica el pipeline de features y genera predicciones
    sobre datos nuevos. Devuelve el dataframe con una columna
    'O3_pred' añadida.
    """
    df = build_all_features(df_nuevos.copy())

    cols_a_excluir = ['datetime', 'station_id', 'O3']
    features = [c for c in df.columns if c not in cols_a_excluir]

    df['O3_pred'] = modelo.predict(df[features])
    return df


def predecir_con_alerta(modelo, df_nuevos, umbral=180):
    """
    Igual que predecir(), pero añade además una columna binaria
    'alerta_o3' que marca los registros donde la predicción
    supera el umbral de riesgo.
    """
    df = predecir(modelo, df_nuevos)
    df['alerta_o3'] = df['O3_pred'] > umbral
    n_alertas = df['alerta_o3'].sum()
    print(f"[INFO] Predicciones generadas. Alertas O3>{umbral}: {n_alertas}")
    return df