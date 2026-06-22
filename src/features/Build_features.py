
import pandas as pd
import numpy as np


def crear_lags(df, columnas_objetivo=["O3", "NO2"], lags=[1, 24, 48]):
    """
    Crea variables de retraso (lags) agrupando por estación para evitar
    mezclar datos de diferentes puntos geográficos.

    el dropna final elimina las primeras N horas de cada estación,
    donde N es el lag máximo (por defecto 48h).
    """
    df = df.sort_values(["station_id", "datetime"])

    for col in columnas_objetivo:
        if col in df.columns:
            for lag in lags:
                nombre_col = f"{col}_lag_{lag}"
                df[nombre_col] = df.groupby("station_id")[col].shift(lag)

    columnas_creadas = [
        f"{col}_lag_{lag}"
        for col in columnas_objetivo
        for lag in lags
        if col in df.columns
    ]
    df = df.dropna(subset=columnas_creadas)

    return df


def crear_features_ciclicas(df):
    """
    Transforma horas y meses en senos y cosenos para que el modelo
    entienda la continuidad temporal (ej: 23h y 0h están cerca).
    """
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    return df


def crear_target_futuro(df, horizon=24):
    """
    Crea la variable objetivo desplazada N horas hacia el futuro.
    Predecir O3 en t+horizon usando features de t.
    """
    target_col = f'O3_t{horizon}h'
    df[target_col] = df.groupby('station_id')['O3'].shift(-horizon)
    # Eliminamos los registros donde el target es NaN
    # (las últimas N horas de cada estación no tienen target)
    df = df.dropna(subset=[target_col])
    return df, target_col


def build_all_features(df, horizon=24, construir_target=True):
    """
    Pipeline de feature engineering.

    Parámetros
    ----------
    horizon : int
        Horizonte de predicción en horas.
    construir_target : bool, default True
        Si True (comportamiento de siempre, usado en entrenamiento y
        evaluación), construye la columna target desplazada y elimina
        las filas donde no se puede calcular.
        Si False (modo inferencia sobre datos nuevos), se omite ese
        paso por completo: no se exige conocer el futuro, y no se
        descarta ninguna fila por esa razón. Esto es lo que permite
        usar la función sobre datos "del presente" en predict.py.

    Devuelve
    --------
    df : DataFrame con las features añadidas.
    target_col : str si construir_target=True, o None si construir_target=False.
    """
    df = crear_lags(df)
    df = crear_features_ciclicas(df)

    if construir_target:
        df, target_col = crear_target_futuro(df, horizon=horizon)
        return df, target_col

    return df, None