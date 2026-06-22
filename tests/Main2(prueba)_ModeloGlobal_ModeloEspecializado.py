
# ===================================================================================
# PRUEBA: ENTRENAR MODELO SOLO CON LAS ESTACIONES QUE HAYAN TENIDO EPISODIOS EXTREMOS
# ===================================================================================

from pathlib import Path
import pandas as pd
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from src.Data_treatment.preprocessing import load_all_data, save_processed_data
from src.features.Build_features import build_all_features
from src.Models.model1 import (
    preparar_datos_modelo, entrenar_lgbm, entrenar_lgbm_cuantil,
    guardar_modelo, registrar_experimento
)
from src.Models.evaluate import evaluar_modelo
from src.Visualization.Plots import (
    plot_predicciones_vs_real,
    plot_zoom_extremos,
    plot_degradacion_por_horizonte
)

# Horizonte de predicción
HORIZON = 6

# Mismos hiperparámetros que el resto 
PARAMS_STD = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 30,
    'verbose': -1
}
PARAMS_Q90 = {
    'objective': 'quantile',
    'alpha': 0.90,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 30,
    'verbose': -1
}


def run():
    print("=== PIPELINE DE PREDICCIÓN DE OZONO (CEAM) ===\n")

    # 1. CARGA Y LIMPIEZA
    processed_path = Path("Data/processed/df_clean.parquet")
    if processed_path.exists():
        df = pd.read_parquet(processed_path)
    else:
        df = load_all_data("Data/Raw_Data")
        save_processed_data(df, processed_path)

    # Aseguramos tipos float32 para reducir uso de memoria antes del sort/feature engineering
    cols_float = df.select_dtypes(include=['float64']).columns
    df[cols_float] = df[cols_float].astype('float32')

    # --- INICIO DE FILTRO DE PRUEBA --- (cambiar a False para modelo global)
    FILTRAR_ESTACIONES_CRITICAS = True

    if FILTRAR_ESTACIONES_CRITICAS:
        umbral_critico = 180
        estaciones = df[df['O3'] > umbral_critico]['station_id'].unique()
        df = df[df['station_id'].isin(estaciones)].reset_index(drop=True)
        print(f"[!] MODO PRUEBA: Entrenando con {len(estaciones)} estaciones críticas")
    else:
        print("[!] MODO GLOBAL: Entrenando con toda la red")
    # --- FIN DE FILTRO DE PRUEBA ---

    print(f"[INFO] Registros tras filtrado: {len(df):,}")

    # Liberamos cualquier referencia intermedia antes del paso más pesado en memoria
    gc.collect()

    # 2. INGENIERÍA DE CARACTERÍSTICAS
    df, target_col = build_all_features(df, horizon=HORIZON)
    gc.collect()

    # 3. DIVISIÓN TEMPORAL
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos_modelo(
        df, target_col=target_col
    )

    # Ya no necesitamos el dataframe completo: liberamos memoria antes de entrenar
    del df
    gc.collect()

    # 4. ENTRENAMIENTO
    modelo_std = entrenar_lgbm(X_train, X_val, y_train, y_val, params=PARAMS_STD)
    modelo_q90 = entrenar_lgbm_cuantil(X_train, X_val, y_train, y_val, cuantil=0.90)

    # 5. EVALUACIÓN
    print("\n--- Modelo estándar (estaciones críticas) ---")
    metricas_std = evaluar_modelo(modelo_std, X_test, y_test, umbral_extremo=170)

    print("\n--- Modelo cuantil 0.90 (estaciones críticas) ---")
    metricas_q90 = evaluar_modelo(modelo_q90, X_test, y_test, umbral_extremo=170)

    # 6. PREDICCIONES
    preds_std = modelo_std.predict(X_test)
    preds_q90 = modelo_q90.predict(X_test)

    # 7. PERSISTENCIA
    guardar_modelo(modelo_std, f'modelo_estandar_critico_o3_t{HORIZON}h.pkl')
    guardar_modelo(modelo_q90, f'modelo_cuantil_critico_o3_t{HORIZON}h.pkl')

    registrar_experimento(
        metricas=metricas_std,
        hyperparams={**PARAMS_STD, 'horizon': HORIZON, 'modo': 'estaciones_criticas'},
        features_usadas=X_train.columns.tolist()
    )
    registrar_experimento(
        metricas=metricas_q90,
        hyperparams={**PARAMS_Q90, 'horizon': HORIZON, 'modo': 'estaciones_criticas'},
        features_usadas=X_train.columns.tolist()
    )

    print("\n=== PIPELINE COMPLETADO ===")


if __name__ == "__main__":
    run()