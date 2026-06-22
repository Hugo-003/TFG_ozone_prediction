from pathlib import Path
import pandas as pd


from sklearn.metrics import mean_absolute_error, r2_score
from src.Data_treatment.preprocessing import load_all_data, save_processed_data
from src.features.Build_features import build_all_features
from src.Models.model1 import (
    preparar_datos_modelo, entrenar_lgbm, entrenar_lgbm_cuantil,
    guardar_modelo, registrar_experimento
)
from src.Models.evaluate import evaluar_modelo
from src.Visualization.Plots import (
    plot_distribucion_o3,
    plot_estacionalidad_o3,
    plot_predicciones_vs_real,
    plot_zoom_extremos,
    plot_error_por_rango,
    plot_importancia_features,
    plot_importancia_features_comparativa,
    plot_errores_por_estacion,
    plot_extremos_por_estacion_total,
    plot_distribucion_temporal_extremos,
    plot_degradacion_por_horizonte
)

# Horizonte de predicción en horas (cuántas horas hacia el futuro predice el modelo)
HORIZON = 6


def run():
    print("=== PIPELINE DE PREDICCIÓN DE OZONO (CEAM) ===\n")
    print(f"[INFO] Horizonte de predicción: {HORIZON} horas\n")

    # 1. CARGA Y LIMPIEZA
    processed_path = Path("Data/processed/df_clean.parquet")
    if processed_path.exists():
        print("[INFO] Cargando datos preprocesados desde parquet...")
        df = pd.read_parquet(processed_path)
    else:
        print("[INFO] Procesando datos en bruto...")
        df = load_all_data("Data/Raw_Data")
        save_processed_data(df, processed_path)
    print(f"[INFO] Registros tras limpieza: {len(df):,}\n")

    # 2. INGENIERÍA DE CARACTERÍSTICAS + TARGET FUTURO
    df, target_col = build_all_features(df, horizon=HORIZON)
    print(f"[INFO] Registros tras feature engineering: {len(df):,}")
    print(f"[INFO] Variable objetivo: {target_col}\n")

    # 3. DIVISIÓN TEMPORAL
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos_modelo(
        df, target_col=target_col
    )

    # Guardamos station_id del test para el análisis por estación
    test_mask = df['year'] == 2025
    station_ids_test = df[test_mask]['station_id'].reset_index(drop=True)

    # 4. ENTRENAMIENTO
    params_std = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 30,
        'verbose': -1
    }
    params_q90 = {
        'objective': 'quantile',
        'alpha': 0.90,
        'learning_rate': 0.05,
        'num_leaves': 30,
        'verbose': -1
    }

    modelo_std = entrenar_lgbm(X_train, X_val, y_train, y_val, params=params_std)
    modelo_q90 = entrenar_lgbm_cuantil(X_train, X_val, y_train, y_val, cuantil=0.90)

    # 5. EVALUACIÓN
    print("\n--- Modelo estándar ---")
    metricas_std = evaluar_modelo(modelo_std, X_test, y_test, umbral_extremo=170)

    print("\n--- Modelo cuantil 0.90 ---")
    metricas_q90 = evaluar_modelo(modelo_q90, X_test, y_test, umbral_extremo=170)

    # 6. PREDICCIONES
    preds_std = modelo_std.predict(X_test)
    preds_q90 = modelo_q90.predict(X_test)


    # 7. COMPARATIVA DE HORIZONTES

    print("\n[INFO] Comparativa de rendimiento por horizonte temporal...")
    from sklearn.metrics import mean_absolute_error, r2_score
    import gc

    processed_csv = Path("Data/processed/df_clean.csv")
    resultados_horizontes = []

    for h in [6, 12, 24]:
        print(f"  Entrenando horizonte t+{h}h...")
        df_h = pd.read_csv(processed_csv, parse_dates=['datetime'])
        df_h, target_h = build_all_features(df_h, horizon=h)
        Xtr, Xv, Xte, ytr, yv, yte = preparar_datos_modelo(df_h, target_col=target_h)

        # Liberamos el dataframe completo antes de entrenar
        del df_h
        gc.collect()

        m = entrenar_lgbm(Xtr, Xv, ytr, yv, params=params_std)
        preds_h = m.predict(Xte)
        resultados_horizontes.append({
            'horizon': h,
            'MAE': mean_absolute_error(yte, preds_h),
            'R2': r2_score(yte, preds_h)
        })

        # Liberamos todo antes de la siguiente iteración
        del Xtr, Xv, Xte, ytr, yv, yte, preds_h, m
        gc.collect()

    df_horizontes = pd.DataFrame(resultados_horizontes)
    print("\nComparativa por horizonte:")
    print(df_horizontes.to_string(index=False))
    plot_degradacion_por_horizonte(resultados_horizontes)


    # 8. VISUALIZACIONES
    print("\n[INFO] Generando visualizaciones...")

    plot_degradacion_por_horizonte(resultados_horizontes)

    plot_distribucion_o3(df['O3'])
    plot_estacionalidad_o3(df)
    plot_distribucion_temporal_extremos(df, umbral=170)

    plot_predicciones_vs_real(y_test, preds_std, titulo=f'Modelo estándar (t+{HORIZON}h)')
    plot_predicciones_vs_real(y_test, preds_q90, titulo=f'Modelo cuantil 0.90 (t+{HORIZON}h)')

    plot_zoom_extremos(y_test, preds_std, preds_q90, umbral=170)

    plot_error_por_rango(y_test, preds_std, titulo=f'Modelo estándar (t+{HORIZON}h)')
    plot_error_por_rango(y_test, preds_q90, titulo=f'Modelo cuantil 0.90 (t+{HORIZON}h)')

    plot_errores_por_estacion(y_test, preds_std, station_ids_test, umbral=170)
    plot_extremos_por_estacion_total(df, umbral=170)

    plot_importancia_features(modelo_std, X_train.columns.tolist())
    plot_importancia_features_comparativa(
        modelo_std, modelo_q90, X_train.columns.tolist()
    )

    # 9. PERSISTENCIA Y LOGS
    guardar_modelo(modelo_std, f'modelo_estandar_o3_t{HORIZON}h.pkl')
    guardar_modelo(modelo_q90, f'modelo_cuantil_q90_o3_t{HORIZON}h.pkl')

    registrar_experimento(
        metricas=metricas_std,
        hyperparams={**params_std, 'horizon': HORIZON},
        features_usadas=X_train.columns.tolist()
    )
    registrar_experimento(
        metricas=metricas_q90,
        hyperparams={**params_q90, 'horizon': HORIZON},
        features_usadas=X_train.columns.tolist()
    )

    

    print("\n=== PIPELINE COMPLETADO ===")


if __name__ == "__main__":
    run()
