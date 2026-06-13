import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluar_modelo(modelo, X_test, y_test, umbral_extremo=170):
    """
    Evaluación completa del modelo en dos niveles:
    - Métricas globales sobre todo el test set
    - Métricas específicas sobre los registros con O3 > umbral_extremo
    
    Esta separación es el resultado central del TFG: el modelo funciona
    bien en régimen normal pero falla sistemáticamente en los extremos.
    """
    preds = modelo.predict(X_test)

    # --- Métricas globales ---
    mae_global  = mean_absolute_error(y_test, preds)
    rmse_global = np.sqrt(mean_squared_error(y_test, preds))
    r2_global   = r2_score(y_test, preds)

    print("\n=== EVALUACIÓN GLOBAL ===")
    print(f"  MAE:  {mae_global:.2f}")
    print(f"  RMSE: {rmse_global:.2f}")
    print(f"  R²:   {r2_global:.4f}")

    # --- Métricas en rango extremo ---
    mask_extremo = y_test > umbral_extremo
    n_extremos = mask_extremo.sum()

    print(f"\n=== EVALUACIÓN EN RANGO EXTREMO (O3 > {umbral_extremo}) ===")
    print(f"  Registros con O3 > {umbral_extremo}: {n_extremos} "
          f"({100 * n_extremos / len(y_test):.4f}% del test set)")

    if n_extremos > 0:
        mae_ext  = mean_absolute_error(y_test[mask_extremo], preds[mask_extremo])
        rmse_ext = np.sqrt(mean_squared_error(y_test[mask_extremo], preds[mask_extremo]))
        r2_ext   = r2_score(y_test[mask_extremo], preds[mask_extremo]) if n_extremos > 1 else float('nan')
        sesgo_medio = np.mean(preds[mask_extremo] - y_test[mask_extremo])

        print(f"  MAE extremos:   {mae_ext:.2f}")
        print(f"  RMSE extremos:  {rmse_ext:.2f}")
        print(f"  R² extremos:    {r2_ext:.4f}")
        print(f"  Sesgo medio:    {sesgo_medio:.2f}  "
              f"({'subestima' if sesgo_medio < 0 else 'sobreestima'} el valor real)")
    else:
        print("  No hay registros extremos en el test set para evaluar.")
        mae_ext = rmse_ext = r2_ext = sesgo_medio = float('nan')

    metricas = {
        'MAE_global':    mae_global,
        'RMSE_global':   rmse_global,
        'R2_global':     r2_global,
        'MAE_extremos':  mae_ext,
        'RMSE_extremos': rmse_ext,
        'R2_extremos':   r2_ext,
        'Sesgo_extremos': sesgo_medio,
        'N_extremos':    int(n_extremos)
    }

    return metricas