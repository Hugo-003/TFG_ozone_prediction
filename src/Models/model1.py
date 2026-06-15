import lightgbm as lgb
import joblib
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error


def preparar_datos_modelo(df, target_col='O3_t24h'):
    cols_a_excluir = ['datetime', 'station_id', 'O3', target_col]
    features = [c for c in df.columns if c not in cols_a_excluir]

    train_df = df[df['year'] <= 2023]
    val_df   = df[df['year'] == 2024]
    test_df  = df[df['year'] == 2025]

    X_train, y_train = train_df[features], train_df[target_col]
    X_val,   y_val   = val_df[features],   val_df[target_col]
    X_test,  y_test  = test_df[features],  test_df[target_col]

    print(f"[INFO] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def entrenar_lgbm(X_train, X_val, y_train, y_val, params=None):
   
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 30,
            'verbose': -1
        }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    print("Iniciando entrenamiento...")
    modelo = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    return modelo

def entrenar_lgbm_cuantil(X_train, X_val, y_train, y_val, cuantil=0.90):
    """
    Entrena un modelo LightGBM con regresión por cuantiles.
    Diseñado para intentar capturar mejor los valores extremos de O3.
    """
    params = {
        'objective': 'quantile',
        'alpha': cuantil,
        'metric': 'quantile',
        'learning_rate': 0.05,
        'num_leaves': 30,
        'verbose': -1
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    print(f"Entrenando modelo cuantil (alpha={cuantil})...")
    modelo = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    return modelo


def guardar_modelo(modelo, nombre_archivo='modelo_o3_lgbm.pkl'):
    """Guarda el modelo entrenado en la carpeta 'models'."""
    os.makedirs('models', exist_ok=True)
    ruta = os.path.join('models', nombre_archivo)
    joblib.dump(modelo, ruta)
    print(f"[INFO] Modelo guardado en: {ruta}")


def registrar_experimento(metricas, hyperparams, features_usadas):
    """Guarda el resultado del entrenamiento en un log histórico."""
    log_file = 'experimentos_log.txt'
    fecha = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Fecha: {fecha}\n")
        f.write(f"Métricas globales:\n")
        for k, v in metricas.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"Features ({len(features_usadas)}): {', '.join(features_usadas)}\n")
        f.write(f"Hyperparams: {hyperparams}\n")
        f.write(f"{'='*50}\n")

    print(f"[INFO] Experimento registrado en {log_file}")