
# hacer pip install pyarrow
"""
PRUEBA DEL MÓDULO DE INFERENCIA (tras la corrección de predict.py)
====================================================================
Ahora que build_all_features() admite construir_target=False, ya no
es necesario simular un margen futuro artificial: podemos usar
directamente las filas más recientes de cada estación, igual que
ocurriría con datos reales de "ahora mismo".

Esta prueba simplemente comprueba que el pipeline completo
(carga de modelo -> features -> predicción -> alerta) funciona sin
errores y produce resultados con sentido.
"""

import pandas as pd
from src.Models.predict import cargar_modelo, predecir_con_alerta

HORIZON = 6
MARGEN_LAG = 48  # el lag más largo usado en crear_lags; necesitamos
                  # al menos esta historia previa por estación para
                  # que los lags no salgan vacíos

# 1. CARGAR EL MODELO YA ENTRENADO
modelo = cargar_modelo(f"models/modelo_estandar_o3_t{HORIZON}h.pkl")

# 2. CARGAR LOS DATOS LIMPIOS (sin features)
df_clean = pd.read_csv("Data/processed/df_clean.csv", parse_dates=["datetime"])
df_clean = df_clean.sort_values(["station_id", "datetime"])

# 3. SIMULAR "DATOS DE AHORA": las últimas filas disponibles por estación,
# manteniendo algo de historia previa (MARGEN_LAG) para que los lags
# puedan calcularse con normalidad.
muestra = (
    df_clean.groupby("station_id", group_keys=False)
    .apply(lambda g: g.tail(MARGEN_LAG + 10))  # 10 filas "actuales" + historia
    .reset_index(drop=True)
)
print(f"[INFO] Muestra de prueba: {len(muestra)} registros de "
      f"{muestra['station_id'].nunique()} estaciones.\n")

# 4. PREDICCIÓN + ALERTA
resultado = predecir_con_alerta(modelo, muestra, umbral=180)

print("\n[INFO] Últimas predicciones generadas (las más 'actuales' de cada estación):")
print(
    resultado
    .sort_values(["station_id", "datetime"])
    .groupby("station_id")
    .tail(3)
    [["station_id", "datetime", "O3", "O3_pred", "alerta_o3"]]
)

print(f"\n[INFO] Total de filas con predicción: {len(resultado)}")
print(f"[INFO] Alertas activadas: {resultado['alerta_o3'].sum()}")
print(
    "\n[INFO] No tenemos el valor real de t+6h para estas filas "
    "(es precisamente lo que se está prediciendo), así que aquí no "
    "se calcula ningún MAE. Si se quiere verificar la precisión del "
    "modelo de forma rigurosa, esa comprobación ya está hecha sobre "
    "el conjunto de test de 2025 en el apartado 5.3 del TFG."
)