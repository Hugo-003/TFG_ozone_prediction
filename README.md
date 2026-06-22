# TFG — Predicción de Ozono Troposférico mediante Machine Learning
### Grado en Ciencia de Datos · Universidad Europea de Valencia · Curso 2025-2026

## Descripción

Pipeline modular de Machine Learning para la predicción de los niveles de ozono 
troposférico (O₃) con horizonte temporal configurable, desarrollado a partir de 
datos reales de la red de estaciones de monitorización de la Comunitat Valenciana 
(2019–2025), en el marco de las prácticas en la Fundación CEAM.

El sistema predice la concentración de O₃ con N horas de antelación y activa 
automáticamente alertas cuando las predicciones superan el umbral de 180 µg/m³ 
establecido por la Directiva europea 2008/50/CE.

## Estructura del proyecto

TFG_ozone_prediction/

├── src/

│   ├── Data_treatment/

│   │   └── preprocessing.py      # Lectura y limpieza de archivos .MH

│   ├── features/

│   │   └── Build_features.py     # Lags, encoding cíclico y target futuro

│   ├── Models/

│   │   ├── model1.py             # Entrenamiento LightGBM estándar y cuantil

│   │   ├── evaluate.py           # Evaluación global y en rango extremo

│   │   └── predict.py            # Inferencia y detección de alertas

│   └── Visualization/

│       └── Plots.py              # Generación automática de gráficas

├── Main.py                       # Punto de entrada único del pipeline

├── Main2(prueba)_ModeloGlobal_ModeloEspecializado.py    # Prueba entre modelos

├── .gitignore

└── README.md

## Requisitos

Python 3.9 o superior. Instala las dependencias con:

```bash
pip install lightgbm pandas numpy scikit-learn matplotlib joblib
```

## Uso

1. Coloca los archivos `.MH` de las estaciones en `Data/Raw_Data/`
2. Configura el horizonte de predicción en `Main.py`:
```python
   HORIZON = 6  # horas hacia el futuro (6, 12 o 24)
```
3. Ejecuta el pipeline completo:
```bash
   python Main.py
```

El sistema procesará los datos, entrenará los modelos, generará todas las 
gráficas en `outputs/figures/` y registrará el experimento en 
`experimentos_log.txt`.

## Resultados principales

| Configuración | MAE (µg/m³) | R² |
|--------------|-------------|-----|
| Estimación actual (horizon=0) | 5.46 | 0.921 |
| Predicción a 6h | 13.20 | 0.635 |
| Predicción a 12h | 14.64 | 0.561 |
| Predicción a 24h | 14.06 | 0.579 |

## Limitación principal

Los episodios con O₃ superior a 180 µg/m³ representan menos del 0.001% del 
dataset. Esta escasez extrema hace inviable su predicción fiable con supervisión 
estándar. El trabajo caracteriza esta limitación de forma rigurosa y propone 
las condiciones necesarias para abordarla en el futuro.

## Autor

Hugo Crespo Adroguer · [hugo-003](https://github.com/Hugo-003)