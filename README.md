# TFG — Predicción de Ozono Troposférico mediante Machine Learning
### Grado en Ciencia de Datos · Universidad Europea de Valencia · Curso 2025-2026

## Descripción

Pipeline modular de Machine Learning para la predicción de los niveles de ozono
troposférico (O₃) con horizonte temporal configurable, desarrollado a partir de
datos reales de la red de estaciones de monitorización de la Comunitat Valenciana
(2019–2025), en el marco de las prácticas en la Fundación CEAM.

El sistema predice la concentración de O₃ con N horas de antelación y activa
automáticamente alertas cuando las predicciones superan el umbral de 180 µg/m³
establecido por la Directiva europea 2008/50/CE. El trabajo demuestra, sin
embargo, que esta alerta no es fiable para los episodios más extremos debido a
su extrema escasez en los datos disponibles. Caracterizar esa limitación con rigor estadístico es, de hecho, una de las aportaciones centrales del proyecto.

## Limitación principal

Los episodios con O₃ superior a 170 µg/m³ representan menos del 0.002% del
dataset (apenas una decena de registros en más de 3,6 millones de observaciones).
Esta escasez extrema hace inviable su predicción fiable mediante supervisión
estándar: cualquier modelo entrenado sobre estos datos aprende que esos valores
son prácticamente imposibles, y subestima de forma sistemática los episodios
reales cuando ocurren.

Se exploraron dos estrategias para mitigar este problema —regresión por
cuantiles y especialización geográfica del entrenamiento en las estaciones con
histórico de extremos— y ninguna de las dos resolvió la subestimación de forma
relevante. El trabajo lo documenta con detalle y propone las
condiciones que serían necesarias para abordar el problema en el futuro.

## Resultados principales

| Configuración | MAE global (µg/m³) | R² global | MAE en eventos extremos (O₃ > 170 µg/m³) |
|---|---|---|---|
| Estimación actual (horizon=0) | 5.46 | 0.921 | — |
| Predicción a 6h | 13.20 | 0.635 | 78.46 |
| Predicción a 12h | 14.64 | 0.561 | — |
| Predicción a 24h | 14.06 | 0.579 | — |

El sistema funciona bien en el rango habitual de concentraciones, pero falla de
forma sistemática y predecible en los episodios extremos. El MAE en eventos
extremos solo se reporta para horizon=6h, configuración principal de evaluación
del trabajo; no se calculó de forma equivalente para el resto de horizontes.

## Estructura del proyecto

TFG_ozone_prediction/

├── Main.py                       # Punto de entrada único del pipeline

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

├── tests/

│   ├── Main2(prueba)_ModeloGlobal_ModeloEspecializado.py

│   └── test_inferencia.py

├── .gitignore

└── README.md

## Requisitos

Python 3.9 o superior. Instala las dependencias con:

```bash
pip install lightgbm pandas numpy scikit-learn matplotlib joblib pyarrow
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

### Scripts de verificación

La carpeta `tests/` contiene scripts auxiliares usados durante el desarrollo
para verificar partes concretas del sistema (no forman parte del pipeline
principal).

## Memoria completa

El detalle metodológico completo, el marco teórico, el análisis exploratorio y
la discusión de resultados se encuentran en la memoria del TFG, incluida en
este repositorio.

## Autor

Hugo Crespo Adroguer · [hugo-003](https://github.com/Hugo-003/TFG_ozone_prediction)