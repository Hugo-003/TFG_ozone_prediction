import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_distribucion_o3(y, umbral=180, guardar=True):
    """
    Histograma de la distribución completa de O3.
    Muestra visualmente la escasez extrema de valores altos.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(y, bins=100, color='steelblue', edgecolor='none', alpha=0.85)
    ax.axvline(umbral, color='red', linestyle='--', linewidth=1.5,
               label=f'Umbral de alerta ({umbral} µg/m³)')
    ax.axvline(170, color='orange', linestyle='--', linewidth=1.2,
               label='Umbral de análisis (170 µg/m³)')

    n_extremos = (y > umbral).sum()
    pct = 100 * n_extremos / len(y)
    ax.text(umbral + 2, ax.get_ylim()[1] * 0.85,
            f'{n_extremos} registros\n({pct:.4f}%)',
            color='red', fontsize=9)

    ax.set_xlabel('Concentración de O₃ (µg/m³)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de los niveles de O₃ — Red de estaciones CV (2019–2025)')
    ax.legend()
    plt.tight_layout()

    if guardar:
        fig.savefig(OUTPUT_DIR / 'distribucion_o3.png', dpi=150)
        print("[INFO] Guardado: distribucion_o3.png")
    plt.show()
    plt.close()


def plot_predicciones_vs_real(y_test, preds, titulo='Modelo estándar', guardar=True):
    """
    Scatter plot de predicciones vs valores reales.
    La diagonal perfecta sirve de referencia visual.
    Muestra claramente cómo el modelo subestima los valores extremos.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_test, preds, alpha=0.15, s=5, color='steelblue', label='Predicciones')

    lim_max = max(y_test.max(), preds.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], 'r--', linewidth=1.5, label='Predicción perfecta')
    ax.axvline(170, color='orange', linestyle=':', linewidth=1.2, label='Umbral 170 µg/m³')

    ax.set_xlabel('O₃ real (µg/m³)')
    ax.set_ylabel('O₃ predicho (µg/m³)')
    ax.set_title(f'Predicciones vs Valores reales — {titulo}')
    ax.legend()
    plt.tight_layout()

    nombre = f"scatter_pred_real_{titulo.lower().replace(' ', '_')}.png"
    if guardar:
        fig.savefig(OUTPUT_DIR / nombre, dpi=150)
        print(f"[INFO] Guardado: {nombre}")
    plt.show()
    plt.close()


def plot_zoom_extremos(y_test, preds_std, preds_q90, umbral=170, guardar=True):
    """
    Zoom en los registros con O3 > umbral.
    Compara modelo estándar vs cuantil 0.90 sobre los mismos puntos extremos.
    Es la gráfica más importante del TFG: muestra el fallo de ambos modelos
    en los casos de riesgo real.
    """
    mask = y_test > umbral
    idx = np.where(mask)[0]

    if len(idx) == 0:
        print("[INFO] No hay registros extremos para graficar.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(idx)), y_test.values[idx],
            'o-', color='black', label='Valor real', linewidth=1.5, markersize=6)
    ax.plot(range(len(idx)), preds_std[idx],
            's--', color='steelblue', label='Modelo estándar', linewidth=1.2, markersize=5)
    ax.plot(range(len(idx)), preds_q90[idx],
            '^--', color='darkorange', label='Modelo cuantil 0.90', linewidth=1.2, markersize=5)

    ax.axhline(180, color='red', linestyle='--', linewidth=1, label='Umbral alerta 180 µg/m³')
    ax.set_xlabel('Índice del registro extremo')
    ax.set_ylabel('Concentración de O₃ (µg/m³)')
    ax.set_title(f'Comportamiento de los modelos en registros con O₃ > {umbral} µg/m³')
    ax.legend()
    plt.tight_layout()

    if guardar:
        fig.savefig(OUTPUT_DIR / 'zoom_extremos_comparativa.png', dpi=150)
        print("[INFO] Guardado: zoom_extremos_comparativa.png")
    plt.show()
    plt.close()


def plot_error_por_rango(y_test, preds, titulo='Modelo estándar', guardar=True):
    """
    Error absoluto medio agrupado por rangos de O3.
    Hace visible de forma muy clara cómo el error crece
    a medida que el O3 real es más alto.
    """
    df = pd.DataFrame({'real': y_test.values, 'pred': preds})
    df['error_abs'] = np.abs(df['real'] - df['pred'])

    bins = [0, 40, 80, 120, 160, 170, 180, 300]
    labels = ['0–40', '40–80', '80–120', '120–160', '160–170', '170–180', '>180']
    df['rango'] = pd.cut(df['real'], bins=bins, labels=labels)

    resumen = df.groupby('rango', observed=True)['error_abs'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    colores = ['steelblue'] * 5 + ['orange', 'red']
    bars = ax.bar(resumen['rango'], resumen['error_abs'], color=colores, edgecolor='none')

    ax.set_xlabel('Rango de O₃ real (µg/m³)')
    ax.set_ylabel('MAE medio (µg/m³)')
    ax.set_title(f'Error medio por rango de concentración — {titulo}')

    for bar, val in zip(bars, resumen['error_abs']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    nombre = f"error_por_rango_{titulo.lower().replace(' ', '_')}.png"
    if guardar:
        fig.savefig(OUTPUT_DIR / nombre, dpi=150)
        print(f"[INFO] Guardado: {nombre}")
    plt.show()
    plt.close()


def plot_estacionalidad_o3(df, guardar=True):
    """
    Media de O3 por hora del día y por mes del año.
    Muestra los patrones estacionales que el modelo aprende
    y contextualiza cuándo ocurren los picos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    media_hora = df.groupby('hour')['O3'].mean()
    axes[0].plot(media_hora.index, media_hora.values, color='steelblue', linewidth=2)
    axes[0].set_xlabel('Hora del día')
    axes[0].set_ylabel('O₃ medio (µg/m³)')
    axes[0].set_title('Perfil horario medio de O₃')
    axes[0].set_xticks(range(0, 24, 2))

    media_mes = df.groupby('month')['O3'].mean()
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    axes[1].bar(media_mes.index, media_mes.values, color='steelblue',
                edgecolor='none', tick_label=meses)
    axes[1].set_xlabel('Mes')
    axes[1].set_ylabel('O₃ medio (µg/m³)')
    axes[1].set_title('Perfil mensual medio de O₃')

    plt.tight_layout()

    if guardar:
        fig.savefig(OUTPUT_DIR / 'estacionalidad_o3.png', dpi=150)
        print("[INFO] Guardado: estacionalidad_o3.png")
    plt.show()
    plt.close()


def plot_importancia_features(modelo, feature_names, top_n=20, guardar=True):
    """
    Importancia de las variables según LightGBM.
    Útil para la sección de metodología del TFG.
    """
    importancias = modelo.feature_importance(importance_type='gain')
    indices = np.argsort(importancias)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)),
            importancias[indices],
            color='steelblue', edgecolor='none')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Importancia (gain)')
    ax.set_title(f'Top {top_n} variables más importantes — LightGBM')
    plt.tight_layout()


    if guardar:
        fig.savefig(OUTPUT_DIR / 'importancia_features.png', dpi=150)
        print("[INFO] Guardado: importancia_features.png")
    plt.show()
    plt.close()


def plot_importancia_features_comparativa(modelo_std, modelo_q90, feature_names, top_n=20, guardar=True):
    """
    Compara la importancia de variables entre el modelo estándar
    y el modelo cuantil. Muestra si ambos aprenden las mismas señales.
    """
    imp_std = modelo_std.feature_importance(importance_type='gain')
    imp_q90 = modelo_q90.feature_importance(importance_type='gain')

    # Tomamos el top_n por importancia media entre los dos modelos
    imp_media = (imp_std + imp_q90) / 2
    indices = np.argsort(imp_media)[-top_n:]
    nombres = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(indices))
    alto = 0.35

    ax.barh(y + alto/2, imp_std[indices], alto,
            color='steelblue', label='Modelo estándar', alpha=0.85)
    ax.barh(y - alto/2, imp_q90[indices], alto,
            color='darkorange', label='Modelo cuantil 0.90', alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(nombres, fontsize=9)
    ax.set_xlabel('Importancia (gain)')
    ax.set_title(f'Top {top_n} variables — Modelo estándar vs Cuantil 0.90')
    ax.legend()
    plt.tight_layout()

    if guardar:
        fig.savefig(OUTPUT_DIR / 'importancia_features_comparativa.png', dpi=150)
        print("[INFO] Guardado: importancia_features_comparativa.png")
    plt.show()
    plt.close()



def plot_errores_por_estacion(y_test, preds, station_ids, umbral=170, guardar=True):
    """
    MAE medio por estación sobre el conjunto de test.
    Las estaciones con registros extremos en test se marcan en rojo.
    """
    y_test_arr = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else pd.Series(y_test).reset_index(drop=True)
    preds_arr = pd.Series(preds).reset_index(drop=True)
    station_arr = station_ids.reset_index(drop=True) if hasattr(station_ids, 'reset_index') else pd.Series(station_ids).reset_index(drop=True)

    df = pd.DataFrame({
        'real':       y_test_arr,
        'pred':       preds_arr,
        'station_id': station_arr
    })
    df['error_abs'] = np.abs(df['real'] - df['pred'])
    df['es_extremo'] = df['real'] > umbral

    resumen = df.groupby('station_id').agg(
        mae_medio=('error_abs', 'mean'),
        n_extremos=('es_extremo', 'sum')
    ).reset_index().sort_values('mae_medio', ascending=True)

    n_estaciones = len(resumen)
    altura = max(8, min(20, n_estaciones * 0.35))

    fig, ax = plt.subplots(figsize=(10, altura))
    colores = ['red' if n > 0 else 'steelblue' for n in resumen['n_extremos']]
    ax.barh(resumen['station_id'], resumen['mae_medio'],
            color=colores, edgecolor='none', alpha=0.85)
    ax.set_xlabel('MAE medio (µg/m³)', fontsize=10)
    ax.set_title('Error medio por estación — Conjunto de test 2025\n'
                 '(rojo = tiene al menos un registro extremo en test)', fontsize=10)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=8)
    plt.tight_layout()

    if guardar:
        fig.savefig(OUTPUT_DIR / 'mae_por_estacion.png', dpi=150, bbox_inches='tight')
        print("[INFO] Guardado: mae_por_estacion.png")
    plt.show()
    plt.close()


def plot_extremos_por_estacion_total(df, umbral=170, guardar=True):
    """
    Número de registros extremos por estación sobre el dataset completo (2019-2025).
    Se usa el dataset completo y no solo el test para tener una imagen
    representativa de qué estaciones concentran los episodios extremos.
    """
    df_ext = df[df['O3'] > umbral].copy()

    if df_ext.empty:
        print(f"[INFO] No hay registros con O3 > {umbral} en el dataset.")
        return

    conteo = df_ext.groupby('station_id').size().reset_index(name='n_extremos')
    conteo = conteo.sort_values('n_extremos', ascending=True)

    total = int(conteo['n_extremos'].sum())
    n_est = len(conteo)

    print(f"[INFO] Estaciones con O3 > {umbral} en dataset completo:")
    print(conteo.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, max(4, n_est * 0.5)))
    ax.barh(conteo['station_id'], conteo['n_extremos'],
            color='red', edgecolor='none', alpha=0.75)
    ax.set_xlabel(f'Nº registros con O₃ > {umbral} µg/m³', fontsize=10)
    ax.set_title(f'Estaciones con registros extremos — Dataset completo (2019–2025)', fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.text(0.98, 0.02, f'Total: {total} registros\nen {n_est} estaciones',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()

    if guardar:
        fig.savefig(OUTPUT_DIR / 'extremos_por_estacion_total.png',
                    dpi=150, bbox_inches='tight')
        print("[INFO] Guardado: extremos_por_estacion_total.png")
    plt.show()
    plt.close()
    

'''
def plot_errores_por_estacion(y_test, preds, station_ids, umbral=170, guardar=True):
    """
    MAE medio por estación, ordenado de mayor a menor error.
    Identifica qué estaciones son más difíciles de predecir
    y dónde se concentran los registros extremos.
    """
    # Reset de índices para garantizar alineación correcta
    y_test_arr = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else pd.Series(y_test).reset_index(drop=True)
    preds_arr = pd.Series(preds).reset_index(drop=True)
    station_arr = station_ids.reset_index(drop=True) if hasattr(station_ids, 'reset_index') else pd.Series(station_ids).reset_index(drop=True)

    df = pd.DataFrame({
        'real':       y_test_arr,
        'pred':       preds_arr,
        'station_id': station_arr
    })
    df['error_abs'] = np.abs(df['real'] - df['pred'])
    df['es_extremo'] = df['real'] > umbral

    resumen = df.groupby('station_id').agg(
        mae_medio=('error_abs', 'mean'),
        n_extremos=('es_extremo', 'sum'),
        n_registros=('real', 'count')
    ).reset_index().sort_values('mae_medio', ascending=True)

    n_estaciones = len(resumen)
    # Altura dinámica: mínimo 8, máximo 20, proporcional al número de estaciones
    altura = max(8, min(20, n_estaciones * 0.35))

    fig, axes = plt.subplots(1, 2, figsize=(16, altura))

    # Gráfica 1: MAE por estación
    colores = ['red' if n > 0 else 'steelblue' for n in resumen['n_extremos']]
    axes[0].barh(resumen['station_id'], resumen['mae_medio'],
                 color=colores, edgecolor='none', alpha=0.85)
    axes[0].set_xlabel('MAE medio (µg/m³)', fontsize=10)
    axes[0].set_title('Error medio por estación\n(rojo = tiene registros extremos)', fontsize=10)
    axes[0].tick_params(axis='y', labelsize=7)
    axes[0].tick_params(axis='x', labelsize=8)

    # Gráfica 2: Número de extremos por estación
    resumen_ext = resumen[resumen['n_extremos'] > 0].sort_values('n_extremos', ascending=True)

    # Debug: imprime para verificar que el recuento es correcto
    total_extremos = int(resumen_ext['n_extremos'].sum())
    print(f"[INFO] Total registros extremos (O3 > {umbral}) en test: {total_extremos}")
    print(resumen_ext[['station_id', 'n_extremos']].to_string(index=False))

    if not resumen_ext.empty:
        n_ext_estaciones = len(resumen_ext)
        axes[1].barh(resumen_ext['station_id'], resumen_ext['n_extremos'],
                     color='red', edgecolor='none', alpha=0.75)
        axes[1].set_xlabel(f'Nº registros con O₃ > {umbral} µg/m³', fontsize=10)
        axes[1].set_title(f'Estaciones con registros extremos\n(O₃ > {umbral} µg/m³)', fontsize=10)
        axes[1].tick_params(axis='y', labelsize=8)
        axes[1].tick_params(axis='x', labelsize=8)
        # Anotación del total
        axes[1].text(0.98, 0.02, f'Total: {total_extremos} registros\nen {n_ext_estaciones} estaciones',
                     transform=axes[1].transAxes, ha='right', va='bottom',
                     fontsize=8, color='darkred',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    else:
        axes[1].text(0.5, 0.5, 'Sin registros extremos en test',
                     ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()

    if guardar:
        fig.savefig(OUTPUT_DIR / 'errores_por_estacion.png', dpi=150, bbox_inches='tight')
        print("[INFO] Guardado: errores_por_estacion.png")
    plt.show()
    plt.close()

'''
'''def plot_errores_por_estacion(y_test, preds, station_ids, umbral=170, guardar=True):
    """
    MAE medio por estación, ordenado de mayor a menor error.
    Identifica qué estaciones son más difíciles de predecir
    y dónde se concentran los registros extremos.
    """
    df = pd.DataFrame({
        'real':       y_test.values,
        'pred':       preds,
        'station_id': station_ids.values
    })
    df['error_abs'] = np.abs(df['real'] - df['pred'])
    df['es_extremo'] = df['real'] > umbral

    resumen = df.groupby('station_id').agg(
        mae_medio=('error_abs', 'mean'),
        n_extremos=('es_extremo', 'sum'),
        n_registros=('real', 'count')
    ).reset_index().sort_values('mae_medio', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfica 1: MAE por estación
    colores = ['red' if n > 0 else 'steelblue' for n in resumen['n_extremos']]
    axes[0].barh(resumen['station_id'], resumen['mae_medio'],
                 color=colores, edgecolor='none', alpha=0.85)
    axes[0].set_xlabel('MAE medio (µg/m³)')
    axes[0].set_title('Error medio por estación\n(rojo = tiene registros extremos)')

    # Gráfica 2: Número de extremos por estación
    resumen_ext = resumen[resumen['n_extremos'] > 0].sort_values('n_extremos', ascending=True)
    if not resumen_ext.empty:
        axes[1].barh(resumen_ext['station_id'], resumen_ext['n_extremos'],
                     color='red', edgecolor='none', alpha=0.75)
        axes[1].set_xlabel(f'Nº registros con O₃ > {umbral} µg/m³')
        axes[1].set_title(f'Estaciones con registros extremos (O₃ > {umbral})')
    else:
        axes[1].text(0.5, 0.5, 'Sin registros extremos en test',
                     ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()

    if guardar:
        fig.savefig(OUTPUT_DIR / 'errores_por_estacion.png', dpi=150)
        print("[INFO] Guardado: errores_por_estacion.png")
    plt.show()
    plt.close()
'''

def plot_distribucion_temporal_extremos(df, umbral=170, guardar=True):
    """
    Muestra en qué meses y años se concentran los registros extremos.
    Refuerza el argumento de que el fenómeno es raro e irregular,
    lo que explica la limitación del modelo.
    """
    df_ext = df[df['O3'] > umbral].copy()

    if df_ext.empty:
        print(f"[INFO] No hay registros con O3 > {umbral} para graficar.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Por mes
    conteo_mes = df_ext.groupby('month').size().reindex(range(1, 13), fill_value=0)
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    axes[0].bar(conteo_mes.index, conteo_mes.values,
                color='darkorange', edgecolor='none', alpha=0.85,
                tick_label=meses)
    axes[0].set_xlabel('Mes')
    axes[0].set_ylabel('Nº registros')
    axes[0].set_title(f'Distribución mensual de registros con O₃ > {umbral} µg/m³')

    # Por año
    conteo_anyo = df_ext.groupby('year').size()
    axes[1].bar(conteo_anyo.index.astype(str), conteo_anyo.values,
                color='darkorange', edgecolor='none', alpha=0.85)
    axes[1].set_xlabel('Año')
    axes[1].set_ylabel('Nº registros')
    axes[1].set_title(f'Distribución anual de registros con O₃ > {umbral} µg/m³')

    plt.tight_layout()

    if guardar:
        fig.savefig(OUTPUT_DIR / 'distribucion_temporal_extremos.png', dpi=150)
        print("[INFO] Guardado: distribucion_temporal_extremos.png")
    plt.show()
    plt.close()


def plot_degradacion_por_horizonte(resultados_horizontes, guardar=True):
    """
    Muestra cómo degradan MAE y R² al aumentar el horizonte de predicción.
    """
    df = pd.DataFrame(resultados_horizontes)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(df['horizon'], df['MAE'], 'o-', color='steelblue', linewidth=2)
    axes[0].set_xlabel('Horizonte (horas)')
    axes[0].set_ylabel('MAE (µg/m³)')
    axes[0].set_title('Error medio por horizonte de predicción')
    axes[0].set_xticks(df['horizon'])

    axes[1].plot(df['horizon'], df['R2'], 'o-', color='darkorange', linewidth=2)
    axes[1].set_xlabel('Horizonte (horas)')
    axes[1].set_ylabel('R²')
    axes[1].set_title('R² por horizonte de predicción')
    axes[1].set_xticks(df['horizon'])

    plt.tight_layout()
    if guardar:
        fig.savefig(OUTPUT_DIR / 'degradacion_por_horizonte.png', dpi=150)
        print("[INFO] Guardado: degradacion_por_horizonte.png")
    plt.show()
    plt.close()

