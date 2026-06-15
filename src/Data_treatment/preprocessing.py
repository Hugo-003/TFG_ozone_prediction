import pandas as pd
import numpy as np
from pathlib import Path


def find_header_row(path):
    with open(path, "r", encoding="latin-1", errors='replace') as f:
        for i, line in enumerate(f):
            if line.strip().startswith("FECHA"):
                return i
    raise ValueError(f"No se encontró la cabecera FECHA en {path}")


def read_mh(path):
    header_row = find_header_row(path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        encoding="latin-1",
        engine="python",
        skiprows=header_row
    )
    df = df.iloc[1:].reset_index(drop=True)
    return df


def clean_numeric_data(df):
    rename_dict = {
        "VEL": "wind_speed", "VMAX": "wind_speed_max", "DIR": "wind_dir",
        "TEM": "temp_air", "TCAB": "temp_cabin", "HR": "humidity",
        "HRCAB": "humidity_cabin", "RAD": "solar_radiation",
        "PLU": "precipitation", "PRE": "pressure",
        "PM2.5": "pm25", "PM10": "pm10",
        "PM2.5s/c": "pm25_sc", "PM10s/c": "pm10_sc"
    }
    df = df.rename(columns=rename_dict)

    if "O3" not in df.columns:
        return pd.DataFrame()

    # Columnas a borrar por baja representatividad
    cols_a_borrar = [
        'C6H6', 'C7H8', 'C8H10', 'PM1', 'pm25_sc', 'pm10_sc',
        'PM1s/c', 'SH2', 'NH3', 'RUI', 'VDST'
    ]
    df = df.drop(columns=[c for c in cols_a_borrar if c in df.columns])

    metadata_cols = ["station_id", "year", "datetime", "hour", "day", "month", "dayofweek"]
    sensor_cols = [c for c in df.columns if c not in metadata_cols]

    # Conversión a numérico — esto convierte strings "-99" a float primero
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace de códigos de error DESPUÉS de la conversión numérica
    # Así pillamos -99 independientemente de si llegaron como string o float
    ERROR_CODES = [-99, -99.9, -999, -999.9]
    df[sensor_cols] = df[sensor_cols].replace(ERROR_CODES, np.nan)

    # Filtrar registros sin O3
    df = df.dropna(subset=["O3"]).copy()

    if not df.empty:
        df[sensor_cols] = df[sensor_cols].ffill(limit=3).bfill(limit=3)

    return df


def enrich_df(df, filepath):
    name = Path(filepath).stem
    station_id = name[:-2]
    try:
        year = int("20" + name[-2:])
    except ValueError:
        print(f"Warning: no se pudo parsear el año de {name}")
        year = 0

    df["station_id"] = station_id
    df["year"] = year

    if "FECHA" in df.columns and "HORA" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["FECHA"] + " " + df["HORA"],
            format="%m/%d/%Y %H:%M",
            errors='coerce'
        )
        df = df.dropna(subset=["datetime"])
        df["hour"]      = df["datetime"].dt.hour
        df["day"]       = df["datetime"].dt.day
        df["month"]     = df["datetime"].dt.month
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df = df.drop(columns=["FECHA", "HORA"])

    df = clean_numeric_data(df)

    cols_inicio = ["station_id", "year", "datetime", "hour", "day", "month", "dayofweek"]
    existing_cols = [c for c in cols_inicio if c in df.columns]
    df = df[existing_cols + [c for c in df.columns if c not in existing_cols]]

    return df


def load_all_data(folder_path):
    all_dfs = []
    files = list(Path(folder_path).glob("*.MH"))

    for i, file in enumerate(files, 1):
        print(f"Procesando {i}/{len(files)}: {file.name}")
        try:
            df = read_mh(file)
            df = enrich_df(df, file)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"Error procesando {file.name}: {e}")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def save_processed_data(df, output_path):
    output_path = Path(output_path).with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Datos guardados en {output_path}")
