"""Paso 02: limpieza de los datasets satelitales.

Decisiones metodologicas (documentar en el reporte):
  1. El dataset Basico se separa por sensor: las filas de Landsat solo
     traen el indice VI6T (el resto de columnas viene vacio por diseno),
     y las de Sentinel-2 traen los otros 22 indices pero no VI6T.
  2. Cada observacion se marca con la bandera `limpia` si su nubosidad es
     menor al umbral de config (20 %). No se tiran filas: el filtrado
     final lo decide cada experimento, pero la bandera deja el criterio
     unificado para todo el equipo.
  3. Planet (PRO) se mantiene como fuente separada; el diccionario del
     reto advierte que no debe fusionarse con Sentinel-2 como si fueran
     mediciones equivalentes.

Salidas:
  salidas/satelital_s2.csv       (Sentinel-2, 22 indices x 4 estadisticos)
  salidas/satelital_landsat.csv  (Landsat, solo VI6T)
  salidas/satelital_planet.csv   (Planet, 4 indices x 4 estadisticos)
"""
import pandas as pd

from pipeline import conf as config


def _preparar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha_captura"], dayfirst=True)
    df["limpia"] = df["porcentaje_nubosidad"] < config.UMBRAL_NUBOSIDAD
    df["en_ciclo_2025"] = df["fecha"].between(config.CICLO_INICIO, config.CICLO_FIN)
    return df.drop(columns=["fecha_captura"]).sort_values(["ID_POLIGONO", "fecha"])


def procesar_basico() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(config.CSV_BASICO, low_memory=False)
    df = _preparar(df)

    cols_vi6t = [c for c in df.columns if c.startswith("vi6t")]

    s2 = df[df["sensor"] == "Sentinel-2"].drop(columns=cols_vi6t)
    landsat_cols = ["ID_POLIGONO", "fecha", "sensor", "porcentaje_nubosidad",
                    "limpia", "en_ciclo_2025"] + cols_vi6t
    landsat = df[df["sensor"] == "Landsat"][landsat_cols]

    s2.to_csv(config.RUTA_SALIDAS / "satelital_s2.csv", index=False)
    landsat.to_csv(config.RUTA_SALIDAS / "satelital_landsat.csv", index=False)
    return s2, landsat


def procesar_planet() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_PRO, low_memory=False)
    df = _preparar(df)
    df.to_csv(config.RUTA_SALIDAS / "satelital_planet.csv", index=False)
    return df


def main():
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    s2, landsat = procesar_basico()
    planet = procesar_planet()
    for nombre, d in [("Sentinel-2", s2), ("Landsat", landsat), ("Planet", planet)]:
        limpias = d[d.limpia & d.en_ciclo_2025].groupby("ID_POLIGONO").size()
        print(f"{nombre}: {len(d):,} filas | obs limpias en ciclo 2025 por parcela: "
              f"mediana {limpias.median():.0f}, minimo {limpias.min() if len(limpias) else 0:.0f}")


if __name__ == "__main__":
    main()
