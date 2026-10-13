"""Paso 05: ingenieria de features por parcela.

Construye la tabla maestra de modelado (197 filas) a partir de las salidas
de los pasos 01 a 03. Bloques:

  1. Fenologia 2025 por indice (Sentinel-2 y Planet por separado): la serie
     de cada parcela se agrega por dia, se interpola a paso diario y se
     suaviza con Savitzky-Golay; de la curva suavizada se extraen valor y
     fecha del pico, integral del ciclo, tasa de crecimiento (verdeo) y
     tasa de senescencia.
  2. Clima: valores mensuales de abril a octubre 2025, agregados del ciclo,
     grados-dia acumulados (base 0 C, estandar para cebada), meses secos y
     anomalia de precipitacion contra el promedio 2022 a 2024.
  3. Historia de la parcela: integral NDVI de los ciclos 2022 a 2024.
  4. Estaticos: elevacion, pendiente, area, lon, lat, estado y municipio.

Salida:
  salidas/features_modelado.csv
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from pipeline import conf as config

# Indices a los que se les extrae fenologia completa desde Sentinel-2.
INDICES_S2 = ["ndvi", "evi", "lai", "fapar", "savi", "msi", "ndwi",
              "dswi2", "dswi5", "nddi", "ndti", "crc", "mcrc", "ndsvi"]
INDICES_PLANET = ["ndvi", "evi", "lai", "msavi"]

DOY_INICIO = pd.Timestamp(config.CICLO_INICIO).day_of_year   # 91
DOY_FIN = pd.Timestamp(config.CICLO_FIN).day_of_year         # 304


def fenologia_serie(dias: np.ndarray, valores: np.ndarray) -> dict:
    """Metricas fenologicas de una serie (dia del anio, valor) ya limpia."""
    if len(dias) < 8:
        return {}
    # Agregar por dia (puede haber mas de una observacion el mismo dia)
    s = pd.Series(valores, index=dias).groupby(level=0).mean()
    # Malla diaria dentro del rango observado, interpolacion lineal
    malla = np.arange(s.index.min(), s.index.max() + 1)
    curva = np.interp(malla, s.index.values, s.values)
    # Suavizado Savitzky-Golay: ventana ~1 mes, polinomio de grado 2
    ventana = min(31, len(curva) if len(curva) % 2 == 1 else len(curva) - 1)
    if ventana >= 7:
        curva = savgol_filter(curva, ventana, 2)
    # Recorte al ciclo
    dentro = (malla >= DOY_INICIO) & (malla <= DOY_FIN)
    if dentro.sum() < 30:
        return {}
    mc, cc = malla[dentro], curva[dentro]
    i_max = int(np.argmax(cc))
    pico, doy_pico = float(cc[i_max]), int(mc[i_max])
    dias_subida = max(doy_pico - int(mc[0]), 1)
    dias_bajada = max(int(mc[-1]) - doy_pico, 1)
    return {
        "pico": pico,
        "doy_pico": doy_pico,
        "auc": float(np.trapezoid(cc, mc) / 100.0),
        "tasa_verdeo": float((pico - cc[0]) / dias_subida * 100),
        "tasa_senescencia": float((cc[-1] - pico) / dias_bajada * 100),
    }


def features_fenologia(df: pd.DataFrame, indices: list[str], sufijo: str) -> pd.DataFrame:
    """Fenologia por parcela para cada indice, sobre observaciones limpias de 2025."""
    df = df[df["limpia"] & (df["fecha"].dt.year == 2025)].copy()
    df["doy"] = df["fecha"].dt.day_of_year
    filas = {}
    for pid, g in df.groupby("ID_POLIGONO"):
        fila = {}
        for idx in indices:
            col = f"{idx}_promedio"
            sub = g.dropna(subset=[col])
            met = fenologia_serie(sub["doy"].values, sub[col].values)
            for k, v in met.items():
                fila[f"{idx}_{k}{sufijo}"] = v
        # Variabilidad espacial interna en el pico del ciclo (jul a sep)
        pico = g[(g["fecha"].dt.month.isin([7, 8, 9]))]
        if "ndvi_std" in g.columns and len(pico):
            fila[f"ndvi_std_pico{sufijo}"] = float(pico["ndvi_std"].mean())
        filas[pid] = fila
    return pd.DataFrame(filas).T.rename_axis("ID_POLIGONO").reset_index()


def features_clima() -> pd.DataFrame:
    clima = pd.read_csv(config.RUTA_SALIDAS / "clima_mensual_parcela.csv")
    c25 = clima[(clima["anio"] == 2025) & clima["mes"].between(4, 10)]

    # Valores mensuales del ciclo en formato ancho
    ancho = c25.pivot(index="ID_POLIGONO", columns="mes",
                      values=["prec_mm", "tmax_c", "tmin_c"])
    ancho.columns = [f"{var}_m{mes:02d}" for var, mes in ancho.columns]

    # Agregados del ciclo
    dias_mes = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31}
    agg = c25.assign(
        tmedia=(c25["tmax_c"] + c25["tmin_c"]) / 2,
        dias=c25["mes"].map(dias_mes),
    )
    agg["gdd_mes"] = agg["tmedia"].clip(lower=0) * agg["dias"]  # base 0 C
    resumen = agg.groupby("ID_POLIGONO").agg(
        prec_ciclo=("prec_mm", "sum"),
        tmax_ciclo=("tmax_c", "mean"),
        tmin_ciclo=("tmin_c", "mean"),
        gdd_ciclo=("gdd_mes", "sum"),
        meses_secos=("prec_mm", lambda s: int((s < 50).sum())),
    )
    resumen["amplitud_termica"] = resumen["tmax_ciclo"] - resumen["tmin_ciclo"]

    # Anomalia de precipitacion del ciclo 2025 contra el promedio 2022 a 2024
    hist = clima[(clima["anio"] < 2025) & clima["mes"].between(4, 10)]
    prec_hist = hist.groupby(["ID_POLIGONO", "anio"])["prec_mm"].sum() \
                    .groupby("ID_POLIGONO").mean().rename("prec_ciclo_hist")
    resumen = resumen.join(prec_hist)
    resumen["prec_anomalia"] = resumen["prec_ciclo"] - resumen["prec_ciclo_hist"]

    return ancho.join(resumen).reset_index()


def features_historia_ndvi() -> pd.DataFrame:
    """Integral NDVI (suma de medias mensuales del ciclo) de 2022 a 2024, S2 limpio."""
    s2 = pd.read_csv(config.RUTA_SALIDAS / "satelital_s2.csv", low_memory=False,
                     usecols=["ID_POLIGONO", "fecha", "limpia", "ndvi_promedio"])
    s2["fecha"] = pd.to_datetime(s2["fecha"])
    s2 = s2[s2["limpia"] & s2["ndvi_promedio"].notna()]
    s2["anio"] = s2["fecha"].dt.year
    s2["mes"] = s2["fecha"].dt.month
    s2 = s2[(s2["anio"] < 2025) & s2["mes"].between(4, 10)]
    auc = (s2.groupby(["ID_POLIGONO", "anio", "mes"])["ndvi_promedio"].mean()
             .groupby(["ID_POLIGONO", "anio"]).sum().unstack())
    auc.columns = [f"ndvi_auc_{a}" for a in auc.columns]
    auc["ndvi_auc_hist_media"] = auc.mean(axis=1)
    return auc.reset_index()


def main() -> pd.DataFrame:
    maestro = pd.read_csv(config.RUTA_SALIDAS / "parcelas_maestro.csv")
    topo = pd.read_csv(config.RUTA_SALIDAS / "topografia_parcela.csv")

    s2 = pd.read_csv(config.RUTA_SALIDAS / "satelital_s2.csv", low_memory=False,
                     parse_dates=["fecha"])
    planet = pd.read_csv(config.RUTA_SALIDAS / "satelital_planet.csv", low_memory=False,
                         parse_dates=["fecha"])

    feats = maestro.merge(topo, on="ID_POLIGONO")
    feats = feats.merge(features_fenologia(s2, INDICES_S2, ""), on="ID_POLIGONO", how="left")
    feats = feats.merge(features_fenologia(planet, INDICES_PLANET, "_pl"),
                        on="ID_POLIGONO", how="left")
    feats = feats.merge(features_clima(), on="ID_POLIGONO", how="left")
    feats = feats.merge(features_historia_ndvi(), on="ID_POLIGONO", how="left")

    # Suelo (paso 05b, opcional): se integra solo si ya se corrio
    ruta_suelo = config.RUTA_SALIDAS / "suelo_parcela.csv"
    if ruta_suelo.exists():
        feats = feats.merge(pd.read_csv(ruta_suelo), on="ID_POLIGONO", how="left")
        print("Suelo de SoilGrids integrado.")
    else:
        print("Aviso: suelo_parcela.csv no existe; correr paso05b_soilgrids.py "
              "(requiere internet) y repetir este paso para integrarlo.")

    feats.to_csv(config.RUTA_SALIDAS / "features_modelado.csv", index=False)

    # Auditoria de nulos
    numericas = feats.select_dtypes(include=[float, int]).drop(columns=["RENDIMIENTO_T_HA"])
    pct_nulos = numericas.isna().mean()
    problema = pct_nulos[pct_nulos > 0.05].sort_values(ascending=False)
    print(f"features_modelado.csv: {feats.shape[0]} parcelas x {feats.shape[1]} columnas")
    print(f"Columnas con mas de 5 % de nulos: {len(problema)}")
    if len(problema):
        print(problema.round(3).to_string())
    return feats


if __name__ == "__main__":
    main()
