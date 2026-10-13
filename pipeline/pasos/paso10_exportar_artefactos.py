"""Paso 10: exporta los artefactos congelados que carga la API.

La API NO corre el pipeline: solo lee la carpeta artefactos/ que escribe este paso
(el unico de los pasos que conoce el formato que espera el backend). Exporta solo
la mezcla (ensamble + LightGBM de solo longitud).

artefactos/
  modelo/
    imputador.joblib       SimpleImputer (mediana) ajustado con las 138 parcelas
    escalador.joblib       StandardScaler (solo lo usa Elastic Net)
    lgbm.joblib, rf.joblib, enet.joblib   los tres miembros del ensamble
    lgbm_longitud.joblib   LightGBM que solo usa la longitud
  metadata.json            orden de variables, pesos de la mezcla, versiones de librerias,
                           cuantil del intervalo, metricas, rangos de entrenamiento, etiquetas
                           y checksums de los .joblib
  metricas_dashboard.json  lo que muestra la seccion "Desempeno del modelo" y "Hallazgos"
  tablas/
    parcelas.csv                 las 197 parcelas (id, estado, municipio, area, conjunto, observado, ...)
    predicciones.csv             prediccion e intervalo de las 197 (las 59 ocultas con el modelo final;
                                 las 138 medidas, fuera de muestra)
    features_modelo.csv          variables del modelo de cada parcela (entrada del simulador)
    shap_parcela.csv             8 variables con mas efecto por parcela
    clima_mensual.csv            clima del ciclo 2025
    serie_satelital.csv.gz       curvas satelitales (Sentinel-2 y Planet) si hay dataset
    vecinos.csv                  5 parcelas mas cercanas de cada parcela

Los .joblib se guardan con las versiones de requirements-pipeline.txt, que son las mismas
que fija requirements.txt de la API: si cambias una, cambia la otra y vuelve a exportar.
"""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from pipeline import conf as config
from pipeline import etiquetas
from pipeline.modelado import cargar_datos

VERSION_ARTEFACTOS = 1
INDICES_SERIE = ["ndvi", "evi", "lai", "nddi", "ndwi"]
FUENTES = {"satelital_s2.csv": "S2", "satelital_planet.csv": "Planet"}
TOP_SHAP = 8
N_VECINOS = 5

NOMBRES = {
    "ensamble": "Ensamble", "ubicacion": "Solo ubicación", "mezcla": "Mezcla",
}
DESCRIPCION = {
    "ensamble": "LightGBM, Random Forest y Elastic Net con todas las variables.",
    "ubicacion": "LightGBM que solo usa la longitud. Es lo que encontró el análisis de modelos clásicos.",
    "mezcla": "Promedio de las predicciones del ensamble y del modelo de ubicación.",
}


def r(x, d=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def sha256(ruta) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------
def exportar_modelo(modelo, destino) -> dict:
    d = destino / "modelo"
    d.mkdir(parents=True, exist_ok=True)
    piezas = {
        "imputador": modelo.imputador, "escalador": modelo.escalador, "lgbm": modelo.lgbm,
        "rf": modelo.rf, "enet": modelo.enet, "lgbm_longitud": modelo.ubicacion,
    }
    sumas = {}
    for nombre, obj in piezas.items():
        ruta = d / f"{nombre}.joblib"
        joblib.dump(obj, ruta, compress=3)
        sumas[f"modelo/{nombre}.joblib"] = sha256(ruta)
    return sumas


def exportar_tablas(datos, modelo, q90, destino, s) -> dict:
    t = destino / "tablas"
    t.mkdir(parents=True, exist_ok=True)
    cols = modelo.cols

    maestro = pd.read_csv(s / "parcelas_maestro.csv").set_index("ID_POLIGONO")
    topo = pd.read_csv(s / "topografia_parcela.csv").set_index("ID_POLIGONO")

    # parcelas.csv
    parcelas = pd.DataFrame({
        "id": maestro.index, "estado": maestro["Estado"].values, "municipio": maestro["Municipio"].values,
        "area_ha": maestro["AREA_HA"].values, "conjunto": maestro["CONJUNTO"].values,
        "rendimiento_obs": maestro["RENDIMIENTO_T_HA"].values, "lat": maestro["lat"].values,
        "lon": maestro["lon"].values, "elevacion_m": topo.loc[maestro.index, "elevacion_m"].values,
        "pendiente_grados": topo.loc[maestro.index, "pendiente_grados"].values,
    }).sort_values("id")
    parcelas.to_csv(t / "parcelas.csv", index=False)

    # features_modelo.csv: entrada cruda del modelo (con NaN, el imputador los resuelve)
    feats = datos.X_todas[cols].copy()
    feats.insert(0, "id", datos.ids_todas)
    feats.sort_values("id").to_csv(t / "features_modelo.csv", index=False)

    # predicciones.csv: las 59 ocultas con el modelo final, las 138 medidas fuera de muestra
    oof = pd.read_csv(s / "oof_predicciones.csv").set_index("ID_POLIGONO")
    p59 = modelo.predecir(datos.X_pred)["mezcla"]
    pred = {pid: float(v) for pid, v in zip(datos.ids_pred, p59)}
    for pid in datos.ids:
        pred[pid] = float(oof.loc[pid, "estrat_mezcla"])
    medidas = set(datos.ids)

    base = float(datos.y.mean())
    shap = pd.read_csv(s / "shap_parcela.csv")
    filas_shap, otras = [], {}
    for pid, g in shap.groupby("ID_POLIGONO"):
        suma = 0.0
        for f in g.sort_values("orden").head(TOP_SHAP).itertuples():
            valor, unidad = etiquetas.valor_y_unidad(f.variable, f.valor_variable)
            filas_shap.append({"id": pid, "orden": f.orden, "variable": f.variable,
                               "etiqueta": etiquetas.etiqueta(f.variable), "valor_variable": round(valor, 5),
                               "unidad": unidad, "contribucion": round(f.contribucion, 3)})
            suma += round(f.contribucion, 3)
        otras[pid] = round(pred[pid] - base - suma, 3)
    pd.DataFrame(filas_shap).to_csv(t / "shap_parcela.csv", index=False)

    ids = sorted(pred)
    pd.DataFrame({
        "id": ids, "prediccion": [pred[i] for i in ids],
        "inferior": [pred[i] - q90 for i in ids], "superior": [pred[i] + q90 for i in ids],
        "fuera_de_muestra": [i in medidas for i in ids], "shap_otras": [otras[i] for i in ids],
    }).to_csv(t / "predicciones.csv", index=False, float_format="%.12g")

    # clima del ciclo 2025
    clima = pd.read_csv(s / "clima_mensual_parcela.csv")
    clima = clima[(clima["anio"] == 2025) & clima["mes"].between(4, 10)]
    clima.rename(columns={"ID_POLIGONO": "id"}).drop(columns="anio").round(3).to_csv(t / "clima_mensual.csv", index=False)

    # vecinos
    lat, lon = parcelas["lat"].values, parcelas["lon"].values
    filas_v = []
    for i, pid in enumerate(parcelas["id"].values):
        d = haversine_km(lat[i], lon[i], lat, lon)
        d[i] = np.inf
        for orden, j in enumerate(np.argsort(d)[:N_VECINOS], start=1):
            filas_v.append({"id": pid, "orden": orden, "vecino": parcelas["id"].values[j], "distancia_km": round(float(d[j]), 3)})
    pd.DataFrame(filas_v).to_csv(t / "vecinos.csv", index=False)

    # series satelitales (solo si el paso 02 las genero)
    n_serie = 0
    series = []
    for archivo, fuente in FUENTES.items():
        ruta = s / archivo
        if not ruta.exists():
            continue
        usar = ["ID_POLIGONO", "fecha", "limpia"] + [f"{i}_promedio" for i in INDICES_SERIE]
        df = pd.read_csv(ruta, low_memory=False, usecols=lambda c: c in usar, parse_dates=["fecha"])
        df = df[df["limpia"] & df["fecha"].between("2025-03-15", "2025-11-15")]
        for indice in INDICES_SERIE:
            col = f"{indice}_promedio"
            if col in df.columns:
                sub = df.dropna(subset=[col])
                series.append(pd.DataFrame({"id": sub["ID_POLIGONO"], "fecha": sub["fecha"].dt.strftime("%Y-%m-%d"),
                                            "fuente": fuente, "indice": indice, "valor": sub[col].round(4)}))
    if series:
        serie = pd.concat(series).sort_values(["id", "fuente", "indice", "fecha"])
        serie.to_csv(t / "serie_satelital.csv.gz", index=False)
        n_serie = len(serie)
    else:
        (t / "serie_satelital.csv.gz").unlink(missing_ok=True)
    return {"parcelas": len(parcelas), "serie_satelital": n_serie, "base": base, "parcelas_df": parcelas}


# ----------------------------------------------------------------------
def metricas_dashboard(s, datos, base_rmse) -> dict:
    est = pd.read_csv(s / "metricas_optimizadas.csv").set_index("modelo")
    lomo = pd.read_csv(s / "metricas_lomo.csv").set_index("modelo")
    comp = pd.read_csv(s / "comparacion_candidatos.csv")
    maestro = pd.read_csv(s / "parcelas_maestro.csv").set_index("ID_POLIGONO")
    final = config.MODELO_FINAL

    comparacion = []
    for c in config.CANDIDATOS:
        fila = {"id": c, "nombre": NOMBRES[c], "descripcion": DESCRIPCION[c], "elegido": c == final,
                "n_variables": int(comp[comp.candidato == c].iloc[0].n_variables)}
        for v in ("estratificada", "lomo"):
            x = comp[(comp.candidato == c) & (comp.validacion == v)].iloc[0]
            fila[v] = {"rmse": r(x.rmse), "mae": r(x.mae), "r2": r(x.r2, 2), "sesgo": r(x.sesgo)}
        comparacion.append(fila)

    oof = pd.read_csv(s / "oof_predicciones.csv")
    dispersion = [{"id": f.ID_POLIGONO, "estado": maestro.loc[f.ID_POLIGONO, "Estado"], "municipio": f.municipio,
                   "obs": r(f.y, 2), "pred": r(f.estrat_mezcla, 2)} for f in oof.itertuples()]
    residuales = [r(d["obs"] - d["pred"]) for d in dispersion]

    rob = pd.read_csv(s / "robustez_candidatos.csv")
    rob = rob[(rob.n_parcelas >= 5) & (rob.modelo == final)].sort_values("rmse")
    robustez = [{"municipio": f.municipio_fuera, "rmse": r(f.rmse, 2), "n": int(f.n_parcelas)} for f in rob.itertuples()]

    imp = pd.read_csv(s / "shap_importancia.csv")
    familias = imp.groupby("familia")["shap_medio_abs"].sum().sort_values(ascending=False)

    mez_e, mez_l = est.loc[final], lomo.loc[final]
    return {
        "modelo": {"nombre": NOMBRES[final], "descripcion": DESCRIPCION[final],
                   "n_variables": int(comp[(comp.candidato == final)].iloc[0].n_variables)},
        "validacion": {
            "esquema": "Validación cruzada anidada estratificada por municipio",
            "rmse": r(mez_e.rmse), "mae": r(mez_e.mae), "r2": r(mez_e.r2, 2), "sesgo": r(mez_e.sesgo),
            "baseline": {"nombre": "Predecir el promedio", "rmse": r(base_rmse), "r2": 0},
        },
        "lomo": {
            "esquema": "Un municipio completo fuera en cada ronda (LOMO)",
            "rmse": r(mez_l.rmse), "mae": r(mez_l.mae), "r2": r(mez_l.r2, 2), "sesgo": r(mez_l.sesgo),
            "baseline": {"nombre": "Promedio de las demás zonas", "rmse": r(lomo.loc["media_resto", "rmse"]),
                         "r2": r(lomo.loc["media_resto", "r2"], 2)},
        },
        "comparacion": comparacion,
        "dispersion": dispersion,
        "residuales": residuales,
        "importancia_familias": [{"familia": etiquetas.familia_nombre(f), "valor": r(v)} for f, v in familias.head(10).items()],
        "robustez_espacial": robustez,
        "hallazgos": hallazgos(s, imp, familias),
        "n_parcelas": {"total": int(len(maestro)), "medidas": int(len(datos.y)), "ocultas": int(len(maestro) - len(datos.y))},
        "rango_entrenamiento": {
            "lluvia_mayo": {"min": float(datos.X["prec_mm_m05"].min()), "max": float(datos.X["prec_mm_m05"].max())},
            "tmin_abril": {"min": float(datos.X["tmin_c_m04"].min()), "max": float(datos.X["tmin_c_m04"].max())},
        },
    }


def hallazgos(s, imp, familias) -> dict:
    sat = familias[[f for f in familias.index if f.startswith("satelital:")]]
    indices = [{"etiqueta": etiquetas.familia_nombre(f), "valor": r(v)} for f, v in sat.head(3).items()]

    ndvi = imp[imp.familia == "satelital:ndvi"].copy()

    def grupo(v):
        if "tasa_verdeo" in v:
            return "Velocidad de verdeo"
        if "tasa_senescencia" in v:
            return "Velocidad de senescencia"
        if v in ("ndvi_pico", "ndvi_pico_pl"):
            return "Valor máximo"
        return "Otras medidas"

    ndvi["grupo"] = ndvi.variable.map(grupo)
    dentro = ndvi.groupby("grupo")["shap_medio_abs"].sum()
    orden = ["Velocidad de verdeo", "Velocidad de senescencia", "Valor máximo", "Otras medidas"]
    dentro_ndvi = [{"etiqueta": g, "valor": r(dentro.get(g, 0.0))} for g in orden]

    clima = imp[imp.familia == "clima"].sort_values("shap_medio_abs", ascending=False)
    clima_items = [{"etiqueta": etiquetas.CORTAS.get(f.variable, etiquetas.etiqueta(f.variable)), "valor": r(f.shap_medio_abs)}
                   for f in clima.head(6).itertuples()]
    clima_items.append({"etiqueta": "Resto del clima", "valor": r(clima.iloc[6:].shap_medio_abs.sum())})

    abla = pd.read_csv(s / "ablacion_bloques.csv")
    bloques = [{"etiqueta": etiquetas.BLOQUES.get(f.bloque, f.bloque), "clave": f.bloque, "rmse": r(f.rmse),
                "n_variables": int(f.n_variables)} for f in abla.itertuples()]
    return {"indices": indices, "dentro_ndvi": dentro_ndvi, "clima": clima_items, "bloques": bloques,
            "familia_clima": r(familias.get("clima", 0.0), 3)}


# ----------------------------------------------------------------------
def main():
    s = config.RUTA_SALIDAS
    destino = config.RUTA_ARTEFACTOS
    destino.mkdir(parents=True, exist_ok=True)

    art = joblib.load(config.RUTA_MODELOS / "modelo_final.joblib")
    modelo, cols, q90 = art["modelo"], art["cols"], art["q90"][config.MODELO_FINAL]
    datos = cargar_datos()

    sumas = exportar_modelo(modelo, destino)
    info = exportar_tablas(datos, modelo, q90, destino, s)
    parcelas = info.pop("parcelas_df")

    base_rmse = float(pd.read_csv(s / "metricas_baseline.csv").set_index("modelo").loc["media_global", "rmse"])
    dash = metricas_dashboard(s, datos, base_rmse)
    (destino / "metricas_dashboard.json").write_text(json.dumps(dash, ensure_ascii=False, indent=1))

    # rangos de entrenamiento de cada variable del modelo
    rangos = {c: {"min": float(datos.X[c].min()), "max": float(datos.X[c].max())} for c in cols}
    detalle_vars = {}
    for c in cols:
        _, unidad = etiquetas.valor_y_unidad(c, 1.0)
        factor = 100.0 if unidad == "por día" else 1.0
        detalle_vars[c] = {"etiqueta": etiquetas.etiqueta(c), "unidad": unidad, "divisor": factor}

    entrena = parcelas[parcelas.conjunto == "ENTRENAMIENTO"]
    municipios = {}
    for (est, mun), g in entrena.groupby(["estado", "municipio"]):
        municipios[mun] = {"estado": est, "n": int(len(g)), "lat": float(g.lat.mean()), "lon": float(g.lon.mean()),
                           "lat_min": float(g.lat.min()), "lat_max": float(g.lat.max()),
                           "lon_min": float(g.lon.min()), "lon_max": float(g.lon.max())}
    # distancia al vecino mas cercano dentro de las 197: referencia de "que tan lejos es lejos"
    lat, lon = parcelas["lat"].values, parcelas["lon"].values
    nn = []
    for i in range(len(parcelas)):
        d = haversine_km(lat[i], lon[i], lat, lon)
        d[i] = np.inf
        nn.append(d.min())

    metadata = {
        "version_artefactos": VERSION_ARTEFACTOS,
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": {"nombre": "mezcla", "descripcion": DESCRIPCION["mezcla"]},
        "variables": cols,
        "variable_longitud": "lon",
        "detalle_variables": detalle_vars,
        "pesos_mezcla": {"lgbm": 1 / 6, "rf": 1 / 6, "enet": 1 / 6, "lgbm_longitud": 1 / 2},
        "cuantil_intervalo": q90,
        "nivel_intervalo": 0.90,
        "base": info["base"],
        "metricas": {"estratificada": dash["validacion"], "lomo": dash["lomo"]},
        "rangos_entrenamiento": rangos,
        "municipios_entrenamiento": municipios,
        "zona": {
            "lat_min": float(entrena.lat.min()), "lat_max": float(entrena.lat.max()),
            "lon_min": float(entrena.lon.min()), "lon_max": float(entrena.lon.max()),
            "vecino_mediana_km": float(np.median(nn)), "vecino_p95_km": float(np.quantile(nn, 0.95)),
        },
        "librerias": {
            "python": platform.python_version(),
            **{m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "lightgbm", "joblib")},
        },
        "checksums": sumas,
        "tablas": {"parcelas": info["parcelas"], "observaciones_satelitales": info["serie_satelital"]},
    }
    metadata["librerias"]["scikit-learn"] = metadata["librerias"].pop("sklearn")
    (destino / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1))

    print(f"Artefactos escritos en {destino}")
    print(f"  variables: {len(cols)} | cuantil del intervalo (mezcla): {q90:.4f} t/ha | base: {info['base']:.3f} t/ha")
    print(f"  {info['parcelas']} parcelas, {info['serie_satelital']} observaciones satelitales")
    print("  librerias:", ", ".join(f"{k} {v}" for k, v in metadata["librerias"].items()))
    if not info["serie_satelital"]:
        print("  Aviso: sin satelital_s2.csv ni satelital_planet.csv (corre los pasos 01 a 03): la ficha no tendra curvas.")


if __name__ == "__main__":
    main()
