"""Prediccion para una parcela nueva a partir de lo poco que se sabe de ella.

El usuario da lat, lon, municipio, area y, si quiere, clima mensual. El modelo usa 121
variables: casi todas (curvas satelitales, topografia, historia del NDVI) no se pueden
conocer sin imagenes de esa parcela, asi que se imputan asi:

  1. Se buscan las `K_VECINOS` parcelas conocidas mas cercanas (de las 197), dando prioridad
     al mismo municipio si tiene suficientes.
  2. Cada variable que falta se toma como el promedio de esos vecinos ponderado por
     1 / distancia^2 (con un piso de 0.5 km), no de uno solo, para no heredar el ruido de un
     vecino. Si el vecino mas cercano esta a mas de `DIST_MEDIANA_KM`, se usa la mediana del
     municipio (o la de entrenamiento si el municipio no se conoce).
  3. Lo que el usuario da (lat, lon, area, municipio y clima) siempre manda.

La respuesta dice que variables se imputaron y con que, y avisa cuando la ubicacion queda
lejos o fuera de las zonas de entrenamiento. Es una ESTIMACION DE ZONA: el modelo no ve
nada propio de la parcela, solo su ubicacion y el clima que se le de.

El intervalo se ensancha con la distancia al vecino mas cercano y con la dispersion de las
predicciones de los vecinos. Es una regla practica, no un intervalo calibrado.
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np
import pandas as pd

from api.artefactos import Artefactos

K_VECINOS = 3
PISO_DIST_KM = 0.5
DIST_MEDIANA_KM = 10.0       # mas lejos que esto el vecino ya no representa a la parcela
DIST_LEJOS_KM = 5.0          # aviso: lejos de las parcelas conocidas
DIST_FUERA_KM = 15.0         # aviso: fuera de las zonas de entrenamiento
MARGEN_CAJA_GRADOS = 0.05    # unos 5 km alrededor de la caja de entrenamiento
FACTOR_DIST_MAX = 4.0
MESES = range(4, 11)
AREA_MIN, AREA_MAX = 0.1, 500.0


class ErrorEntrada(ValueError):
    """Entrada invalida; el mensaje se devuelve tal cual al usuario."""


def _sin_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").casefold().strip()


def _haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def grupo_de(variable: str) -> str:
    if re.match(r"^(prec_|tmax_|tmin_)|^meses_secos$", variable):
        return "Clima"
    if variable in ("elevacion_m", "pendiente_grados"):
        return "Topografía"
    if variable.startswith("ndvi_auc_"):
        return "Historia NDVI"
    if variable.startswith(("mun_", "est_")) or variable in ("lat", "lon", "AREA_HA"):
        return "Ubicación y parcela"
    return "Satélite"


def opciones(art: Artefactos) -> dict:
    """Lo que necesita el formulario: municipios conocidos, campos de clima que usa el modelo y limites."""
    cols = art.metadata["variables"]
    det = art.metadata["detalle_variables"]
    municipios = {}
    for p in art.parcelas.values():
        m = municipios.setdefault(p["municipio"], {"municipio": p["municipio"], "estado": p["estado"], "n": 0})
        m["n"] += 1
    clima = {"prec_mm": [], "tmax_c": [], "tmin_c": []}
    for c in cols:
        m = re.fullmatch(r"(prec_mm|tmax_c|tmin_c)_m(\d\d)", c)
        if m:
            clima[m.group(1)].append(int(m.group(2)))
    rangos = art.metadata["rangos_entrenamiento"]
    return {
        "municipios": sorted(municipios.values(), key=lambda x: x["municipio"]),
        "clima": clima,
        "rangos": {
            "lat": rangos["lat"], "lon": rangos["lon"], "area_ha": rangos["AREA_HA"],
            "prec_mm": {"min": min(rangos[f"prec_mm_m{m:02d}"]["min"] for m in clima["prec_mm"]),
                        "max": max(rangos[f"prec_mm_m{m:02d}"]["max"] for m in clima["prec_mm"])},
            "tmax_c": {"min": min(rangos[f"tmax_c_m{m:02d}"]["min"] for m in clima["tmax_c"]),
                       "max": max(rangos[f"tmax_c_m{m:02d}"]["max"] for m in clima["tmax_c"])},
            "tmin_c": {"min": min(rangos[f"tmin_c_m{m:02d}"]["min"] for m in clima["tmin_c"]),
                       "max": max(rangos[f"tmin_c_m{m:02d}"]["max"] for m in clima["tmin_c"])},
        },
        "limites": {"area_ha": [AREA_MIN, AREA_MAX]},
        "etiquetas": {c: det[c]["etiqueta"] for c in cols if c.startswith(("prec_mm_m", "tmax_c_m", "tmin_c_m"))},
    }


def _numero(valor, nombre, minimo=None, maximo=None) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
        raise ErrorEntrada(f"{nombre} debe ser un número.")
    if minimo is not None and valor < minimo or maximo is not None and valor > maximo:
        raise ErrorEntrada(f"{nombre} debe estar entre {minimo} y {maximo}.")
    return float(valor)


def _leer_clima(clima, cols: set[str]) -> dict[str, float]:
    """clima = {"prec_mm": {"5": 120, ...}, "tmax_c": {...}, "tmin_c": {...}} con meses 4 a 10; todo opcional."""
    dados: dict[str, float] = {}
    if clima in (None, {}):
        return dados
    if not isinstance(clima, dict):
        raise ErrorEntrada("clima debe ser un objeto con prec_mm, tmax_c y tmin_c por mes.")
    for campo, (lo, hi) in {"prec_mm": (0, 1000), "tmax_c": (-10, 50), "tmin_c": (-20, 40)}.items():
        valores = clima.get(campo)
        if valores in (None, {}):
            continue
        if not isinstance(valores, dict):
            raise ErrorEntrada(f"clima.{campo} debe ser un objeto con el mes (4 a 10) como llave.")
        for mes, v in valores.items():
            if v is None or v == "":
                continue
            try:
                n = int(mes)
            except (TypeError, ValueError):
                raise ErrorEntrada(f"clima.{campo}: el mes {mes!r} no es válido (4 a 10).") from None
            if n not in MESES:
                raise ErrorEntrada(f"clima.{campo}: el mes {n} está fuera del ciclo (4 a 10).")
            dados[f"{campo}_m{n:02d}"] = _numero(v, f"clima.{campo} del mes {n}", lo, hi)
    return dados


def predecir(art: Artefactos, cuerpo: dict) -> dict:
    meta = art.metadata
    cols: list[str] = meta["variables"]
    det = meta["detalle_variables"]
    if not isinstance(cuerpo, dict):
        raise ErrorEntrada("Se espera un objeto JSON.")

    # ---------- entrada ----------
    lat = _numero(cuerpo.get("lat"), "lat", -90, 90)
    lon = _numero(cuerpo.get("lon"), "lon", -180, 180)
    area = _numero(cuerpo.get("area_ha"), "area_ha", AREA_MIN, AREA_MAX)
    municipio_in = cuerpo.get("municipio")
    if not isinstance(municipio_in, str) or not municipio_in.strip():
        raise ErrorEntrada("municipio es obligatorio.")
    dados_clima = _leer_clima(cuerpo.get("clima"), set(cols))

    conocidos = {_sin_acentos(p["municipio"]): (p["municipio"], p["estado"]) for p in art.parcelas.values()}
    municipio, estado = conocidos.get(_sin_acentos(municipio_in), (None, None))
    avisos: list[dict] = [{
        "codigo": "estimacion_de_zona", "nivel": "info",
        "mensaje": "Es una estimación de zona: el modelo no conoce las imágenes de satélite ni la topografía de esta parcela, "
                   "solo su ubicación, su área y el clima que le des. El resto se toma de parcelas conocidas cercanas."}]
    if municipio is None:
        avisos.append({"codigo": "municipio_desconocido", "nivel": "alerta",
                       "mensaje": f"El municipio «{municipio_in.strip()}» no tiene parcelas de entrenamiento: el modelo no tiene su efecto de zona."})

    # ---------- vecinos ----------
    ids = list(art.features.index)
    pl = np.array([art.parcelas[i]["lat"] for i in ids])
    po = np.array([art.parcelas[i]["lon"] for i in ids])
    dist = _haversine_km(lat, lon, pl, po)
    mismo = np.array([art.parcelas[i]["municipio"] == municipio for i in ids]) if municipio else np.zeros(len(ids), bool)
    d_min_global = float(dist.min())
    pool = np.where(mismo)[0] if mismo.sum() >= K_VECINOS else np.arange(len(ids))
    orden = pool[np.argsort(dist[pool])][:K_VECINOS]
    d_vec = dist[orden]
    d_nn = float(d_vec[0])
    vecino_global = art.parcelas[ids[int(dist.argmin())]]

    usar_mediana = d_nn > DIST_MEDIANA_KM
    X = art.features
    if usar_mediana:
        if municipio:
            base = X[[art.parcelas[i]["municipio"] == municipio for i in ids]].median(numeric_only=True)
            origen_def, metodo = "mediana_municipio", "mediana_municipio"
        else:
            base = X.median(numeric_only=True)
            origen_def, metodo = "mediana_entrenamiento", "mediana_entrenamiento"
        fila = base.reindex(cols)
        sd_vec = 0.0
        pesos = np.array([])
    else:
        pesos = 1.0 / np.maximum(d_vec, PISO_DIST_KM) ** 2
        pesos = pesos / pesos.sum()
        M = X.iloc[orden][cols].to_numpy(dtype=float)
        valido = ~np.isnan(M)
        num = np.nansum(M * pesos[:, None], axis=0)
        den = (valido * pesos[:, None]).sum(axis=0)
        fila = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=cols)
        origen_def, metodo = "vecinos", "vecinos_ponderados"
        pred_vec = np.array([art.parcelas[ids[i]]["rendimiento_pred"] for i in orden])
        sd_vec = float(np.sqrt(np.sum(pesos * (pred_vec - np.sum(pesos * pred_vec)) ** 2)))

    # ---------- lo que el usuario da manda ----------
    origen: dict[str, str] = {c: origen_def for c in cols}
    proporcionadas: set[str] = set()

    def poner(var, valor, de):
        if var in fila.index:
            fila[var] = valor
            origen[var] = de
            proporcionadas.add(var)

    poner("lat", lat, "usuario")
    poner("lon", lon, "usuario")
    poner("AREA_HA", area, "usuario")
    for c in cols:
        if c.startswith("mun_"):
            poner(c, 1.0 if municipio and c == f"mun_{municipio}" else 0.0, "usuario")
        elif c.startswith("est_"):
            poner(c, 1.0 if estado and c == f"est_{estado}" else 0.0, "usuario")
    ignoradas = []
    for var, v in dados_clima.items():
        if var in fila.index:
            poner(var, v, "usuario")
        else:
            ignoradas.append(var)
    if ignoradas:
        avisos.append({"codigo": "clima_sin_uso", "nivel": "info",
                       "mensaje": "Estos datos de clima no entran al modelo y se ignoraron: " + ", ".join(sorted(ignoradas)) + "."})

    # derivadas del clima: se recalculan solo si el usuario dio lluvia
    if any(v.startswith("prec_mm_m") for v in dados_clima):
        prec = np.array([fila[f"prec_mm_m{m:02d}"] for m in MESES])
        if "meses_secos" in fila.index:
            fila["meses_secos"] = float((prec < 50).sum())
            origen["meses_secos"] = "calculada"
        if "prec_anomalia" in fila.index and "prec_ciclo_hist" in fila.index:
            fila["prec_anomalia"] = float(prec.sum() - fila["prec_ciclo_hist"])
            origen["prec_anomalia"] = "calculada"

    # ---------- prediccion ----------
    pred = float(art.modelo.predecir(fila.to_frame().T)[0])
    pred = float(np.clip(pred, 1.0, 6.5))

    # ---------- intervalo ----------
    q = art.semiancho
    factor_dist = 1 + 0.25 * min(d_nn / DIST_LEJOS_KM, FACTOR_DIST_MAX) if not usar_mediana else 1 + 0.25 * FACTOR_DIST_MAX
    semiancho = math.sqrt(q ** 2 + sd_vec ** 2) * factor_dist

    # ---------- avisos de ubicacion y de rango ----------
    z = meta["zona"]
    fuera_caja = not (z["lat_min"] - MARGEN_CAJA_GRADOS <= lat <= z["lat_max"] + MARGEN_CAJA_GRADOS
                      and z["lon_min"] - MARGEN_CAJA_GRADOS <= lon <= z["lon_max"] + MARGEN_CAJA_GRADOS)
    if fuera_caja or d_min_global > DIST_FUERA_KM:
        avisos.append({"codigo": "fuera_de_zona", "nivel": "alerta",
                       "mensaje": f"La ubicación queda fuera de las zonas de entrenamiento (la parcela conocida más cercana está a {d_min_global:.1f} km). "
                                  "El modelo de longitud solo funciona en el altiplano entre Calpulalpan y Chignahuapan; aquí la predicción no es confiable."})
    elif d_min_global > DIST_LEJOS_KM:
        avisos.append({"codigo": "lejos_de_parcelas", "nivel": "aviso",
                       "mensaje": f"La parcela conocida más cercana está a {d_min_global:.1f} km: la estimación es menos confiable y el intervalo es más ancho."})
    if usar_mediana:
        avisos.append({"codigo": "sin_vecinos_cercanos", "nivel": "aviso",
                       "mensaje": f"No hay parcelas a menos de {DIST_MEDIANA_KM:.0f} km en {'el municipio' if municipio else 'la zona'}; "
                                  "lo que falta se tomó de la mediana " + ("del municipio." if municipio else "de entrenamiento.")})
    if municipio and vecino_global["municipio"] != municipio and not fuera_caja:
        avisos.append({"codigo": "municipio_no_coincide", "nivel": "aviso",
                       "mensaje": f"La parcela conocida más cercana es de {vecino_global['municipio']}, no de {municipio}. Revisa lat, lon y municipio."})

    rangos = meta["rangos_entrenamiento"]
    for var in sorted(proporcionadas):
        if var.startswith(("mun_", "est_")) or var in ("lat", "lon"):
            continue
        r = rangos[var]
        if not r["min"] <= fila[var] <= r["max"]:
            avisos.append({"codigo": "fuera_de_rango", "nivel": "aviso",
                           "mensaje": f"{det[var]['etiqueta']} de {fila[var]:.1f} {det[var]['unidad']} queda fuera del rango visto al entrenar "
                                      f"({r['min']:.1f} a {r['max']:.1f}): toma el resultado con más cautela."})

    # ---------- respuesta ----------
    imputadas = []
    for c in cols:
        if c in proporcionadas or origen[c] == "calculada":
            continue
        v = fila[c]
        o = origen[c]
        if pd.isna(v):
            v, o = float(art.modelo.imputador.statistics_[cols.index(c)]), "mediana_entrenamiento"
        valor = float(v) / det[c]["divisor"]
        imputadas.append({"variable": c, "etiqueta": det[c]["etiqueta"], "grupo": grupo_de(c), "valor": round(valor, 5),
                          "unidad": det[c]["unidad"], "origen": o})
    resumen: dict[str, int] = {}
    for x in imputadas:
        resumen[x["grupo"]] = resumen.get(x["grupo"], 0) + 1
    calculadas = [c for c in cols if origen[c] == "calculada"]

    return {
        "prediccion": {"pred": round(pred, 2), "inf": round(pred - semiancho, 2), "sup": round(pred + semiancho, 2)},
        "entradas": {"lat": lat, "lon": lon, "area_ha": area, "municipio": municipio or municipio_in.strip(),
                     "estado": estado, "clima": {k: v for k, v in sorted(dados_clima.items())}},
        "intervalo": {"semiancho": round(semiancho, 3), "semiancho_base": round(q, 3),
                      "factor_distancia": round(factor_dist, 2), "dispersion_vecinos": round(sd_vec, 3),
                      "nota": "Regla práctica: el intervalo base de las demás parcelas, ensanchado por la distancia al vecino más cercano y por la dispersión entre vecinos. No está calibrado."},
        "imputacion": {
            "metodo": metodo,
            "n_variables": len(cols),
            # el municipio ocupa varias columnas (one-hot de municipio y de estado)
            "n_proporcionadas": len(proporcionadas),
            "datos_usuario": ["Latitud", "Longitud", "Área", "Municipio"] + [det[v]["etiqueta"] for v in sorted(dados_clima) if v in det],
            "n_calculadas": len(calculadas),
            "n_imputadas": len(imputadas),
            "por_grupo": resumen,
            "vecinos": [{"id": ids[i], "municipio": art.parcelas[ids[i]]["municipio"], "distancia_km": round(float(dist[i]), 2),
                         "peso": round(float(w), 3), "rendimiento": art.parcelas[ids[i]]["rendimiento_pred"]}
                        for i, w in zip(orden, pesos)] if not usar_mediana else [],
            "variables_imputadas": imputadas,
        },
        "avisos": avisos,
    }
