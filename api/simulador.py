"""Simulador de escenarios sobre una parcela existente.

Cambia la lluvia de mayo (en %) y la temperatura minima de abril (en grados C) en la fila de
variables de la parcela, recalcula las variables del ciclo que dependen de esos meses
(meses secos y anomalia de lluvia) y vuelve a predecir con la mezcla. La prediccion del
escenario es la prediccion mostrada de la parcela mas el cambio que calcula el modelo; asi
la base siempre coincide con lo que se ve en el mapa y en la ficha.

El intervalo del escenario se ensancha con el tamano del cambio: entre mas se aleja de lo
observado, menos se puede confiar en el modelo.
"""
from __future__ import annotations

import numpy as np

from api.artefactos import Artefactos

MESES_CICLO = range(4, 11)
LIM_LLUVIA_PCT = 40.0
LIM_TMIN_C = 2.0
UMBRAL_MES_SECO_MM = 50.0

LLUVIA = "prec_mm_m05"
TMIN = "tmin_c_m04"


def aplicar_escenario(fila, delta_lluvia_pct: float, delta_tmin_c: float):
    """Devuelve una copia de la fila (Series) con el escenario aplicado y las derivadas al dia."""
    f = fila.copy()
    prec_antes = np.array([f[f"prec_mm_m{m:02d}"] for m in MESES_CICLO])
    f[LLUVIA] = f[LLUVIA] * (1 + delta_lluvia_pct / 100)
    f[TMIN] = f[TMIN] + delta_tmin_c
    prec = np.array([f[f"prec_mm_m{m:02d}"] for m in MESES_CICLO])
    if "meses_secos" in f.index:
        f["meses_secos"] = float((prec < UMBRAL_MES_SECO_MM).sum())
    if "prec_anomalia" in f.index:
        f["prec_anomalia"] = f["prec_anomalia"] + (prec.sum() - prec_antes.sum())
    return f


def fuera_de_rango(art: Artefactos, original, escenario) -> list[dict]:
    """Variables que el escenario movio y que quedaron fuera del rango visto al entrenar."""
    det = art.metadata["detalle_variables"]
    out = []
    for var in (LLUVIA, TMIN, "prec_anomalia", "meses_secos"):
        if var not in escenario.index or escenario[var] == original[var]:
            continue
        r = art.metadata["rangos_entrenamiento"][var]
        v = float(escenario[var])
        if not r["min"] <= v <= r["max"]:
            d = 1 if det[var]["unidad"] == "°C" else 0
            out.append({"variable": {LLUVIA: "lluvia_mayo", TMIN: "tmin_abril"}.get(var, var),
                        "etiqueta": det[var]["etiqueta"], "valor": round(v, d) if d else round(v),
                        "min": round(r["min"], d) if d else round(r["min"]),
                        "max": round(r["max"], d) if d else round(r["max"]), "unidad": det[var]["unidad"]})
    return out


def simular(art: Artefactos, id_poligono: str, delta_lluvia_pct: float, delta_tmin_c: float) -> dict:
    p = art.parcelas[id_poligono]
    base = {"pred": p["rendimiento_pred"], "inf": p["intervalo_inf"], "sup": p["intervalo_sup"]}

    dl = float(np.clip(delta_lluvia_pct or 0, -LIM_LLUVIA_PCT, LIM_LLUVIA_PCT))
    dt = float(np.clip(delta_tmin_c or 0, -LIM_TMIN_C, LIM_TMIN_C))

    fila = art.features.loc[id_poligono]
    esc = aplicar_escenario(fila, dl, dt)
    pares = art.features.loc[[id_poligono]].copy()
    pares.loc["escenario"] = esc
    antes, despues = art.modelo.predecir(pares)
    cambio = float(despues - antes)

    pred_esc = float(np.clip(base["pred"] + cambio, 1.0, 6.5))
    hw_base = (base["sup"] - base["inf"]) / 2
    hw_esc = hw_base * (1 + 0.18 * abs(dl) / LIM_LLUVIA_PCT + 0.12 * abs(dt) / LIM_TMIN_C)

    r = art.metadata["rangos_entrenamiento"]
    return {
        "id": id_poligono,
        "base": {k: round(v, 2) for k, v in base.items()},
        "escenario": {"pred": round(pred_esc, 2), "inf": round(pred_esc - hw_esc, 2), "sup": round(pred_esc + hw_esc, 2)},
        "diferencia": round(pred_esc - base["pred"], 2),
        "diferencia_pct": round((pred_esc / base["pred"] - 1) * 100, 1),
        "entradas": {
            "lluvia_mayo_base": round(float(fila[LLUVIA])), "lluvia_mayo_escenario": round(float(esc[LLUVIA])),
            "tmin_abril_base": round(float(fila[TMIN]), 1), "tmin_abril_escenario": round(float(esc[TMIN]), 1),
        },
        "fuera_de_rango": fuera_de_rango(art, fila, esc),
        "rango_entrenamiento": {
            "lluvia_mayo": {"min": round(r[LLUVIA]["min"]), "max": round(r[LLUVIA]["max"])},
            "tmin_abril": {"min": round(r[TMIN]["min"], 1), "max": round(r[TMIN]["max"], 1)},
        },
    }
