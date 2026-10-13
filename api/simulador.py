"""Simulador de escenarios: re-predice una parcela con otra lluvia de mayo y otra Tmin de abril.

Usa el modelo ajustado que guarda el paso 07 (salidas/modelos/modelo_final.joblib).
La prediccion del escenario es la prediccion mostrada de la parcela mas el cambio
que el modelo calcula al mover las dos variables; asi la base del simulador
coincide siempre con lo que se ve en el mapa y en la ficha.

El intervalo del escenario se ensancha con el tamano del cambio, porque entre mas
se aleja el escenario de lo observado, menos se puede confiar en el modelo.
"""
from functools import lru_cache

import joblib
import numpy as np

from pipeline import conf
from pipeline.modelado import cargar_datos

MESES_CICLO = range(4, 11)
DIAS_MES = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31}
LIM_LLUVIA_PCT, LIM_TMIN_C = 40.0, 2.0


class ModeloNoDisponible(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _artefactos():
    ruta = conf.RUTA_MODELOS / "modelo_final.joblib"
    if not ruta.exists():
        raise ModeloNoDisponible("Falta salidas/modelos/modelo_final.joblib: corre el paso 07 del pipeline.")
    art = joblib.load(ruta)
    datos = cargar_datos()
    X = datos.X_todas.copy()
    X.index = datos.ids_todas
    return art, X


def limpiar_cache():
    _artefactos.cache_clear()


def _recalcular_derivadas(fila):
    """Las variables del ciclo que dependen de los meses editados se vuelven a calcular."""
    prec = np.array([fila[f"prec_mm_m{m:02d}"] for m in MESES_CICLO])
    tmax = np.array([fila[f"tmax_c_m{m:02d}"] for m in MESES_CICLO])
    tmin = np.array([fila[f"tmin_c_m{m:02d}"] for m in MESES_CICLO])
    dias = np.array([DIAS_MES[m] for m in MESES_CICLO])
    nuevos = {
        "prec_ciclo": prec.sum(),
        "tmax_ciclo": tmax.mean(),
        "tmin_ciclo": tmin.mean(),
        "gdd_ciclo": (np.clip((tmax + tmin) / 2, 0, None) * dias).sum(),
        "meses_secos": float((prec < 50).sum()),
    }
    nuevos["amplitud_termica"] = nuevos["tmax_ciclo"] - nuevos["tmin_ciclo"]
    if "prec_ciclo_hist" in fila.index:
        nuevos["prec_anomalia"] = nuevos["prec_ciclo"] - fila["prec_ciclo_hist"]
    for k, v in nuevos.items():
        if k in fila.index:
            fila[k] = v
    return fila


def simular(id_poligono: str, base: dict, delta_lluvia_pct: float, delta_tmin_c: float,
            rango: dict, modelo: str | None = None) -> dict:
    """base = {"pred", "inf", "sup"} de la parcela; devuelve el escenario completo."""
    art, X = _artefactos()
    if id_poligono not in X.index:
        raise KeyError(id_poligono)
    modelo = modelo or art["modelo_final"]
    ajustado = art["modelo"]

    dl = float(np.clip(delta_lluvia_pct or 0, -LIM_LLUVIA_PCT, LIM_LLUVIA_PCT))
    dt = float(np.clip(delta_tmin_c or 0, -LIM_TMIN_C, LIM_TMIN_C))

    fila = X.loc[[id_poligono]].copy()
    lluvia_base = float(fila["prec_mm_m05"].iloc[0])
    tmin_base = float(fila["tmin_c_m04"].iloc[0])
    antes = float(ajustado.predecir(fila)[modelo][0])

    esc = fila.iloc[0].copy()
    esc["prec_mm_m05"] = lluvia_base * (1 + dl / 100)
    esc["tmin_c_m04"] = tmin_base + dt
    esc = _recalcular_derivadas(esc)
    despues = float(ajustado.predecir(esc.to_frame().T.astype(float))[modelo][0])

    pred_esc = float(np.clip(base["pred"] + (despues - antes), 1.0, 6.5))
    hw_base = (base["sup"] - base["inf"]) / 2
    hw_esc = hw_base * (1 + 0.18 * abs(dl) / LIM_LLUVIA_PCT + 0.12 * abs(dt) / LIM_TMIN_C)

    lluvia_esc, tmin_esc = round(lluvia_base * (1 + dl / 100)), round(tmin_base + dt, 1)
    fuera = []
    rl, rt = rango["lluvia_mayo"], rango["tmin_abril"]
    if not rl["min"] <= lluvia_esc <= rl["max"]:
        fuera.append({"variable": "lluvia_mayo", "etiqueta": "Lluvia de mayo", "valor": lluvia_esc,
                      "min": round(rl["min"]), "max": round(rl["max"]), "unidad": "mm"})
    if not rt["min"] <= tmin_esc <= rt["max"]:
        fuera.append({"variable": "tmin_abril", "etiqueta": "Temperatura mínima de abril", "valor": tmin_esc,
                      "min": round(rt["min"], 1), "max": round(rt["max"], 1), "unidad": "°C"})

    return {
        "id": id_poligono,
        "modelo": modelo,
        "base": {"pred": round(base["pred"], 2), "inf": round(base["inf"], 2), "sup": round(base["sup"], 2)},
        "escenario": {"pred": round(pred_esc, 2), "inf": round(pred_esc - hw_esc, 2), "sup": round(pred_esc + hw_esc, 2)},
        "diferencia": round(pred_esc - base["pred"], 2),
        "diferencia_pct": round((pred_esc / base["pred"] - 1) * 100, 1),
        "entradas": {"lluvia_mayo_base": round(lluvia_base), "lluvia_mayo_escenario": lluvia_esc,
                     "tmin_abril_base": round(tmin_base, 1), "tmin_abril_escenario": tmin_esc},
        "fuera_de_rango": fuera,
        "rango_entrenamiento": {
            "lluvia_mayo": {"min": round(rl["min"]), "max": round(rl["max"])},
            "tmin_abril": {"min": round(rt["min"], 1), "max": round(rt["max"], 1)},
        },
    }
