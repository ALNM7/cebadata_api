"""Nombres legibles y unidades de las variables del modelo, para el dashboard."""
import re

MESES = {4: "abril", 5: "mayo", 6: "junio", 7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre"}

# Etiqueta corta para las graficas de hallazgos
CORTAS = {
    "prec_mm_m05": "Lluvia de mayo", "prec_mm_m04": "Lluvia de abril", "prec_mm_m06": "Lluvia de junio",
    "prec_mm_m07": "Lluvia de julio", "prec_mm_m08": "Lluvia de agosto", "prec_mm_m09": "Lluvia de septiembre",
    "prec_mm_m10": "Lluvia de octubre",
    "tmin_c_m04": "Temp. mínima de abril", "tmax_c_m04": "Temp. máxima de abril",
}

FAMILIAS = {
    "clima": "Clima", "ubicacion": "Ubicación", "topografia": "Topografía", "area": "Área",
    "suelo": "Suelo", "historia_ndvi": "NDVI histórico", "otra": "Otras",
}

BLOQUES = {
    "solo_clima": "Solo clima", "todo": "Todas las variables", "clima_mas_satelital": "Clima + satelital",
    "solo_topografia_ubicacion": "Topografía + ubicación", "clima_mas_ubicacion": "Clima + ubicación",
    "solo_satelital": "Solo satelital", "solo_suelo": "Solo suelo", "todo_sin_suelo": "Todas sin suelo",
}

_MEDIDAS = {
    "pico": "{i} máximo", "doy_pico": "Día del pico de {i}", "auc": "Integral de {i}",
    "tasa_verdeo": "Velocidad de verdeo {i}", "tasa_senescencia": "Velocidad de senescencia {i}",
}


def familia_nombre(familia: str) -> str:
    if familia.startswith("satelital:"):
        return familia.split(":", 1)[1].upper()
    return FAMILIAS.get(familia, familia)


def etiqueta(variable: str) -> str:
    m = re.fullmatch(r"(prec_mm|tmax_c|tmin_c)_m(\d\d)", variable)
    if m:
        mes = MESES.get(int(m.group(2)), m.group(2))
        base = {"prec_mm": "Lluvia de", "tmax_c": "Temperatura máxima de", "tmin_c": "Temperatura mínima de"}[m.group(1)]
        return f"{base} {mes}"
    fijas = {
        "lon": "Longitud", "lat": "Latitud", "elevacion_m": "Elevación", "pendiente_grados": "Pendiente del terreno",
        "AREA_HA": "Área de la parcela", "meses_secos": "Meses secos del ciclo",
        "prec_ciclo_hist": "Lluvia media de ciclos anteriores", "prec_anomalia": "Anomalía de lluvia del ciclo",
        "prec_ciclo": "Lluvia total del ciclo", "tmax_ciclo": "Temperatura máxima media del ciclo",
        "tmin_ciclo": "Temperatura mínima media del ciclo", "gdd_ciclo": "Grados-día del ciclo",
        "amplitud_termica": "Amplitud térmica del ciclo", "ndvi_std_pico": "Variación interna del NDVI en el pico",
        "ndvi_std_pico_pl": "Variación interna del NDVI en el pico (Planet)",
        "ndvi_auc_hist_media": "NDVI acumulado, promedio histórico",
    }
    if variable in fijas:
        return fijas[variable]
    m = re.fullmatch(r"ndvi_auc_(\d{4})", variable)
    if m:
        return f"NDVI acumulado {m.group(1)}"
    if variable.startswith("mun_"):
        return f"Municipio {variable[4:]}"
    if variable.startswith("est_"):
        return f"Estado {variable[4:]}"
    if variable.startswith("suelo_"):
        return {"suelo_phh2o": "pH del suelo", "suelo_soc": "Carbono orgánico del suelo",
                "suelo_clay": "Arcilla del suelo", "suelo_sand": "Arena del suelo",
                "suelo_cec": "Capacidad de intercambio catiónico"}.get(variable, variable)
    m = re.fullmatch(r"([a-z0-9]+)_(doy_pico|pico|auc|tasa_verdeo|tasa_senescencia)(_pl)?", variable)
    if m:
        texto = _MEDIDAS[m.group(2)].format(i=m.group(1).upper())
        return texto + (" (Planet)" if m.group(3) else "")
    return variable


def valor_y_unidad(variable: str, valor: float) -> tuple[float, str]:
    """Valor en unidades fisicas y su unidad. Las tasas del paso 05 vienen por 100 dias."""
    if re.fullmatch(r"prec_mm_m\d\d|prec_ciclo|prec_ciclo_hist|prec_anomalia", variable):
        return valor, "mm"
    if re.fullmatch(r"(tmax_c|tmin_c)_m\d\d|tmax_ciclo|tmin_ciclo|amplitud_termica", variable):
        return valor, "°C"
    if variable in ("lon", "lat", "pendiente_grados"):
        return valor, "°"
    if variable == "elevacion_m":
        return valor, "m"
    if variable == "AREA_HA":
        return valor, "ha"
    if variable.endswith(("tasa_verdeo", "tasa_senescencia", "tasa_verdeo_pl", "tasa_senescencia_pl")):
        return valor / 100.0, "por día"
    if variable.startswith(("mun_", "est_")):
        return valor, "sí/no"
    return valor, ""
