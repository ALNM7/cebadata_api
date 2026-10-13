import json

import pytest


def _pred(cliente, **cuerpo):
    r = cliente.post("/api/predecir", data=json.dumps(cuerpo), content_type="application/json")
    return r, r.json()


@pytest.fixture
def parcela(art):
    return art.parcelas["AGC_003"]


def _base(p, **extra):
    return {"lat": p["lat"], "lon": p["lon"], "municipio": p["municipio"], "area_ha": p["area_ha"], **extra}


def test_opciones(cliente):
    o = cliente.get("/api/nueva-parcela/opciones").json()
    assert {m["municipio"] for m in o["municipios"]} >= {"Singuilucan", "Chignahuapan", "Cuautepec de Hinojosa"}
    assert o["clima"]["prec_mm"] == list(range(4, 11)) and o["clima"]["tmin_c"] == [4, 6]
    assert o["rangos"]["prec_mm"]["min"] < o["rangos"]["prec_mm"]["max"]


def test_en_el_mismo_lugar_de_una_parcela_conocida_da_lo_mismo(cliente, parcela):
    r, d = _pred(cliente, **_base(parcela))
    assert r.status_code == 200
    assert d["prediccion"]["pred"] == pytest.approx(parcela["rendimiento_pred"], abs=0.25)
    im = d["imputacion"]
    assert im["metodo"] == "vecinos_ponderados" and len(im["vecinos"]) == 3
    assert sum(v["peso"] for v in im["vecinos"]) == pytest.approx(1.0, abs=0.01)
    assert im["n_proporcionadas"] + im["n_calculadas"] + im["n_imputadas"] == im["n_variables"] == 121
    assert {a["codigo"] for a in d["avisos"]} == {"estimacion_de_zona"}
    assert im["datos_usuario"][:4] == ["Latitud", "Longitud", "Área", "Municipio"]


def test_dice_que_variables_se_imputaron(cliente, parcela):
    d = _pred(cliente, **_base(parcela))[1]
    v = {x["variable"]: x for x in d["imputacion"]["variables_imputadas"]}
    assert "ndvi_pico" in v and v["ndvi_pico"]["origen"] == "vecinos" and v["ndvi_pico"]["grupo"] == "Satélite"
    assert "lon" not in v and "AREA_HA" not in v and "mun_Singuilucan" not in v
    assert "prec_mm_m05" in v          # sin clima del usuario, la lluvia tambien se imputa
    assert d["imputacion"]["por_grupo"]["Satélite"] > 50


def test_el_clima_del_usuario_manda_y_recalcula_derivadas(cliente, parcela):
    sin = _pred(cliente, **_base(parcela))[1]
    seco = _pred(cliente, **_base(parcela, clima={"prec_mm": {str(m): 30 for m in range(4, 11)}}))[1]
    v = {x["variable"] for x in seco["imputacion"]["variables_imputadas"]}
    assert not any(x.startswith("prec_mm_m") for x in v)
    assert "meses_secos" not in v and "prec_anomalia" not in v       # se calcularon con la lluvia dada
    assert seco["imputacion"]["n_calculadas"] == 2
    assert seco["prediccion"]["pred"] != sin["prediccion"]["pred"]
    assert any(a["codigo"] == "fuera_de_rango" for a in seco["avisos"])   # 30 mm todos los meses no se vio al entrenar


def test_fuera_de_zona_avisa_y_ensancha_el_intervalo(cliente, parcela):
    dentro = _pred(cliente, **_base(parcela))[1]
    lejos = _pred(cliente, lat=19.43, lon=-99.13, municipio="Coyoacán", area_ha=3)[1]
    codigos = {a["codigo"] for a in lejos["avisos"]}
    assert {"fuera_de_zona", "municipio_desconocido", "sin_vecinos_cercanos"} <= codigos
    assert lejos["imputacion"]["metodo"] == "mediana_entrenamiento"
    assert lejos["intervalo"]["semiancho"] > 1.5 * dentro["intervalo"]["semiancho"]


def test_municipio_que_no_coincide_con_la_ubicacion(cliente, parcela):
    d = _pred(cliente, **_base(parcela, municipio="Calpulalpan"))[1]
    assert any(a["codigo"] == "municipio_no_coincide" for a in d["avisos"])


def test_acepta_el_municipio_sin_acentos_ni_mayusculas(cliente, parcela):
    d = _pred(cliente, **_base(parcela, municipio="CUAUTEPEC DE HINOJOSA"))[1]
    assert d["entradas"]["municipio"] == "Cuautepec de Hinojosa" and d["entradas"]["estado"] == "Hidalgo"


@pytest.mark.parametrize("cambio", [
    {"lat": "x"}, {"lat": 95}, {"lon": None}, {"area_ha": 0}, {"area_ha": -3}, {"municipio": ""}, {"municipio": 5},
    {"clima": {"prec_mm": {"3": 10}}}, {"clima": {"prec_mm": {"5": -4}}}, {"clima": {"prec_mm": [1, 2]}}, {"clima": "mucho"},
])
def test_entradas_invalidas(cliente, parcela, cambio):
    r, d = _pred(cliente, **{**_base(parcela), **cambio})
    assert r.status_code == 400 and d["error"]


def test_el_cuerpo_tiene_que_ser_un_objeto(cliente):
    assert cliente.post("/api/predecir", data="[1]", content_type="application/json").status_code == 400
    assert cliente.get("/api/predecir").status_code == 405
