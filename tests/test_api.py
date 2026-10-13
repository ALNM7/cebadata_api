import json

import pytest

from pipeline import conf

pytestmark = pytest.mark.django_db


def test_salud(cliente):
    r = cliente.get("/api/salud").json()
    assert r["resultados_cargados"] and r["parcelas"] == conf.TOTAL_PARCELAS


def test_parcelas_contrato(cliente):
    datos = cliente.get("/api/parcelas").json()
    assert len(datos) == 197
    p = datos[0]
    assert set(p) == {"id", "estado", "municipio", "area_ha", "conjunto", "rendimiento_obs", "rendimiento_pred",
                      "intervalo_inf", "intervalo_sup", "lat", "lon", "elevacion_m", "pendiente_grados"}
    assert sum(x["conjunto"] == "PREDICCION" for x in datos) == conf.N_PREDICCION
    assert all(x["rendimiento_obs"] is None for x in datos if x["conjunto"] == "PREDICCION")
    assert all(x["intervalo_inf"] < x["rendimiento_pred"] < x["intervalo_sup"] for x in datos)


def test_parcelas_por_modelo_difieren(cliente):
    mezcla = {p["id"]: p["rendimiento_pred"] for p in cliente.get("/api/parcelas?modelo=mezcla").json()}
    ubic = {p["id"]: p["rendimiento_pred"] for p in cliente.get("/api/parcelas?modelo=ubicacion").json()}
    ens = {p["id"]: p["rendimiento_pred"] for p in cliente.get("/api/parcelas?modelo=ensamble").json()}
    # la mezcla es el promedio de los otros dos (con redondeo a 2 decimales)
    assert all(abs(mezcla[i] - (ubic[i] + ens[i]) / 2) <= 0.011 for i in mezcla)
    assert cliente.get("/api/parcelas?modelo=otro").status_code == 400


def test_detalle(cliente):
    d = cliente.get("/api/parcelas/AGC_003").json()
    assert {"parcela", "serie", "clima", "shap", "shap_otras", "base", "vecinos", "otros_enfoques"} <= set(d)
    assert len(d["clima"]) == 7 and [c["mes"] for c in d["clima"]] == list(range(4, 11))
    assert len(d["shap"]) == 8 and len(d["vecinos"]) == 5
    # la cascada cierra: base + contribuciones + otras = prediccion
    total = d["base"] + sum(s["contribucion"] for s in d["shap"]) + d["shap_otras"]
    assert total == pytest.approx(d["parcela"]["rendimiento_pred"], abs=0.02)
    assert [v["distancia_km"] for v in d["vecinos"]] == sorted(v["distancia_km"] for v in d["vecinos"])
    assert cliente.get("/api/parcelas/NOPE").status_code == 404


def test_metricas(cliente):
    m = cliente.get("/api/modelo/metricas").json()
    assert m["modelo_final"] == conf.MODELO_FINAL
    assert {c["id"] for c in m["candidatos"]} == set(conf.CANDIDATOS)
    for c in m["candidatos"]:
        assert c["estratificada"]["rmse"] > 0 and c["lomo"]["rmse"] > 0
    assert sum(x.get("elegido", False) for x in m["modelos"]) == 1
    assert len(m["dispersion"]) == len(m["residuales"]) == conf.N_ENTRENAMIENTO
    assert m["validacion"]["rmse"] < m["validacion"]["baseline"]["rmse"]
    assert {"indices", "dentro_ndvi", "clima", "bloques"} <= set(m["hallazgos"])


def _simular(cliente, **cuerpo):
    r = cliente.post("/api/simulador", data=json.dumps(cuerpo), content_type="application/json")
    return r, r.json()


def test_simulador_sin_cambios_no_mueve_la_prediccion(cliente):
    r, s = _simular(cliente, id="AGC_003", delta_lluvia_mayo_pct=0, delta_tmin_abril_c=0)
    assert r.status_code == 200
    assert s["diferencia"] == 0 and s["escenario"]["pred"] == s["base"]["pred"]
    assert s["fuera_de_rango"] == []


def test_simulador_aviso_fuera_de_rango_y_limites(cliente):
    _, s = _simular(cliente, id="AGC_003", delta_lluvia_mayo_pct=500, delta_tmin_abril_c=9)
    assert s["entradas"]["tmin_abril_escenario"] == pytest.approx(s["entradas"]["tmin_abril_base"] + 2, abs=0.11)
    assert s["entradas"]["lluvia_mayo_escenario"] == pytest.approx(s["entradas"]["lluvia_mayo_base"] * 1.4, abs=1)
    assert any(f["variable"] == "tmin_abril" for f in s["fuera_de_rango"])
    ancho_base = s["base"]["sup"] - s["base"]["inf"]
    assert s["escenario"]["sup"] - s["escenario"]["inf"] > ancho_base


def test_simulador_errores(cliente):
    assert _simular(cliente, id="NOPE")[0].status_code == 404
    r = cliente.post("/api/simulador", data="no es json", content_type="application/json")
    assert r.status_code == 400
    assert cliente.get("/api/simulador").status_code == 405
