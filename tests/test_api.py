import pytest

CAMPOS = {"id", "estado", "municipio", "area_ha", "conjunto", "rendimiento_obs", "rendimiento_pred",
          "intervalo_inf", "intervalo_sup", "lat", "lon", "elevacion_m", "pendiente_grados"}


def test_salud(cliente):
    r = cliente.get("/api/salud").json()
    assert r["estado"] == "ok" and r["modelo"] == "mezcla" and r["parcelas"] == 197


def test_parcelas_contrato(cliente):
    datos = cliente.get("/api/parcelas").json()
    assert len(datos) == 197 and set(datos[0]) == CAMPOS
    assert all(x["rendimiento_obs"] is None for x in datos if x["conjunto"] == "PREDICCION")
    assert all(x["intervalo_inf"] < x["rendimiento_pred"] < x["intervalo_sup"] for x in datos)
    # un solo modelo: ya no hay parametro ?modelo
    assert cliente.get("/api/parcelas?modelo=ensamble").json() == datos


def test_detalle_sin_enfoques(cliente):
    d = cliente.get("/api/parcelas/AGC_003").json()
    assert set(d) == {"parcela", "serie", "clima", "shap", "shap_otras", "base", "vecinos"}
    assert [c["mes"] for c in d["clima"]] == list(range(4, 11))
    assert len(d["shap"]) == 8 and len(d["vecinos"]) == 5 and d["serie"]
    total = d["base"] + sum(s["contribucion"] for s in d["shap"]) + d["shap_otras"]
    assert total == pytest.approx(d["parcela"]["rendimiento_pred"], abs=0.02)
    assert [v["distancia_km"] for v in d["vecinos"]] == sorted(v["distancia_km"] for v in d["vecinos"])
    assert cliente.get("/api/parcelas/NOPE").status_code == 404


def test_metricas_solo_de_la_mezcla(cliente):
    m = cliente.get("/api/modelo/metricas").json()
    assert m["modelo"]["nombre"] == "Mezcla"
    assert m["validacion"]["rmse"] < m["validacion"]["baseline"]["rmse"]
    assert m["lomo"]["rmse"] > m["validacion"]["rmse"] and "baseline" in m["lomo"]
    assert len(m["dispersion"]) == len(m["residuales"]) == 138
    # la comparacion A/B/C viaja solo para la tabla estatica de la metodologia
    assert {c["id"] for c in m["comparacion"]} == {"ensamble", "ubicacion", "mezcla"}
    for clave in ("candidatos", "modelos", "modelo_final"):
        assert clave not in m


def test_no_queda_rastro_de_enfoques_en_las_respuestas(cliente):
    for ruta in ("/api/parcelas", "/api/parcelas/AGC_003"):
        assert "otros_enfoques" not in cliente.get(ruta).content.decode()
