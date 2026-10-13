import json

import pytest

from api import simulador


def _sim(cliente, **cuerpo):
    r = cliente.post("/api/simulador", data=json.dumps(cuerpo), content_type="application/json")
    return r, r.json()


def test_sin_cambios_no_mueve_la_prediccion(cliente):
    r, s = _sim(cliente, id="AGC_003", delta_lluvia_mayo_pct=0, delta_tmin_abril_c=0)
    assert r.status_code == 200
    assert s["diferencia"] == 0 and s["escenario"]["pred"] == s["base"]["pred"] and s["fuera_de_rango"] == []


def test_la_base_es_la_prediccion_mostrada(cliente):
    s = _sim(cliente, id="AGC_003")[1]
    p = next(x for x in cliente.get("/api/parcelas").json() if x["id"] == "AGC_003")
    assert s["base"] == {"pred": p["rendimiento_pred"], "inf": p["intervalo_inf"], "sup": p["intervalo_sup"]}


def test_limites_y_aviso_fuera_de_rango(cliente):
    s = _sim(cliente, id="AGC_003", delta_lluvia_mayo_pct=500, delta_tmin_abril_c=9)[1]
    assert s["entradas"]["tmin_abril_escenario"] == pytest.approx(s["entradas"]["tmin_abril_base"] + 2, abs=0.11)
    assert s["entradas"]["lluvia_mayo_escenario"] == pytest.approx(s["entradas"]["lluvia_mayo_base"] * 1.4, abs=1)
    assert any(f["variable"] == "tmin_abril" for f in s["fuera_de_rango"])
    assert s["escenario"]["sup"] - s["escenario"]["inf"] > s["base"]["sup"] - s["base"]["inf"]


def test_el_cambio_viene_solo_de_la_parte_del_ensamble(art):
    """Con la mezcla, el LightGBM de longitud no reacciona al clima: el efecto es la mitad del ensamble."""
    fila = art.features.loc["AGC_003"]
    esc = simulador.aplicar_escenario(fila, 20, 0)
    pares = art.features.loc[["AGC_003"]].copy()
    pares.loc["esc"] = esc
    m = art.modelo
    Xi = m.imputador.transform(pares[m.cols].astype(float).to_numpy())
    ensamble = (m.lgbm.predict(Xi) + m.rf.predict(Xi) + m.enet.predict(m.escalador.transform(Xi))) / 3
    longitud = m.lgbm_longitud.predict(Xi[:, [m.i_longitud]])
    assert longitud[0] == longitud[1]
    mezcla = m.predecir(pares)
    assert mezcla[1] - mezcla[0] == pytest.approx((ensamble[1] - ensamble[0]) / 2)


def test_variables_derivadas_se_recalculan(art):
    fila = art.features.loc["AGC_003"]
    esc = simulador.aplicar_escenario(fila, -40, 0)
    assert esc["prec_anomalia"] - fila["prec_anomalia"] == pytest.approx(-0.4 * fila["prec_mm_m05"])
    assert esc["meses_secos"] >= fila["meses_secos"]


def test_errores(cliente):
    assert _sim(cliente, id="NOPE")[0].status_code == 404
    assert _sim(cliente, id="AGC_003", delta_lluvia_mayo_pct="mucho")[0].status_code == 400
    assert cliente.post("/api/simulador", data="no es json", content_type="application/json").status_code == 400
    assert cliente.get("/api/simulador").status_code == 405
