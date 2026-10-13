"""Pruebas del pipeline. Necesitan el entorno de requirements-pipeline.txt (optuna, shap, ...)."""
import pandas as pd
import pytest

pytest.importorskip("optuna")

from pipeline import conf, etiquetas  # noqa: E402
from pipeline.modelado import podar_redundantes, recomendar, resumen_metricas  # noqa: E402


def _tabla(estr, lomo):
    return ({c: {"rmse": v} for c, v in zip(conf.CANDIDATOS, estr)},
            {c: {"rmse": v} for c, v in zip(conf.CANDIDATOS, lomo)})


def test_recomendar_prefiere_robustez_cuando_hay_empate():
    e, l = _tabla([0.508, 0.525, 0.508], [0.722, 0.539, 0.594])
    assert recomendar(e, l)[0] == "mezcla"


def test_recomendar_respeta_una_ventaja_clara():
    e, l = _tabla([0.480, 0.525, 0.508], [0.722, 0.539, 0.594])
    assert recomendar(e, l)[0] == "ensamble"


def test_podar_redundantes_quita_el_duplicado():
    x = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [2.0, 4, 6, 8, 10], "c": [5.0, 1, 4, 2, 3]})
    assert podar_redundantes(x) in (["a", "c"], ["b", "c"])


def test_sesgo_positivo_si_sobreestima():
    m = resumen_metricas(pd.Series([1.0, 2.0, 3.0]).values, pd.Series([1.5, 2.5, 3.5]).values)
    assert m["sesgo"] == pytest.approx(0.5) and m["rmse"] == pytest.approx(0.5)


@pytest.mark.parametrize("variable,esperado", [
    ("prec_mm_m05", "Lluvia de mayo"), ("tmin_c_m04", "Temperatura mínima de abril"),
    ("ndvi_tasa_verdeo", "Velocidad de verdeo NDVI"), ("mun_Singuilucan", "Municipio Singuilucan"), ("lon", "Longitud"),
])
def test_etiquetas(variable, esperado):
    assert etiquetas.etiqueta(variable) == esperado


def test_tasas_se_pasan_a_por_dia():
    valor, unidad = etiquetas.valor_y_unidad("ndvi_tasa_verdeo", 0.52)
    assert unidad == "por día" and valor == pytest.approx(0.0052)


def test_el_modelo_que_se_sirve_es_la_mezcla():
    assert conf.MODELO_FINAL == "mezcla"
