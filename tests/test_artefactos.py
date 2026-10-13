import re
import sys
from pathlib import Path

import pytest

from pipeline_rutas import RAIZ


def _pins(ruta: Path) -> dict[str, str]:
    out = {}
    for linea in ruta.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([\w.]+)", linea.strip())
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def test_metadata_es_coherente(art):
    m = art.metadata
    assert len(m["variables"]) == 121 and m["variable_longitud"] in m["variables"]
    assert sum(m["pesos_mezcla"].values()) == pytest.approx(1.0)
    assert m["pesos_mezcla"]["lgbm_longitud"] == 0.5
    assert 0.5 < m["cuantil_intervalo"] < 1.0
    assert set(m["rangos_entrenamiento"]) == set(m["variables"]) == set(m["detalle_variables"])
    assert all(r["min"] <= r["max"] for r in m["rangos_entrenamiento"].values())
    assert list(art.features.columns) == m["variables"]


def test_versiones_fijadas_coinciden_con_el_entrenamiento(art):
    entrenadas = art.metadata["librerias"]
    api = _pins(RAIZ / "requirements.txt")
    pipe = _pins(RAIZ / "pipeline" / "requirements-pipeline.txt")
    for paquete in ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib"):
        assert api[paquete] == pipe[paquete] == entrenadas[paquete], paquete


def test_la_api_no_importa_el_pipeline(art):
    # ni los modulos cargados ni el codigo de la app
    assert not any(n == "pipeline" or n.startswith("pipeline.") for n in sys.modules)
    for archivo in (RAIZ / "api").rglob("*.py"):
        assert not re.search(r"^\s*(from|import)\s+pipeline\b", archivo.read_text(), re.M), archivo
    runtime = (RAIZ / "requirements.txt").read_text().lower()
    for pesado in ("optuna", "shap", "rasterio", "geopandas", "matplotlib"):
        assert pesado not in runtime


def test_tablas_completas(art):
    assert len(art.parcelas) == 197 and len(art.features) == 197
    assert sum(p["conjunto"] == "PREDICCION" for p in art.parcelas.values()) == 59
    assert all(len(v) == 5 for v in art.vecinos.values()) and len(art.vecinos) == 197
    assert all(len(v) == 8 for v in art.shap.values())


def test_si_se_altera_un_modelo_el_arranque_falla(tmp_path):
    import shutil

    from api import artefactos

    copia = tmp_path / "artefactos"
    shutil.copytree(RAIZ / "artefactos", copia)
    with open(copia / "modelo" / "rf.joblib", "ab") as f:
        f.write(b"x")
    anterior = artefactos.ACTUAL
    try:
        with pytest.raises(artefactos.ErrorArtefactos, match="checksum"):
            artefactos.cargar(copia)
    finally:
        artefactos.ACTUAL = anterior
