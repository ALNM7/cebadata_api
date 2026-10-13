"""La API tiene que reproducir las predicciones de las 59 parcelas con los artefactos cargados."""
import numpy as np
import pandas as pd

from pipeline_rutas import RAIZ


def test_modelo_reproduce_la_tabla_de_las_59(art):
    tabla = pd.read_csv(RAIZ / "artefactos" / "tablas" / "predicciones.csv").set_index("id")
    ids = tabla.index[~tabla["fuera_de_muestra"]]
    assert len(ids) == 59
    pred = art.modelo.predecir(art.features.loc[ids])
    # la tabla guarda 12 cifras significativas
    assert np.abs(pred - tabla.loc[ids, "prediccion"].to_numpy()).max() < 1e-9


def test_modelo_reproduce_el_csv_del_pipeline(art):
    csv = pd.read_csv(RAIZ / "pipeline" / "salidas" / "predicciones_59_parcelas.csv").set_index("ID_POLIGONO")
    pred = art.modelo.predecir(art.features.loc[csv.index])
    assert (np.round(pred, 3) == csv["RENDIMIENTO_T_HA_PRED"].to_numpy()).all()
    # intervalo: prediccion mas/menos el cuantil de CV de la mezcla
    q = art.metadata["cuantil_intervalo"]
    assert np.allclose(np.round(pred - q, 3), csv["intervalo_inf"].to_numpy(), atol=1.1e-3)
    assert np.allclose(np.round(pred + q, 3), csv["intervalo_sup"].to_numpy(), atol=1.1e-3)


def test_la_api_sirve_esas_mismas_predicciones(art, cliente):
    csv = pd.read_csv(RAIZ / "pipeline" / "salidas" / "predicciones_59_parcelas.csv").set_index("ID_POLIGONO")
    tabla = pd.read_csv(RAIZ / "artefactos" / "tablas" / "predicciones.csv").set_index("id")
    servido = {p["id"]: p for p in cliente.get("/api/parcelas").json()}
    for pid, fila in csv.iterrows():
        assert servido[pid]["conjunto"] == "PREDICCION"
        # se sirve la tabla, no se recalcula: coincide con la tabla y con el CSV (que trae 3 decimales)
        assert servido[pid]["rendimiento_pred"] == round(tabla.loc[pid, "prediccion"], 2)
        assert abs(servido[pid]["rendimiento_pred"] - fila["RENDIMIENTO_T_HA_PRED"]) < 0.0051
