"""Carga unica de los artefactos congelados (carpeta artefactos/).

`cargar()` se llama una vez al arrancar (api.apps.ApiConfig.ready). Todo queda en memoria:
las respuestas salen de las tablas precalculadas y solo el simulador vuelve a predecir.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from importlib import metadata as _importlib_metadata
from pathlib import Path

import joblib
import pandas as pd

from api.predictor import Mezcla

log = logging.getLogger(__name__)

VERSION_SOPORTADA = 1
# Estas versiones tienen que coincidir con las del entrenamiento para que los .joblib carguen igual.
LIBRERIAS_ESTRICTAS = {"scikit-learn": "scikit-learn", "lightgbm": "lightgbm"}
LIBRERIAS_AVISO = {"numpy": "numpy", "pandas": "pandas", "joblib": "joblib"}
COMPONENTES = ["imputador", "escalador", "lgbm", "rf", "enet", "lgbm_longitud"]


class ErrorArtefactos(RuntimeError):
    pass


@dataclass
class Artefactos:
    ruta: Path
    metadata: dict
    dashboard: dict
    modelo: Mezcla
    parcelas: dict[str, dict]            # id -> datos de la parcela (+ prediccion e intervalo)
    orden: list[str]                     # ids ordenados
    features: pd.DataFrame               # indexado por id, columnas = variables del modelo
    shap: dict[str, list[dict]]
    shap_otras: dict[str, float]
    clima: dict[str, list[dict]]
    serie: dict[str, list[dict]]
    vecinos: dict[str, list[dict]]
    extras: dict = field(default_factory=dict)

    @property
    def base(self) -> float:
        return self.metadata["base"]

    @property
    def semiancho(self) -> float:
        return self.metadata["cuantil_intervalo"]


ACTUAL: Artefactos | None = None


def obtener() -> Artefactos:
    if ACTUAL is None:
        raise ErrorArtefactos("Los artefactos no se han cargado.")
    return ACTUAL


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _version(paquete: str) -> str | None:
    try:
        return _importlib_metadata.version(paquete)
    except _importlib_metadata.PackageNotFoundError:
        return None


def _validar(ruta: Path, meta: dict) -> None:
    if meta.get("version_artefactos") != VERSION_SOPORTADA:
        raise ErrorArtefactos(
            f"Version de artefactos {meta.get('version_artefactos')!r}; esta API soporta {VERSION_SOPORTADA}.")
    for nombre, esperado in meta.get("checksums", {}).items():
        archivo = ruta / nombre
        if not archivo.exists():
            raise ErrorArtefactos(f"Falta {archivo}.")
        if _sha256(archivo) != esperado:
            raise ErrorArtefactos(f"{nombre} no coincide con el checksum de metadata.json (archivo alterado o de otra corrida).")
    entrenadas = meta.get("librerias", {})
    for clave, paquete in LIBRERIAS_ESTRICTAS.items():
        if entrenadas.get(clave) != _version(paquete):
            raise ErrorArtefactos(
                f"{paquete} {_version(paquete)} instalado, pero los artefactos se entrenaron con {entrenadas.get(clave)}. "
                "Instala las versiones de requirements.txt o regenera los artefactos.")
    for clave, paquete in LIBRERIAS_AVISO.items():
        if entrenadas.get(clave) != _version(paquete):
            log.warning("%s %s instalado; los artefactos se entrenaron con %s.", paquete, _version(paquete), entrenadas.get(clave))


def _agrupar(df: pd.DataFrame, clave: str) -> dict[str, list[dict]]:
    return {k: g.drop(columns=clave).to_dict("records") for k, g in df.groupby(clave, sort=False)}


def cargar(ruta: str | Path) -> Artefactos:
    """Lee y valida los artefactos; deja el resultado en `ACTUAL`."""
    global ACTUAL
    ruta = Path(ruta)
    meta_ruta = ruta / "metadata.json"
    if not meta_ruta.exists():
        raise ErrorArtefactos(
            f"No encuentro {meta_ruta}. Genera los artefactos con `python -m pipeline.run_pipeline --desde 10` "
            "o apunta CEBADATA_ARTEFACTOS a la carpeta correcta.")
    meta = json.loads(meta_ruta.read_text())
    _validar(ruta, meta)

    componentes = {n: joblib.load(ruta / "modelo" / f"{n}.joblib") for n in COMPONENTES}
    modelo = Mezcla(componentes, meta)
    dash = json.loads((ruta / "metricas_dashboard.json").read_text())

    t = ruta / "tablas"
    parcelas_df = pd.read_csv(t / "parcelas.csv")
    pred = pd.read_csv(t / "predicciones.csv").set_index("id")
    parcelas: dict[str, dict] = {}
    for f in parcelas_df.itertuples():
        p = pred.loc[f.id]
        obs = None if pd.isna(f.rendimiento_obs) else round(float(f.rendimiento_obs), 2)
        parcelas[f.id] = {
            "id": f.id, "estado": f.estado, "municipio": f.municipio, "area_ha": round(float(f.area_ha), 1),
            "conjunto": f.conjunto, "rendimiento_obs": obs,
            "rendimiento_pred": round(float(p.prediccion), 2),
            "intervalo_inf": round(float(p.inferior), 2), "intervalo_sup": round(float(p.superior), 2),
            "lat": round(float(f.lat), 5), "lon": round(float(f.lon), 5),
            "elevacion_m": round(float(f.elevacion_m)), "pendiente_grados": round(float(f.pendiente_grados), 1),
        }
    orden = sorted(parcelas)

    feats = pd.read_csv(t / "features_modelo.csv").set_index("id")
    faltan = [c for c in meta["variables"] if c not in feats.columns]
    if faltan:
        raise ErrorArtefactos(f"features_modelo.csv no trae las variables {faltan[:3]}...")
    feats = feats[meta["variables"]]

    shap_df = pd.read_csv(t / "shap_parcela.csv", keep_default_na=False, na_values=[]).sort_values(["id", "orden"])
    shap = {k: [{c: x[c] for c in ("variable", "etiqueta", "valor_variable", "unidad", "contribucion")} for x in rows]
            for k, rows in _agrupar(shap_df, "id").items()}
    clima = _agrupar(pd.read_csv(t / "clima_mensual.csv").sort_values(["id", "mes"]), "id")

    serie_ruta = t / "serie_satelital.csv.gz"
    serie = _agrupar(pd.read_csv(serie_ruta), "id") if serie_ruta.exists() else {}

    v = pd.read_csv(t / "vecinos.csv").sort_values(["id", "orden"])
    vecinos = {}
    for pid, g in v.groupby("id"):
        vecinos[pid] = [{"id": x.vecino, "municipio": parcelas[x.vecino]["municipio"],
                         "distancia_km": round(float(x.distancia_km), 2),
                         "rendimiento": (parcelas[x.vecino]["rendimiento_obs"] if parcelas[x.vecino]["rendimiento_obs"] is not None
                                         else parcelas[x.vecino]["rendimiento_pred"]),
                         "tipo": "observado" if parcelas[x.vecino]["rendimiento_obs"] is not None else "predicho"}
                        for x in g.itertuples()]

    ACTUAL = Artefactos(
        ruta=ruta, metadata=meta, dashboard=dash, modelo=modelo, parcelas=parcelas, orden=orden, features=feats,
        shap=shap, shap_otras={pid: float(o) for pid, o in pred["shap_otras"].items()},
        clima=clima, serie=serie, vecinos=vecinos,
    )
    log.info("Artefactos cargados: %d parcelas, %d variables, generados %s",
             len(parcelas), len(meta["variables"]), meta.get("generado"))
    return ACTUAL
