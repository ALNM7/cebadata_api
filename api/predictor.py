"""La mezcla: promedio del ensamble (LightGBM + Random Forest + Elastic Net) y de un
LightGBM que solo usa la longitud.

Reconstruye la prediccion con los componentes que exporta el paso 10 del pipeline, sin
importar el pipeline. El orden de las variables y los pesos salen de metadata.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Mezcla:
    def __init__(self, componentes: dict, metadata: dict):
        self.cols: list[str] = list(metadata["variables"])
        self.pesos: dict[str, float] = metadata["pesos_mezcla"]
        self.i_longitud = self.cols.index(metadata["variable_longitud"])
        self.imputador = componentes["imputador"]
        self.escalador = componentes["escalador"]
        self.lgbm = componentes["lgbm"]
        self.rf = componentes["rf"]
        self.enet = componentes["enet"]
        self.lgbm_longitud = componentes["lgbm_longitud"]
        # Una fila por peticion: los hilos extra solo estorban (y los hilos de OpenMP no se
        # llevan bien con el fork de gunicorn --preload).
        for m in (self.lgbm, self.rf, self.lgbm_longitud):
            m.set_params(n_jobs=1)

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        """X trae (al menos) las columnas del modelo; los faltantes (NaN) se imputan con la mediana de entrenamiento."""
        Xi = self.imputador.transform(X[self.cols].astype(float).to_numpy())
        p = self.pesos
        return (
            p["lgbm"] * self.lgbm.predict(Xi)
            + p["rf"] * self.rf.predict(Xi)
            + p["enet"] * self.enet.predict(self.escalador.transform(Xi))
            + p["lgbm_longitud"] * self.lgbm_longitud.predict(Xi[:, [self.i_longitud]])
        )
