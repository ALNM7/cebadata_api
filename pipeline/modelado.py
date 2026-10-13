"""Piezas compartidas de modelado para los pasos 07 a 09 y para la API.

Aqui vive la definicion de los tres candidatos que se comparan en el paso 07:

  ensamble   LightGBM + Random Forest + Elastic Net con todas las variables
             (poda de redundancia incluida). Es el modelo del pipeline original.
  ubicacion  LightGBM que solo usa la longitud. Es lo que encontro el
             "superciclo" de modelos clasicos del companero: en esta region
             el rendimiento cambia de oeste a este y la longitud lo captura.
  mezcla     promedio simple de los dos anteriores.

`ModeloAgroCebada` ajusta (con Optuna sobre los datos que recibe) y predice
los tres candidatos a la vez. Se usa igual dentro de cada fold de la
validacion anidada, en el modelo final y en el simulador de la API.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pipeline import conf

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEMILLA = 42
COL_UBICACION = "lon"
# Intentos de Optuna por modelo dentro de cada fold: lgbm, rf, enet, ubicacion
TRIALS = {"lgbm": 20, "rf": 8, "enet": 15, "ubicacion": 10}
MIEMBROS = ["lgbm", "rf", "enet", "ensamble", "ubicacion", "mezcla"]


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------
@dataclass
class Datos:
    X: pd.DataFrame            # 138 parcelas de entrenamiento (con one-hot de estado y municipio)
    y: np.ndarray
    estrato: np.ndarray        # municipio (los de menos de 5 parcelas, agrupados en OTRO)
    municipios: np.ndarray     # municipio original de cada parcela de entrenamiento
    ids: np.ndarray
    X_pred: pd.DataFrame       # 59 parcelas ocultas
    ids_pred: np.ndarray
    X_todas: pd.DataFrame      # las 197, mismo orden que ids_todas
    ids_todas: np.ndarray


def cargar_datos() -> Datos:
    df = pd.read_csv(conf.RUTA_SALIDAS / "features_modelado.csv")

    # Municipio y estado a one-hot (pocas categorias, sin orden natural)
    df = pd.get_dummies(df, columns=["Estado", "Municipio"], prefix=["est", "mun"])

    entrena = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
    predice = df[df["CONJUNTO"] == "PREDICCION"].reset_index(drop=True)

    quitar = ["ID_POLIGONO", "CONJUNTO", "RENDIMIENTO_T_HA"]
    y = entrena["RENDIMIENTO_T_HA"].values
    X = entrena.drop(columns=quitar)
    X_pred = predice.drop(columns=quitar)
    X_todas = df.drop(columns=quitar)

    # Estrato de validacion: municipio original (antes del one-hot)
    maestro = pd.read_csv(conf.RUTA_SALIDAS / "parcelas_maestro.csv").set_index("ID_POLIGONO")
    mun = maestro.loc[entrena["ID_POLIGONO"], "Municipio"]
    conteo = mun.value_counts()
    estrato = mun.where(mun.map(conteo) >= 5, "OTRO").values

    return Datos(X=X, y=y, estrato=estrato, municipios=mun.values,
                 ids=entrena["ID_POLIGONO"].values,
                 X_pred=X_pred, ids_pred=predice["ID_POLIGONO"].values,
                 X_todas=X_todas, ids_todas=df["ID_POLIGONO"].values)


def podar_redundantes(X: pd.DataFrame, umbral: float = 0.97) -> list[str]:
    """Devuelve la lista de columnas a conservar tras podar pares |r| > umbral."""
    num = X.select_dtypes(include=[float, int])
    corr = num.corr().abs()
    superior = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    tirar = [c for c in superior.columns if (superior[c] > umbral).any()]
    return [c for c in X.columns if c not in tirar]


# ----------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------
def rmse(y, pred) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def resumen_metricas(y, pred) -> dict:
    return {"rmse": rmse(y, pred), "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)), "sesgo": float(np.mean(np.asarray(pred) - y))}


# ----------------------------------------------------------------------
# Busqueda de hiperparametros (cada modelo se ajusta SOLO con los datos del
# fold de entrenamiento que recibe; nada del fold externo se filtra)
# ----------------------------------------------------------------------
def _cv_interna_rmse(construir, X, y, n_splits=3):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEMILLA)
    errores = []
    for tr, va in cv.split(X):
        m = construir()
        m.fit(X[tr], y[tr])
        errores.append(np.sqrt(mean_squared_error(y[va], m.predict(X[va]))))
    return float(np.mean(errores))


def _estudio(objetivo, n_trials):
    estudio = optuna.create_study(direction="minimize",
                                  sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    estudio.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    return estudio.best_params


def tunear_lgbm(X, y, n_trials):
    def objetivo(t):
        params = dict(
            n_estimators=t.suggest_int("n_estimators", 100, 600),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=t.suggest_int("num_leaves", 4, 31),
            min_child_samples=t.suggest_int("min_child_samples", 5, 30),
            reg_lambda=t.suggest_float("reg_lambda", 0.0, 10.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1, random_state=SEMILLA, verbosity=-1, n_jobs=2,
        )
        return _cv_interna_rmse(lambda: lgb.LGBMRegressor(**params), X, y)
    mejores = _estudio(objetivo, n_trials) | dict(subsample_freq=1, random_state=SEMILLA,
                                                  verbosity=-1, n_jobs=2)
    return lgb.LGBMRegressor(**mejores), mejores


def tunear_rf(X, y, n_trials):
    def objetivo(t):
        params = dict(
            n_estimators=150,
            max_depth=t.suggest_int("max_depth", 4, 20),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 8),
            max_features=t.suggest_float("max_features", 0.2, 1.0),
            random_state=SEMILLA, n_jobs=2,
        )
        return _cv_interna_rmse(lambda: RandomForestRegressor(**params), X, y)
    mejores = _estudio(objetivo, n_trials) | dict(n_estimators=600, random_state=SEMILLA, n_jobs=2)
    return RandomForestRegressor(**mejores), mejores


def tunear_enet(X, y, n_trials):
    def objetivo(t):
        params = dict(alpha=t.suggest_float("alpha", 1e-3, 1.0, log=True),
                      l1_ratio=t.suggest_float("l1_ratio", 0.05, 0.95),
                      max_iter=20000, random_state=SEMILLA)
        return _cv_interna_rmse(lambda: ElasticNet(**params), X, y)
    mejores = _estudio(objetivo, n_trials) | dict(max_iter=20000, random_state=SEMILLA)
    return ElasticNet(**mejores), mejores


def tunear_ubicacion(X, y, n_trials):
    """LightGBM de una sola variable (la longitud): arboles chicos y muy regularizados."""
    def objetivo(t):
        params = dict(
            n_estimators=t.suggest_int("n_estimators", 50, 400),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=t.suggest_int("num_leaves", 2, 8),
            min_child_samples=t.suggest_int("min_child_samples", 5, 30),
            reg_lambda=t.suggest_float("reg_lambda", 0.0, 10.0),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1, random_state=SEMILLA, verbosity=-1, n_jobs=2,
        )
        return _cv_interna_rmse(lambda: lgb.LGBMRegressor(**params), X, y)
    mejores = _estudio(objetivo, n_trials) | dict(subsample_freq=1, random_state=SEMILLA,
                                                  verbosity=-1, n_jobs=2)
    return lgb.LGBMRegressor(**mejores), mejores


# ----------------------------------------------------------------------
# Modelo con los tres candidatos
# ----------------------------------------------------------------------
class ModeloAgroCebada:
    """Ajusta y predice ensamble, ubicacion y mezcla a partir de las mismas columnas."""

    def __init__(self, cols: list[str], factor_trials: int = 1):
        if COL_UBICACION not in cols:
            raise ValueError(f"La columna {COL_UBICACION!r} tiene que estar en las variables")
        self.cols = list(cols)
        self.factor_trials = factor_trials
        self.parametros: dict[str, dict] = {}

    # -- ajuste --------------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> "ModeloAgroCebada":
        Xn = X[self.cols].astype(float).values
        y = np.asarray(y, dtype=float)
        self.imputador = SimpleImputer(strategy="median").fit(Xn)
        Xi = self.imputador.transform(Xn)
        self.escalador = StandardScaler().fit(Xi)
        k = self.factor_trials

        self.lgbm, self.parametros["lgbm"] = tunear_lgbm(Xi, y, TRIALS["lgbm"] * k)
        self.rf, self.parametros["rf"] = tunear_rf(Xi, y, TRIALS["rf"] * k)
        self.enet, self.parametros["enet"] = tunear_enet(
            self.escalador.transform(Xi), y, TRIALS["enet"] * k)
        self.ubicacion, self.parametros["ubicacion"] = tunear_ubicacion(
            Xi[:, [self.i_ubicacion]], y, TRIALS["ubicacion"] * k)

        self.lgbm.fit(Xi, y)
        self.rf.fit(Xi, y)
        self.enet.fit(self.escalador.transform(Xi), y)
        self.ubicacion.fit(Xi[:, [self.i_ubicacion]], y)
        return self

    @property
    def i_ubicacion(self) -> int:
        return self.cols.index(COL_UBICACION)

    # -- prediccion ----------------------------------------------------
    def preparar(self, X: pd.DataFrame) -> np.ndarray:
        return self.imputador.transform(X[self.cols].astype(float).values)

    def predecir_matriz(self, Xi: np.ndarray) -> dict[str, np.ndarray]:
        p = {
            "lgbm": self.lgbm.predict(Xi),
            "rf": self.rf.predict(Xi),
            "enet": self.enet.predict(self.escalador.transform(Xi)),
            "ubicacion": self.ubicacion.predict(Xi[:, [self.i_ubicacion]]),
        }
        p["ensamble"] = (p["lgbm"] + p["rf"] + p["enet"]) / 3
        p["mezcla"] = (p["ensamble"] + p["ubicacion"]) / 2
        return p

    def predecir(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.predecir_matriz(self.preparar(X))


def recomendar(estratificada: dict[str, dict], lomo: dict[str, dict],
               umbral: float = conf.UMBRAL_RUIDO_RMSE) -> tuple[str, str]:
    """Sugiere un candidato con la logica del documento de decision.

    - Donde se califica (CV estratificada) cuenta quien quede a menos de `umbral`
      del mejor RMSE como empatado.
    - Entre los empatados se prefiere el que mejor aguante zonas nuevas (LOMO).
    Es una justificacion para el reporte; el modelo que se sirve es siempre la mezcla.
    """
    cand = conf.CANDIDATOS
    mejor_est = min(estratificada[c]["rmse"] for c in cand)
    empatados = [c for c in cand if estratificada[c]["rmse"] - mejor_est < umbral]
    elegido = min(empatados, key=lambda c: lomo[c]["rmse"])
    motivo = (f"{'/'.join(empatados)} quedan a menos de {umbral:.3f} t/ha del mejor RMSE estratificado "
              f"({mejor_est:.3f}); entre ellos {elegido} tiene el menor RMSE en LOMO "
              f"({lomo[elegido]['rmse']:.3f}).")
    return elegido, motivo
