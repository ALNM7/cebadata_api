"""Paso 06: modelos baseline con validacion honesta.

Entrena tres familias sobre features_modelado.csv y reporta metricas con
validacion cruzada repetida (5 particiones x 10 repeticiones) estratificada
por municipio, que es el protocolo que definimos en la propuesta. Los
municipios con menos de 5 parcelas de entrenamiento se agrupan en OTRO
para poder estratificar.

Modelos:
  - Media global (piso absoluto de referencia)
  - Elastic Net (lineal regularizado, con estandarizacion)
  - Random Forest
  - HistGradientBoosting (gradient boosting de sklearn, sin dependencias extra)

Salidas:
  salidas/metricas_baseline.csv
  salidas/importancias_baseline.csv  (importancia por permutacion del mejor modelo)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline import conf as config

SEMILLA = 42
N_SPLITS, N_REPETICIONES = 5, 10


def preparar_datos():
    df = pd.read_csv(config.RUTA_SALIDAS / "features_modelado.csv")
    entrena = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)

    y = entrena["RENDIMIENTO_T_HA"].values
    X = entrena.drop(columns=["ID_POLIGONO", "CONJUNTO", "RENDIMIENTO_T_HA"])

    # Estrato para la validacion: municipio, agrupando los chicos
    conteo = entrena["Municipio"].value_counts()
    estrato = entrena["Municipio"].where(entrena["Municipio"].map(conteo) >= 5, "OTRO")

    cols_cat = ["Estado", "Municipio"]
    cols_num = [c for c in X.columns if c not in cols_cat]
    return X, y, estrato.values, cols_num, cols_cat


def construir_modelos(cols_num, cols_cat):
    prep_arboles = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), cols_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
    ])
    prep_lineal = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("esc", StandardScaler())]), cols_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
    ])
    return {
        "media_global": Pipeline([("prep", prep_arboles),
                                  ("m", DummyRegressor(strategy="mean"))]),
        "elastic_net": Pipeline([("prep", prep_lineal),
                                 ("m", ElasticNet(alpha=0.05, l1_ratio=0.5,
                                                  max_iter=20000, random_state=SEMILLA))]),
        "random_forest": Pipeline([("prep", prep_arboles),
                                   ("m", RandomForestRegressor(n_estimators=600,
                                                               min_samples_leaf=2,
                                                               random_state=SEMILLA,
                                                               n_jobs=-1))]),
        "hist_gradient_boosting": Pipeline([("prep", prep_arboles),
                                            ("m", HistGradientBoostingRegressor(
                                                max_iter=400, learning_rate=0.05,
                                                max_leaf_nodes=15, l2_regularization=1.0,
                                                random_state=SEMILLA))]),
    }


def evaluar(modelos, X, y, estrato):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPETICIONES,
                                 random_state=SEMILLA)
    filas = []
    for nombre, modelo in modelos.items():
        rmse, mae, r2 = [], [], []
        for tr, te in cv.split(X, estrato):
            modelo.fit(X.iloc[tr], y[tr])
            pred = modelo.predict(X.iloc[te])
            rmse.append(np.sqrt(mean_squared_error(y[te], pred)))
            mae.append(mean_absolute_error(y[te], pred))
            r2.append(r2_score(y[te], pred))
        filas.append({
            "modelo": nombre,
            "rmse": np.mean(rmse), "rmse_std": np.std(rmse),
            "mae": np.mean(mae), "mae_std": np.std(mae),
            "r2": np.mean(r2), "r2_std": np.std(r2),
        })
        print(f"{nombre:24s} RMSE {np.mean(rmse):.3f} +/- {np.std(rmse):.3f} | "
              f"MAE {np.mean(mae):.3f} | R2 {np.mean(r2):.3f} +/- {np.std(r2):.3f}")
    return pd.DataFrame(filas)


def importancias(modelo, X, y, cols_num, cols_cat):
    """Importancia por permutacion sobre las columnas originales (pre One-Hot)."""
    modelo.fit(X, y)
    resultado = permutation_importance(modelo, X, y, n_repeats=15,
                                       random_state=SEMILLA, n_jobs=-1)
    imp = pd.DataFrame({"variable": X.columns,
                        "importancia": resultado.importances_mean,
                        "std": resultado.importances_std})
    return imp.sort_values("importancia", ascending=False)


def main():
    X, y, estrato, cols_num, cols_cat = preparar_datos()
    print(f"Entrenamiento: {len(y)} parcelas, {X.shape[1]} variables\n")

    modelos = construir_modelos(cols_num, cols_cat)
    metricas = evaluar(modelos, X, y, estrato)
    metricas.round(4).to_csv(config.RUTA_SALIDAS / "metricas_baseline.csv", index=False)

    mejor = metricas.sort_values("rmse").iloc[0]["modelo"]
    print(f"\nMejor baseline: {mejor}. Calculando importancia por permutacion...")
    imp = importancias(modelos[mejor], X, y, cols_num, cols_cat)
    imp.round(5).to_csv(config.RUTA_SALIDAS / "importancias_baseline.csv", index=False)
    print("\nTop 15 variables:")
    print(imp.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
