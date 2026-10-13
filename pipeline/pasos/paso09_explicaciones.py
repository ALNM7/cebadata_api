"""Paso 09: explicacion por parcela (valores SHAP) para la ficha del dashboard.

Explica la parte A (LightGBM del ensamble), que es la que tiene sentido por
parcela: el modelo de ubicacion solo usa la longitud y casi no tiene que
contar. Para cada una de las 197 parcelas guarda las variables con mas efecto.

  - Parcelas ocultas (59): SHAP del LightGBM final.
  - Parcelas medidas (138): SHAP de un LightGBM con los mismos hiperparametros
    entrenado SIN esa parcela (leave-one-out). Asi la explicacion corresponde a
    una prediccion que el modelo no vio, igual que la prediccion fuera de
    muestra que muestra el dashboard.

La cascada del dashboard parte del rendimiento promedio de las parcelas
medidas y suma estas contribuciones. Lo que falta para llegar a la
prediccion final (variables fuera del top, ajuste de los otros modelos del
ensamble y la mezcla con la ubicacion) se agrupa en "otras variables"; esa
resta la hace la API al cargar los resultados.

Salidas:
  salidas/shap_parcela.csv   (ID_POLIGONO, orden, variable, valor_variable, contribucion)
"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from pipeline import conf as config
from pipeline.modelado import cargar_datos

TOP = 12


def main():
    d = cargar_datos()
    art = joblib.load(config.RUTA_MODELOS / "modelo_final.joblib")
    modelo, cols = art["modelo"], art["cols"]

    Xi = pd.DataFrame(modelo.preparar(d.X_todas), columns=cols)
    params = modelo.parametros["lgbm"]
    ids_medidas = {pid: i for i, pid in enumerate(d.ids)}
    y = d.y

    pos = {pid: k for k, pid in enumerate(d.ids_todas)}
    medidas = Xi.iloc[[pos[q] for q in d.ids]]   # mismo orden que d.y
    contrib = np.zeros(Xi.shape)
    final = shap.TreeExplainer(modelo.lgbm).shap_values(Xi)
    for k, pid in enumerate(d.ids_todas):
        if pid not in ids_medidas:
            contrib[k] = final[k]
            continue
        i = ids_medidas[pid]
        mascara = np.ones(len(y), dtype=bool)
        mascara[i] = False
        m = lgb.LGBMRegressor(**params).fit(medidas.iloc[mascara], y[mascara])
        contrib[k] = shap.TreeExplainer(m).shap_values(Xi.iloc[[k]])[0]

    filas = []
    for k, pid in enumerate(d.ids_todas):
        top = np.argsort(-np.abs(contrib[k]))[:TOP]
        for orden, j in enumerate(top, start=1):
            filas.append({"ID_POLIGONO": pid, "orden": orden, "variable": cols[j],
                          "valor_variable": float(Xi.iloc[k, j]), "contribucion": float(contrib[k, j])})
    salida = pd.DataFrame(filas)
    salida.round(5).to_csv(config.RUTA_SALIDAS / "shap_parcela.csv", index=False)
    print(f"shap_parcela.csv: {salida.ID_POLIGONO.nunique()} parcelas x {TOP} variables")


if __name__ == "__main__":
    main()
