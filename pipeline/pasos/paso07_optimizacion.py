"""Paso 07: optimizacion, comparacion de candidatos y predicciones finales.

Mejora sobre los baselines del paso 06 usando solo el dataset oficial y compara
tres candidatos bajo las mismas reglas (ver pipeline/modelado.py):

  A. ensamble   LightGBM + Random Forest + Elastic Net con todas las variables
  B. ubicacion  LightGBM que solo usa la longitud (modelo del superciclo)
  C. mezcla     promedio de A y B

Dos validaciones, las dos anidadas (Optuna solo ve el fold de entrenamiento):

  - Estratificada por municipio (5 folds x 3 repeticiones). Mide lo que se va a
    calificar, porque las 59 parcelas ocultas estan en los mismos municipios.
  - LOMO (leave-one-municipality-out): se deja un municipio completo fuera en
    cada ronda. Mide que tan bien funciona en una zona nueva de la region.

Pasos del modelo:
  1. Poda de redundancia: de cada par con |correlacion| > 0.97 se conserva una.
  2. Validacion anidada de los tres candidatos con las dos validaciones.
  3. Modelo final: se re-optimiza sobre las 138 parcelas y se predicen las 59
     ocultas. El intervalo es empirico por candidato (cuantil 90 del error
     absoluto de la validacion estratificada).
  4. El candidato congelado lo fija conf.MODELO_FINAL (CEBADATA_MODELO_FINAL).
     El paso imprime ademas una recomendacion con la regla de ruido de 0.010.

Salidas (en salidas/):
  metricas_optimizadas.csv         estratificada: lgbm, rf, enet, ensamble, ubicacion, mezcla
  metricas_lomo.csv                LOMO, mismos modelos mas la media del resto
  comparacion_candidatos.csv       las dos validaciones de A, B y C lado a lado
  robustez_candidatos.csv          RMSE por municipio dejado fuera, por candidato
  oof_predicciones.csv             prediccion fuera de muestra de cada parcela medida
  hiperparametros_finales.json
  predicciones_59_parcelas.csv     prediccion final + intervalo + las de A, B y C
  predicciones_parcelas.csv        las 197 (medidas con prediccion fuera de muestra)
  modelos/modelo_final.joblib      modelo ajustado (lo usa el simulador de la API)
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold

from pipeline import conf as config
from pipeline.modelado import (MIEMBROS, SEMILLA, Datos, ModeloAgroCebada, cargar_datos,
                               podar_redundantes, recomendar, resumen_metricas, rmse)

N_SPLITS, N_REPETICIONES = 5, 3


# ----------------------------------------------------------------------
# Evaluaciones anidadas
# ----------------------------------------------------------------------
def evaluacion_estratificada(d: Datos, cols: list[str]):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPETICIONES, random_state=SEMILLA)
    n = len(d.y)
    pred = {m: np.full((N_REPETICIONES, n), np.nan) for m in MIEMBROS}
    errores_abs = {m: [] for m in MIEMBROS}
    inicio = time.time()

    for i, (tr, te) in enumerate(cv.split(d.X, d.estrato)):
        rep = i // N_SPLITS
        modelo = ModeloAgroCebada(cols).fit(d.X.iloc[tr], d.y[tr])
        p = modelo.predecir(d.X.iloc[te])
        for m in MIEMBROS:
            pred[m][rep, te] = p[m]
            errores_abs[m].extend(np.abs(d.y[te] - p[m]))
        print(f"  estratificada: fold {i + 1}/{N_SPLITS * N_REPETICIONES} "
              f"({time.time() - inicio:.0f} s)", flush=True)
    return pred, {m: float(np.quantile(e, 0.90)) for m, e in errores_abs.items()}


def evaluacion_lomo(d: Datos, cols: list[str]):
    """Un municipio completo fuera en cada ronda (se incluyen los chicos)."""
    n = len(d.y)
    pred = {m: np.full(n, np.nan) for m in MIEMBROS}
    pred["media_resto"] = np.full(n, np.nan)
    inicio = time.time()
    grupos = list(LeaveOneGroupOut().split(d.X, d.y, groups=d.municipios))

    for i, (tr, te) in enumerate(grupos):
        modelo = ModeloAgroCebada(cols).fit(d.X.iloc[tr], d.y[tr])
        p = modelo.predecir(d.X.iloc[te])
        for m in MIEMBROS:
            pred[m][te] = p[m]
        pred["media_resto"][te] = d.y[tr].mean()
        print(f"  LOMO: sin {d.municipios[te][0]} ({len(te)} parcelas) -> "
              f"{i + 1}/{len(grupos)} ({time.time() - inicio:.0f} s)", flush=True)
    return pred


# ----------------------------------------------------------------------
# Tablas
# ----------------------------------------------------------------------
def tabla_estratificada(d: Datos, pred) -> pd.DataFrame:
    filas = []
    for m in MIEMBROS:
        por_rep = [resumen_metricas(d.y, pred[m][r]) for r in range(N_REPETICIONES)]
        filas.append({
            "modelo": m,
            "rmse": np.mean([x["rmse"] for x in por_rep]),
            "rmse_std": np.std([x["rmse"] for x in por_rep]),
            "mae": np.mean([x["mae"] for x in por_rep]),
            "r2": np.mean([x["r2"] for x in por_rep]),
            "r2_std": np.std([x["r2"] for x in por_rep]),
            "sesgo": np.mean([x["sesgo"] for x in por_rep]),
        })
    return pd.DataFrame(filas)


def tabla_lomo(d: Datos, pred) -> pd.DataFrame:
    filas = []
    for m in MIEMBROS + ["media_resto"]:
        filas.append({"modelo": m} | resumen_metricas(d.y, pred[m]))
    return pd.DataFrame(filas)


def tabla_robustez(d: Datos, pred) -> pd.DataFrame:
    filas = []
    for mun in sorted(set(d.municipios)):
        idx = d.municipios == mun
        for m in MIEMBROS + ["media_resto"]:
            filas.append({"municipio_fuera": mun, "n_parcelas": int(idx.sum()), "modelo": m,
                          "rmse": rmse(d.y[idx], pred[m][idx])})
    return pd.DataFrame(filas)


def tabla_comparacion(est: pd.DataFrame, lomo: pd.DataFrame, n_vars: dict) -> pd.DataFrame:
    filas = []
    est, lomo = est.set_index("modelo"), lomo.set_index("modelo")
    for c in config.CANDIDATOS:
        for nombre, t in (("estratificada", est), ("lomo", lomo)):
            r = t.loc[c]
            filas.append({"candidato": c, "validacion": nombre, "rmse": r["rmse"],
                          "rmse_std": r.get("rmse_std", np.nan), "mae": r["mae"], "r2": r["r2"],
                          "sesgo": r["sesgo"], "n_variables": n_vars[c]})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------
# Modelo final y predicciones
# ----------------------------------------------------------------------
def modelo_final(d: Datos, cols: list[str], oof: dict, q90: dict):
    modelo = ModeloAgroCebada(cols, factor_trials=2).fit(d.X, d.y)
    final = config.MODELO_FINAL

    p59 = modelo.predecir(d.X_pred)

    salida = pd.DataFrame({
        "ID_POLIGONO": d.ids_pred,
        "RENDIMIENTO_T_HA_PRED": np.round(p59[final], 3),
        "intervalo_inf": np.round(p59[final] - q90[final], 3),
        "intervalo_sup": np.round(p59[final] + q90[final], 3),
        "pred_ensamble": np.round(p59["ensamble"], 3),
        "pred_ubicacion": np.round(p59["ubicacion"], 3),
        "pred_mezcla": np.round(p59["mezcla"], 3),
    })
    salida.to_csv(config.RUTA_SALIDAS / "predicciones_59_parcelas.csv", index=False)

    # Las 197: las medidas llevan su prediccion fuera de muestra (promedio de las
    # repeticiones de la CV estratificada); las ocultas, la del modelo final.
    maestro = pd.read_csv(config.RUTA_SALIDAS / "parcelas_maestro.csv").set_index("ID_POLIGONO")
    filas = []
    for pid in d.ids_todas:
        fila = {"ID_POLIGONO": pid, "CONJUNTO": maestro.loc[pid, "CONJUNTO"],
                "RENDIMIENTO_T_HA": maestro.loc[pid, "RENDIMIENTO_T_HA"]}
        if pid in set(d.ids):
            i = int(np.where(d.ids == pid)[0][0])
            for c in config.CANDIDATOS:
                fila[f"pred_{c}"] = float(np.nanmean(oof[c][:, i]))
            fila["fuera_de_muestra"] = True
        else:
            j = int(np.where(d.ids_pred == pid)[0][0])
            for c in config.CANDIDATOS:
                fila[f"pred_{c}"] = float(p59[c][j])
            fila["fuera_de_muestra"] = False
        fila["pred"] = fila[f"pred_{final}"]
        fila["intervalo_inf"] = fila["pred"] - q90[final]
        fila["intervalo_sup"] = fila["pred"] + q90[final]
        filas.append(fila)
    pd.DataFrame(filas).round(3).to_csv(config.RUTA_SALIDAS / "predicciones_parcelas.csv", index=False)

    config.RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "modelo_final": final, "q90": q90, "cols": cols},
                config.RUTA_MODELOS / "modelo_final.joblib", compress=3)
    with open(config.RUTA_SALIDAS / "hiperparametros_finales.json", "w") as f:
        json.dump({**modelo.parametros, "variables": cols, "modelo_final": final,
                   "q90_error_abs": q90[final], "q90_por_candidato": q90},
                  f, indent=2, default=str)

    assert salida.RENDIMIENTO_T_HA_PRED.between(1.5, 6.5).all(), "prediccion fuera de rango"
    print(f"\npredicciones_59_parcelas.csv ({final}): media {p59[final].mean():.2f} t/ha, "
          f"rango {p59[final].min():.2f} a {p59[final].max():.2f}")
    return salida


def main():
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    d = cargar_datos()
    cols = podar_redundantes(d.X)
    print(f"Poda de redundancia: {d.X.shape[1]} -> {len(cols)} variables")

    print("\nValidacion anidada estratificada por municipio:")
    pred_est, q90 = evaluacion_estratificada(d, cols)
    est = tabla_estratificada(d, pred_est)
    est.round(4).to_csv(config.RUTA_SALIDAS / "metricas_optimizadas.csv", index=False)
    print(est.round(3).to_string(index=False))

    print("\nValidacion anidada LOMO (un municipio completo fuera):")
    pred_lomo = evaluacion_lomo(d, cols)
    lomo = tabla_lomo(d, pred_lomo)
    lomo.round(4).to_csv(config.RUTA_SALIDAS / "metricas_lomo.csv", index=False)
    print(lomo.round(3).to_string(index=False))
    tabla_robustez(d, pred_lomo).round(4).to_csv(
        config.RUTA_SALIDAS / "robustez_candidatos.csv", index=False)

    n_vars = {"ensamble": len(cols), "ubicacion": 1, "mezcla": len(cols) + 1}
    comp = tabla_comparacion(est, lomo, n_vars)
    comp.round(4).to_csv(config.RUTA_SALIDAS / "comparacion_candidatos.csv", index=False)

    # Prediccion fuera de muestra por parcela (para la grafica observado vs predicho)
    oof = pd.DataFrame({"ID_POLIGONO": d.ids, "municipio": d.municipios, "y": d.y})
    for m in MIEMBROS:
        oof[f"estrat_{m}"] = np.nanmean(pred_est[m], axis=0)
        oof[f"lomo_{m}"] = pred_lomo[m]
    oof.round(4).to_csv(config.RUTA_SALIDAS / "oof_predicciones.csv", index=False)

    e = est.set_index("modelo").to_dict("index")
    l = lomo.set_index("modelo").to_dict("index")
    sugerido, motivo = recomendar(e, l)
    print(f"\nCuantil 90 del error absoluto (para intervalos): "
          + ", ".join(f"{c} {q90[c]:.3f}" for c in config.CANDIDATOS) + " t/ha")
    print(f"Recomendacion con la regla de ruido: {sugerido}. {motivo}")
    print(f"Modelo congelado (CEBADATA_MODELO_FINAL): {config.MODELO_FINAL}"
          + ("" if config.MODELO_FINAL == sugerido else "  <- distinto de la recomendacion"))

    modelo_final(d, cols, pred_est, q90)


if __name__ == "__main__":
    main()
