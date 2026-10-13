"""Paso 08: interpretacion del modelo y robustez espacial.

Tres analisis que alimentan el reporte y las preguntas guia del reto:

  1. SHAP sobre el LightGBM final (hiperparametros del paso 07): importancia
     global por variable y agrupada por familia (satelital por indice,
     climatica por tipo, topografia, ubicacion). Responde que indice y que
     variable climatica pesan mas.
  2. Ablacion por bloques: se evalua el mismo modelo con subconjuntos de
     variables (solo clima, solo satelital, solo topografia y ubicacion, y
     combinaciones) con la misma CV estratificada. Responde que combinacion
     de variables funciona mejor.
  3. Robustez espacial: validacion dejando municipios completos fuera, que
     mide que pasaria en zonas nuevas (esperamos degradacion; se reporta
     por honestidad metodologica, el conjunto oculto NO esta en esa
     situacion).

La comparacion de candidatos (ensamble, ubicacion y mezcla) con validacion
estratificada y LOMO vive en el paso 07; aqui se resume en las respuestas.

Salidas:
  salidas/shap_importancia.csv          salidas/fig_shap_resumen.png
  salidas/ablacion_bloques.csv          salidas/fig_shap_familias.png
  salidas/robustez_espacial.csv
  salidas/respuestas_preguntas_guia.txt
"""
import json
import re
import warnings

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold

from pipeline import conf as config

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
AZUL, NARANJA, VERDE, GRIS = "#2563EB", "#EA580C", "#059669", "#6B7280"
SEMILLA = 42

INDICES = ["ndvi", "evi", "lai", "fapar", "savi", "msi", "ndwi", "dswi2", "dswi5",
           "nddi", "ndti", "crc", "mcrc", "ndsvi", "msavi"]


def cargar():
    df = pd.read_csv(config.RUTA_SALIDAS / "features_modelado.csv")
    df = pd.get_dummies(df, columns=["Estado", "Municipio"], prefix=["est", "mun"])
    entrena = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
    y = entrena["RENDIMIENTO_T_HA"].values
    X = entrena.drop(columns=["ID_POLIGONO", "CONJUNTO", "RENDIMIENTO_T_HA"])

    with open(config.RUTA_SALIDAS / "hiperparametros_finales.json") as f:
        hp = json.load(f)
    cols = hp["variables"]
    params = {k: v for k, v in hp["lgbm"].items()}

    maestro = pd.read_csv(config.RUTA_SALIDAS / "parcelas_maestro.csv")
    mun = maestro.set_index("ID_POLIGONO").loc[entrena["ID_POLIGONO"], "Municipio"].values
    conteo = pd.Series(mun).value_counts()
    estrato = np.where(pd.Series(mun).map(conteo) >= 5, mun, "OTRO")
    return X, y, cols, params, mun, estrato


def familia(col: str) -> str:
    if col.startswith(("prec_", "tmax_", "tmin_", "gdd_", "meses_secos", "amplitud")):
        return "clima"
    if col.startswith(("est_", "mun_")) or col in ("lon", "lat"):
        return "ubicacion"
    if col in ("elevacion_m", "pendiente_grados"):
        return "topografia"
    if col == "AREA_HA":
        return "area"
    if col.startswith("suelo_"):
        return "suelo"
    if col.startswith("ndvi_auc_"):
        return "historia_ndvi"
    for idx in INDICES:
        if re.match(rf"^{idx}_", col):
            return f"satelital:{idx}"
    return "otra"


def analisis_shap(X, y, cols, params):
    Xn = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X[cols].astype(float)),
                      columns=cols)
    modelo = lgb.LGBMRegressor(**params).fit(Xn, y)
    valores = shap.TreeExplainer(modelo).shap_values(Xn)

    imp = pd.DataFrame({"variable": cols,
                        "shap_medio_abs": np.abs(valores).mean(axis=0)})
    imp["familia"] = imp["variable"].map(familia)
    imp = imp.sort_values("shap_medio_abs", ascending=False)
    imp.round(5).to_csv(config.RUTA_SALIDAS / "shap_importancia.csv", index=False)

    # Figura 1: top 15 variables individuales
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["variable"], top["shap_medio_abs"], color=AZUL, height=0.62)
    ax.set_xlabel("Media de |SHAP| (t/ha)")
    ax.set_title("Contribucion de las 15 variables mas influyentes (LightGBM final)")
    plt.tight_layout()
    plt.savefig(config.RUTA_SALIDAS / "fig_shap_resumen.png", bbox_inches="tight")
    plt.close()

    # Figura 2: importancia agregada por familia
    fam = imp.groupby("familia")["shap_medio_abs"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colores = [VERDE if f.startswith("satelital") else
               NARANJA if f == "clima" else
               AZUL if f == "ubicacion" else GRIS for f in fam.index]
    ax.barh(fam.index, fam.values, color=colores, height=0.62)
    ax.set_xlabel("Suma de |SHAP| por familia (t/ha)")
    ax.set_title("Peso agregado por familia de variables")
    plt.tight_layout()
    plt.savefig(config.RUTA_SALIDAS / "fig_shap_familias.png", bbox_inches="tight")
    plt.close()
    return imp


def rmse_cv(Xn, y, estrato, params, n_repeats=3):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEMILLA)
    errores = []
    for tr, te in cv.split(Xn, estrato):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xn[tr], y[tr])
        errores.append(np.sqrt(mean_squared_error(y[te], m.predict(Xn[te]))))
    return float(np.mean(errores)), float(np.std(errores))


def ablacion(X, y, cols, params, estrato):
    fams = pd.Series({c: familia(c) for c in cols})
    bloques = {
        "solo_clima": fams[fams == "clima"].index.tolist(),
        "solo_satelital": fams[fams.str.startswith("satelital")].index.tolist(),
        "solo_topografia_ubicacion": fams[fams.isin(["topografia", "ubicacion"])].index.tolist(),
        "clima_mas_satelital": fams[fams.isin(["clima"]) | fams.str.startswith("satelital")].index.tolist(),
        "clima_mas_ubicacion": fams[fams.isin(["clima", "ubicacion", "topografia"])].index.tolist(),
        "todo": cols,
    }
    # Bloques de suelo solo si el paso 05b ya se corrio y hay columnas suelo_*
    suelo = fams[fams == "suelo"].index.tolist()
    if suelo:
        bloques["solo_suelo"] = suelo
        bloques["todo_sin_suelo"] = [c for c in cols if c not in suelo]
    imputador = SimpleImputer(strategy="median")
    filas = []
    for nombre, subcols in bloques.items():
        Xn = imputador.fit_transform(X[subcols].astype(float))
        rmse, std = rmse_cv(Xn, y, estrato, params)
        filas.append({"bloque": nombre, "n_variables": len(subcols),
                      "rmse": rmse, "rmse_std": std})
        print(f"  {nombre:28s} ({len(subcols):3d} vars) RMSE {rmse:.3f} +/- {std:.3f}")
    tabla = pd.DataFrame(filas).sort_values("rmse")
    tabla.round(4).to_csv(config.RUTA_SALIDAS / "ablacion_bloques.csv", index=False)
    return tabla


def robustez_espacial(X, y, cols, params, municipios):
    Xn = SimpleImputer(strategy="median").fit_transform(X[cols].astype(float))
    logo = LeaveOneGroupOut()
    filas = []
    for tr, te in logo.split(Xn, y, groups=municipios):
        grupo = municipios[te][0]
        if len(te) < 5:
            continue  # municipios con 1 a 4 parcelas no dan una medida estable
        m = lgb.LGBMRegressor(**params).fit(Xn[tr], y[tr])
        rmse = np.sqrt(mean_squared_error(y[te], m.predict(Xn[te])))
        filas.append({"municipio_fuera": grupo, "n_parcelas": len(te), "rmse": rmse})
        print(f"  sin {grupo:28s} (n={len(te):3d}) RMSE {rmse:.3f}")
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(config.RUTA_SALIDAS / "robustez_espacial.csv", index=False)
    return tabla


def seccion_candidatos() -> str:
    ruta = config.RUTA_SALIDAS / "comparacion_candidatos.csv"
    if not ruta.exists():
        return ""
    c = pd.read_csv(ruta)
    lineas = []
    for r in c.itertuples():
        lineas.append(f"     - {r.candidato:10s} {r.validacion:13s} RMSE {r.rmse:.3f}  MAE {r.mae:.3f}  "
                      f"R2 {r.r2:.2f}  sesgo {r.sesgo:+.3f}")
    return ("\nComparacion de candidatos (paso 07, validacion anidada, mismas parcelas):\n"
            + "\n".join(lineas) + "\n"
            "   La estratificada mide lo que se califica (las 59 ocultas estan en los mismos\n"
            "   municipios). LOMO mide el desempeno en una zona nueva de la region. Diferencias\n"
            f"   menores a {config.UMBRAL_RUIDO_RMSE:.3f} t/ha estan dentro del ruido de la validacion.\n")


def escribir_respuestas(imp, abla, robu):
    sat = imp[imp["familia"].str.startswith("satelital")]
    idx_top = sat.groupby("familia")["shap_medio_abs"].sum().sort_values(ascending=False)
    clima = imp[imp["familia"] == "clima"].head(5)
    mejor_bloque = abla.iloc[0]

    texto = f"""Respuestas a las preguntas guia del Reto AgroCebada (generadas por paso08)
================================================================================

1. Que indice satelital es mas importante en la prediccion?
   Sumando la contribucion SHAP de todas las variables derivadas de cada
   indice, el mas influyente es {idx_top.index[0].split(':')[1].upper()}
   (suma |SHAP| = {idx_top.iloc[0]:.3f} t/ha), seguido de
   {idx_top.index[1].split(':')[1].upper()} ({idx_top.iloc[1]:.3f}) y
   {idx_top.index[2].split(':')[1].upper()} ({idx_top.iloc[2]:.3f}).
   Nota del EDA: el valor pico de los indices casi no discrimina; lo que
   aporta son sus dinamicas (tasas de verdeo y senescencia, integral).

2. Que variable climatica tiene mayor peso?
   Las variables climaticas individuales con mayor |SHAP| son:
{chr(10).join(f'     - {r.variable}: {r.shap_medio_abs:.3f} t/ha' for r in clima.itertuples())}

3. Que combinacion de variables funciona mejor?
   Ablacion por bloques (RMSE con la misma CV estratificada, mismo modelo):
{chr(10).join(f'     - {r.bloque}: RMSE {r.rmse:.3f} ({r.n_variables} variables)' for r in abla.itertuples())}
   El mejor bloque es "{mejor_bloque.bloque}" con RMSE {mejor_bloque.rmse:.3f}.

Robustez espacial (dejar un municipio completo fuera):
{chr(10).join(f'     - sin {r.municipio_fuera}: RMSE {r.rmse:.3f} (n={r.n_parcelas})' for r in robu.itertuples())}
   Como se anticipo, el desempeno se degrada al extrapolar a municipios no
   vistos porque parte de la senal es regional. El conjunto de PREDICCION
   del reto esta en los mismos municipios que el de entrenamiento, asi que
   la CV estratificada es la estimacion representativa; esta tabla se
   reporta como analisis de limites del modelo.
{seccion_candidatos()}"""
    (config.RUTA_SALIDAS / "respuestas_preguntas_guia.txt").write_text(texto)
    print(texto)


def main():
    X, y, cols, params, municipios, estrato = cargar()
    print("SHAP...")
    imp = analisis_shap(X, y, cols, params)
    print("Ablacion por bloques:")
    abla = ablacion(X, y, cols, params, estrato)
    print("Robustez espacial:")
    robu = robustez_espacial(X, y, cols, params, municipios)
    escribir_respuestas(imp, abla, robu)


if __name__ == "__main__":
    main()
