"""Carga en la base de datos los resultados del pipeline (carpeta salidas/).

Uso:
    python manage.py migrate
    python manage.py cargar_resultados

Lee las tablas que escriben los pasos 01 a 09 y deja listo lo que sirve la API:
parcelas con su prediccion y su intervalo, clima del ciclo, curvas satelitales
(si existen satelital_s2.csv y satelital_planet.csv), explicaciones por parcela
y el resumen de metricas del modelo. Se puede repetir las veces que haga falta.
"""
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from api import etiquetas
from api.models import (ClimaMensual, Corrida, Explicacion, Parcela, Prediccion,
                        SerieSatelital)
from pipeline import conf
from pipeline.modelado import cargar_datos

REQUERIDOS = [
    "parcelas_maestro.csv", "topografia_parcela.csv", "clima_mensual_parcela.csv",
    "predicciones_parcelas.csv", "comparacion_candidatos.csv", "metricas_optimizadas.csv",
    "metricas_lomo.csv", "robustez_candidatos.csv", "oof_predicciones.csv",
    "shap_importancia.csv", "ablacion_bloques.csv", "shap_parcela.csv", "metricas_baseline.csv",
]
INDICES_SERIE = ["ndvi", "evi", "lai", "nddi", "ndwi"]
FUENTES = {"satelital_s2.csv": "S2", "satelital_planet.csv": "Planet"}

NOMBRES = {
    "lgbm": "LightGBM", "rf": "Random Forest", "enet": "Elastic Net", "ensamble": "Ensamble",
    "ubicacion": "Solo ubicación", "mezcla": "Mezcla",
}
DETALLE = {
    "ensamble": "Promedio de LightGBM, Random Forest y Elastic Net",
    "ubicacion": "LightGBM que solo usa la longitud",
    "mezcla": "Promedio del ensamble y el modelo de ubicación",
}
DESCRIPCION = {
    "ensamble": "LightGBM, Random Forest y Elastic Net con todas las variables.",
    "ubicacion": "LightGBM que solo usa la longitud. Es lo que encontró el análisis de modelos clásicos.",
    "mezcla": "Promedio de las predicciones del ensamble y del modelo de ubicación.",
}


def r(x, d=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


class Command(BaseCommand):
    help = "Carga en la base de datos los resultados del pipeline"

    def handle(self, *args, **opciones):
        s = conf.RUTA_SALIDAS
        faltan = [n for n in REQUERIDOS if not (s / n).exists()]
        if faltan:
            raise CommandError(
                "Faltan salidas del pipeline: " + ", ".join(faltan)
                + ". Corre `python manage.py run_pipeline` (o desde el paso 07) primero.")

        final = conf.MODELO_FINAL
        maestro = pd.read_csv(s / "parcelas_maestro.csv").set_index("ID_POLIGONO")
        topo = pd.read_csv(s / "topografia_parcela.csv").set_index("ID_POLIGONO")
        pred = pd.read_csv(s / "predicciones_parcelas.csv").set_index("ID_POLIGONO")
        hp = pd.read_json(s / "hiperparametros_finales.json", typ="series") \
            if (s / "hiperparametros_finales.json").exists() else None
        if hp is not None and hp.get("modelo_final") not in (None, final):
            self.stdout.write(self.style.WARNING(
                f"El pipeline congelo '{hp['modelo_final']}' pero CEBADATA_MODELO_FINAL es '{final}'. "
                "Las predicciones guardadas son las de la corrida; repite el paso 07 para cambiarlo."))
            final = hp["modelo_final"]

        datos = cargar_datos()
        base = float(datos.y.mean())

        with transaction.atomic():
            for modelo in (Explicacion, SerieSatelital, ClimaMensual, Prediccion, Parcela, Corrida):
                modelo.objects.all().delete()

            # ---- parcelas y predicciones ----
            parcelas = []
            for pid, f in maestro.iterrows():
                obs = None if pd.isna(f["RENDIMIENTO_T_HA"]) else float(f["RENDIMIENTO_T_HA"])
                parcelas.append(Parcela(
                    id_poligono=pid, estado=f["Estado"], municipio=f["Municipio"], area_ha=f["AREA_HA"],
                    conjunto=f["CONJUNTO"], rendimiento_obs=obs, lat=f["lat"], lon=f["lon"],
                    elevacion_m=topo.loc[pid, "elevacion_m"], pendiente_grados=topo.loc[pid, "pendiente_grados"]))
            Parcela.objects.bulk_create(parcelas)

            q90 = {}
            if hp is not None and "q90_por_candidato" in hp:
                q90 = dict(hp["q90_por_candidato"])
            predicciones = []
            for pid, f in pred.iterrows():
                for c in conf.CANDIDATOS:
                    medio = q90.get(c)
                    if medio is None:  # corrida antigua: usa el intervalo del modelo final
                        medio = (f["intervalo_sup"] - f["intervalo_inf"]) / 2
                    predicciones.append(Prediccion(
                        parcela_id=pid, modelo=c, valor=f[f"pred_{c}"], inferior=f[f"pred_{c}"] - medio,
                        superior=f[f"pred_{c}"] + medio, fuera_de_muestra=bool(f["fuera_de_muestra"])))
            Prediccion.objects.bulk_create(predicciones)

            # ---- clima del ciclo 2025 ----
            clima = pd.read_csv(s / "clima_mensual_parcela.csv")
            clima = clima[(clima["anio"] == 2025) & clima["mes"].between(4, 10)]
            ClimaMensual.objects.bulk_create([
                ClimaMensual(parcela_id=c.ID_POLIGONO, mes=int(c.mes), prec_mm=c.prec_mm,
                             tmax_c=c.tmax_c, tmin_c=c.tmin_c) for c in clima.itertuples()])

            # ---- curvas satelitales (opcionales) ----
            n_serie = self._series(s)

            # ---- explicaciones ----
            n_exp = self._explicaciones(s, pred, final, base)

            # ---- metricas ----
            Corrida.objects.create(
                modelo_final=final, base=round(base, 3),
                metricas=self._metricas(s, final, datos),
                rango_entrenamiento=self._rango(datos))

        self.stdout.write(self.style.SUCCESS(
            f"Listo: {len(parcelas)} parcelas, {len(predicciones)} predicciones, {len(clima)} filas de clima, "
            f"{n_serie} observaciones satelitales, {n_exp} explicaciones. Modelo final: {final}."))
        if not n_serie:
            self.stdout.write(self.style.WARNING(
                "Sin curvas satelitales: faltan salidas/satelital_s2.csv y satelital_planet.csv "
                "(los generan los pasos 01 a 03 con el dataset del reto). La ficha mostrara 'Sin observaciones'."))

    # ------------------------------------------------------------------
    def _series(self, s) -> int:
        total = 0
        for archivo, fuente in FUENTES.items():
            ruta = s / archivo
            if not ruta.exists():
                continue
            cols = ["ID_POLIGONO", "fecha", "limpia"] + [f"{i}_promedio" for i in INDICES_SERIE]
            df = pd.read_csv(ruta, low_memory=False, usecols=lambda c: c in cols, parse_dates=["fecha"])
            df = df[df["limpia"] & df["fecha"].between("2025-03-15", "2025-11-15")]
            filas = []
            for indice in INDICES_SERIE:
                col = f"{indice}_promedio"
                if col not in df.columns:
                    continue
                sub = df.dropna(subset=[col])
                filas.extend(SerieSatelital(parcela_id=x.ID_POLIGONO, fecha=x.fecha.date(), fuente=fuente,
                                            indice=indice, valor=round(float(getattr(x, col)), 4))
                             for x in sub.itertuples())
            SerieSatelital.objects.bulk_create(filas, batch_size=5000)
            total += len(filas)
        return total

    def _explicaciones(self, s, pred, final, base) -> int:
        shap = pd.read_csv(s / "shap_parcela.csv")
        explicaciones = []
        for pid, g in shap.groupby("ID_POLIGONO"):
            items = []
            for f in g.sort_values("orden").head(8).itertuples():
                valor, unidad = etiquetas.valor_y_unidad(f.variable, f.valor_variable)
                items.append({"variable": f.variable, "etiqueta": etiquetas.etiqueta(f.variable),
                              "valor_variable": round(valor, 5), "unidad": unidad,
                              "contribucion": round(f.contribucion, 3)})
            explicaciones.append(Explicacion(parcela_id=pid, items=items))
        Explicacion.objects.bulk_create(explicaciones)
        return len(explicaciones)

    def _rango(self, datos) -> dict:
        x = datos.X
        return {
            "lluvia_mayo": {"min": float(x["prec_mm_m05"].min()), "max": float(x["prec_mm_m05"].max())},
            "tmin_abril": {"min": float(x["tmin_c_m04"].min()), "max": float(x["tmin_c_m04"].max())},
        }

    # ------------------------------------------------------------------
    def _metricas(self, s, final, datos) -> dict:
        est = pd.read_csv(s / "metricas_optimizadas.csv").set_index("modelo")
        lomo = pd.read_csv(s / "metricas_lomo.csv").set_index("modelo")
        comp = pd.read_csv(s / "comparacion_candidatos.csv")
        base_rmse = float(pd.read_csv(s / "metricas_baseline.csv").set_index("modelo").loc["media_global", "rmse"])
        maestro = pd.read_csv(s / "parcelas_maestro.csv").set_index("ID_POLIGONO")

        # Modelos probados, con el RMSE de la validacion anidada estratificada
        modelos = [{"nombre": "Promedio (referencia)", "rmse": r(base_rmse), "referencia": True}]
        for clave in ["enet", "rf", "lgbm", "ensamble", "ubicacion", "mezcla"]:
            modelos.append({"nombre": NOMBRES[clave], "clave": clave, "rmse": r(est.loc[clave, "rmse"]),
                            "elegido": clave == final, "detalle": DETALLE.get(clave)})

        candidatos = []
        for c in conf.CANDIDATOS:
            fila = {"id": c, "nombre": NOMBRES[c], "descripcion": DESCRIPCION[c], "elegido": c == final}
            for v in ("estratificada", "lomo"):
                x = comp[(comp.candidato == c) & (comp.validacion == v)].iloc[0]
                fila[v] = {"rmse": r(x.rmse), "mae": r(x.mae), "r2": r(x.r2, 2), "sesgo": r(x.sesgo)}
            fila["n_variables"] = int(comp[comp.candidato == c].iloc[0].n_variables)
            candidatos.append(fila)

        # Dispersion y residuales: prediccion fuera de muestra del modelo congelado
        oof = pd.read_csv(s / "oof_predicciones.csv")
        dispersion = [{"id": f.ID_POLIGONO, "estado": maestro.loc[f.ID_POLIGONO, "Estado"],
                       "municipio": f.municipio, "obs": r(f.y, 2), "pred": r(getattr(f, f"estrat_{final}"), 2)}
                      for f in oof.itertuples()]
        residuales = [r(d["obs"] - d["pred"]) for d in dispersion]

        # Robustez espacial (LOMO): solo municipios con 5 o mas parcelas dan una medida estable
        rob = pd.read_csv(s / "robustez_candidatos.csv")
        rob = rob[rob.n_parcelas >= 5]
        elegido_rob = rob[rob.modelo == final].sort_values("rmse")
        robustez = [{"municipio": f.municipio_fuera, "rmse": r(f.rmse, 2), "n": int(f.n_parcelas)}
                    for f in elegido_rob.itertuples()]
        por_municipio = []
        for mun, g in rob.groupby("municipio_fuera"):
            fila = {"municipio": mun, "n": int(g.n_parcelas.iloc[0])}
            for c in conf.CANDIDATOS + ["media_resto"]:
                fila[c] = r(g[g.modelo == c].rmse.iloc[0], 2)
            por_municipio.append(fila)
        por_municipio.sort(key=lambda x: x[final])

        imp = pd.read_csv(s / "shap_importancia.csv")
        familias = imp.groupby("familia")["shap_medio_abs"].sum().sort_values(ascending=False)
        importancia = [{"familia": etiquetas.familia_nombre(f), "valor": r(v)} for f, v in familias.head(10).items()]

        return {
            "modelo_final": final,
            "validacion": {
                "esquema": "Validación cruzada anidada",
                "rmse": r(est.loc[final, "rmse"]), "mae": r(est.loc[final, "mae"]), "r2": r(est.loc[final, "r2"], 2),
                "baseline": {"nombre": "Predecir el promedio", "rmse": r(base_rmse), "r2": 0},
            },
            "modelos": modelos,
            "candidatos": candidatos,
            "lomo_media_resto": {"rmse": r(lomo.loc["media_resto", "rmse"]), "nombre": "Promedio de las demás zonas"},
            "dispersion": dispersion,
            "residuales": residuales,
            "importancia_familias": importancia,
            "robustez_espacial": robustez,
            "robustez_candidatos": por_municipio,
            "hallazgos": self._hallazgos(s, imp, familias),
            "n_parcelas": {"total": int(len(maestro)), "medidas": int(len(datos.y)),
                           "ocultas": int(len(maestro) - len(datos.y))},
        }

    def _hallazgos(self, s, imp, familias) -> dict:
        sat = familias[[f for f in familias.index if f.startswith("satelital:")]]
        indices = [{"etiqueta": etiquetas.familia_nombre(f), "valor": r(v)} for f, v in sat.head(3).items()]

        ndvi = imp[imp.familia == "satelital:ndvi"].copy()

        def grupo(v):
            if "tasa_verdeo" in v:
                return "Velocidad de verdeo"
            if "tasa_senescencia" in v:
                return "Velocidad de senescencia"
            if v in ("ndvi_pico", "ndvi_pico_pl"):
                return "Valor máximo"
            return "Otras medidas"

        ndvi["grupo"] = ndvi.variable.map(grupo)
        dentro = ndvi.groupby("grupo")["shap_medio_abs"].sum()
        orden = ["Velocidad de verdeo", "Velocidad de senescencia", "Valor máximo", "Otras medidas"]
        dentro_ndvi = [{"etiqueta": g, "valor": r(dentro.get(g, 0.0))} for g in orden]

        clima = imp[imp.familia == "clima"].sort_values("shap_medio_abs", ascending=False)
        top = clima.head(6)
        clima_items = [{"etiqueta": etiquetas.CORTAS.get(f.variable, etiquetas.etiqueta(f.variable)),
                        "valor": r(f.shap_medio_abs)} for f in top.itertuples()]
        clima_items.append({"etiqueta": "Resto del clima", "valor": r(clima.iloc[6:].shap_medio_abs.sum())})

        abla = pd.read_csv(s / "ablacion_bloques.csv")
        bloques = [{"etiqueta": etiquetas.BLOQUES.get(f.bloque, f.bloque), "clave": f.bloque,
                    "rmse": r(f.rmse), "n_variables": int(f.n_variables)} for f in abla.itertuples()]
        return {"indices": indices, "dentro_ndvi": dentro_ndvi, "clima": clima_items, "bloques": bloques,
                "familia_clima": r(familias.get("clima", 0.0), 3)}
