"""Paso 04: validaciones de integridad del pipeline.

Corre despues de los pasos 01 a 03 y verifica que las salidas cumplan lo
que el resto del proyecto asume. Si algo falla, truena con AssertionError
y un mensaje claro: mejor que fallar silenciosamente a mitad del modelado.

Escribe un resumen legible en salidas/reporte_validacion.txt.
"""
import pandas as pd

from pipeline import conf as config

LINEAS = []


def check(condicion: bool, mensaje: str):
    estado = "OK " if condicion else "FALLA"
    LINEAS.append(f"[{estado}] {mensaje}")
    assert condicion, mensaje


def main():
    s = config.RUTA_SALIDAS
    maestro = pd.read_csv(s / "parcelas_maestro.csv")
    s2 = pd.read_csv(s / "satelital_s2.csv", low_memory=False)
    landsat = pd.read_csv(s / "satelital_landsat.csv")
    planet = pd.read_csv(s / "satelital_planet.csv")
    clima = pd.read_csv(s / "clima_mensual_parcela.csv")
    topo = pd.read_csv(s / "topografia_parcela.csv")

    ids = set(maestro.ID_POLIGONO)

    # --- Maestro ---------------------------------------------------------
    check(len(maestro) == config.TOTAL_PARCELAS,
          f"maestro tiene {config.TOTAL_PARCELAS} parcelas")
    check(maestro.ID_POLIGONO.is_unique, "IDs unicos en maestro")
    check((maestro.CONJUNTO == "ENTRENAMIENTO").sum() == config.N_ENTRENAMIENTO,
          f"{config.N_ENTRENAMIENTO} parcelas de entrenamiento")
    check((maestro.CONJUNTO == "PREDICCION").sum() == config.N_PREDICCION,
          f"{config.N_PREDICCION} parcelas de prediccion")
    ent = maestro[maestro.CONJUNTO == "ENTRENAMIENTO"]
    pre = maestro[maestro.CONJUNTO == "PREDICCION"]
    check(ent.RENDIMIENTO_T_HA.notna().all(),
          "todas las parcelas de entrenamiento tienen rendimiento")
    check(pre.RENDIMIENTO_T_HA.isna().all(),
          "ninguna parcela de prediccion tiene rendimiento (esta oculto)")
    check(ent.RENDIMIENTO_T_HA.between(1.5, 6.5).all(),
          "rendimientos en rango plausible (1.5 a 6.5 t/ha)")

    # --- Satelital -------------------------------------------------------
    for nombre, df in [("s2", s2), ("landsat", landsat), ("planet", planet)]:
        check(set(df.ID_POLIGONO) == ids, f"{nombre}: las 197 parcelas presentes")
        check(not df.duplicated(["ID_POLIGONO", "fecha", "sensor"]).any(),
              f"{nombre}: sin observaciones duplicadas por parcela-fecha")
    check("vi6t_promedio" not in s2.columns, "s2 no arrastra columnas de VI6T")
    limpio_s2 = s2[s2.limpia]
    check(limpio_s2.ndvi_promedio.isna().mean() < 0.01,
          "tras filtrar nubosidad, menos de 1 % de NDVI nulo en S2")
    ciclo = s2[s2.limpia & s2.en_ciclo_2025].groupby("ID_POLIGONO").size()
    check(len(ciclo) == config.TOTAL_PARCELAS and ciclo.min() >= 10,
          "toda parcela tiene al menos 10 obs S2 limpias dentro del ciclo 2025")

    # --- Clima y topografia ---------------------------------------------
    check(len(clima) == config.TOTAL_PARCELAS * len(config.ANIOS_CLIMA) * 12,
          "clima completo: 197 parcelas x 4 anios x 12 meses")
    check(clima[["prec_mm", "tmax_c", "tmin_c"]].notna().all().all(),
          "clima sin valores faltantes")
    check(clima.prec_mm.between(0, 1000).all(), "precipitacion mensual en rango (0 a 1000 mm)")
    check(clima.tmax_c.between(5, 40).all() and clima.tmin_c.between(-5, 25).all(),
          "temperaturas en rango plausible")
    check(set(topo.ID_POLIGONO) == ids and topo.notna().all().all(),
          "topografia completa para las 197 parcelas")
    check(topo.elevacion_m.between(2000, 3200).all(),
          "elevaciones en rango del altiplano (2000 a 3200 m)")

    reporte = "\n".join(LINEAS)
    (s / "reporte_validacion.txt").write_text(reporte + "\n")
    print(reporte)
    print(f"\nTodo en orden: {len(LINEAS)} validaciones pasaron.")


if __name__ == "__main__":
    main()
