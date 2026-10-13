"""Paso 03: extraccion de clima y topografia por parcela.

Decisiones metodologicas (documentar en el reporte):
  1. Clima (CHIRPS y CHIRTS, pixel ~5 km): se muestrea en el centroide de
     cada parcela. La mediana de area de parcela es 5.5 ha (~230 m de
     lado), es decir, cada parcela cae completa dentro de un solo pixel
     climatico, asi que una media zonal seria identica al centroide y
     solo agregaria costo.
  2. Topografia (CEM de INEGI, pixel 120 m): aqui la parcela si abarca
     varios pixeles, por eso se calcula media zonal con all_touched=True
     (cuenta todo pixel tocado por el poligono, importante en parcelas
     chicas) en el CRS nativo del raster (EPSG:6372).

Salidas:
  salidas/clima_mensual_parcela.csv  (formato largo: parcela, anio, mes, prec, tmax, tmin)
  salidas/topografia_parcela.csv     (elevacion y pendiente medias por parcela)
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as mascara_zonal

from pipeline import conf as config


def _muestrear_centroides(ruta, coords):
    with rasterio.open(ruta) as src:
        valores = np.array([v[0] for v in src.sample(coords)], dtype=float)
        valores[valores == src.nodata] = np.nan
    return valores


def extraer_clima() -> pd.DataFrame:
    parcelas = pd.read_csv(config.RUTA_SALIDAS / "parcelas_maestro.csv")
    coords = list(zip(parcelas["lon"], parcelas["lat"]))

    registros = []
    for anio in config.ANIOS_CLIMA:
        for mes in range(1, 13):
            prec = _muestrear_centroides(
                config.DIR_PRECIPITACION / str(anio) / f"PREC_{anio}_{mes:02d}.tif", coords)
            tmax = _muestrear_centroides(
                config.DIR_TEMPERATURA / "Tmax" / str(anio) / f"Tmax_{anio}_{mes:02d}.tif", coords)
            tmin = _muestrear_centroides(
                config.DIR_TEMPERATURA / "Tmin" / str(anio) / f"Tmin_{anio}_{mes:02d}.tif", coords)
            registros.append(pd.DataFrame({
                "ID_POLIGONO": parcelas["ID_POLIGONO"],
                "anio": anio, "mes": mes,
                "prec_mm": prec, "tmax_c": tmax, "tmin_c": tmin,
            }))

    clima = pd.concat(registros, ignore_index=True)
    clima.to_csv(config.RUTA_SALIDAS / "clima_mensual_parcela.csv", index=False)
    return clima


def extraer_topografia() -> pd.DataFrame:
    gdf = gpd.read_file(config.SHAPEFILE).rename(columns={"ID_POLIGON": "ID_POLIGONO"})
    gdf = gdf.to_crs(config.CRS_METRICO)

    resultado = {"ID_POLIGONO": gdf["ID_POLIGONO"].tolist()}
    capas = {
        "elevacion_m": config.DIR_TOPOGRAFIA / "Elevacion_INEGI_CEM4_120m.tif",
        "pendiente_grados": config.DIR_TOPOGRAFIA / "Pendiente_INEGI_CEM4_120m_grados.tif",
    }
    for nombre, ruta in capas.items():
        valores = []
        with rasterio.open(ruta) as src:
            for geom in gdf.geometry:
                arreglo, _ = mascara_zonal(src, [geom.__geo_interface__],
                                           crop=True, all_touched=True, nodata=src.nodata)
                datos = arreglo[0]
                datos = datos[datos != src.nodata]
                valores.append(float(datos.mean()) if datos.size else np.nan)
        resultado[nombre] = valores

    topo = pd.DataFrame(resultado)
    topo.to_csv(config.RUTA_SALIDAS / "topografia_parcela.csv", index=False)
    return topo


def main():
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    clima = extraer_clima()
    print(f"clima_mensual_parcela.csv: {len(clima):,} filas "
          f"({clima.ID_POLIGONO.nunique()} parcelas x {len(config.ANIOS_CLIMA)} anios x 12 meses), "
          f"nulos: {int(clima[['prec_mm', 'tmax_c', 'tmin_c']].isna().sum().sum())}")
    topo = extraer_topografia()
    print(f"topografia_parcela.csv: {len(topo)} parcelas, "
          f"nulos: {int(topo[['elevacion_m', 'pendiente_grados']].isna().sum().sum())}")


if __name__ == "__main__":
    main()
