"""Paso 01: tabla maestra de parcelas.

Une el shapefile con el CSV de rendimiento y deja una sola tabla limpia
con ID, geometria de referencia (centroide), area, municipio, estado,
conjunto y rendimiento. Tambien exporta el centroide en lat/lon porque
los pasos de clima lo necesitan.

Salidas:
  salidas/parcelas_maestro.csv
"""
import geopandas as gpd
import pandas as pd

from pipeline import conf as config


def cargar_parcelas() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(config.SHAPEFILE)
    # El shapefile trunca el nombre del campo a 10 caracteres (ID_POLIGON);
    # lo renombramos para que la llave sea identica en todas las tablas.
    gdf = gdf.rename(columns={"ID_POLIGON": "ID_POLIGONO", "área_ha": "AREA_HA_SHP"})
    return gdf


def cargar_rendimiento() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_RENDIMIENTO)
    # El CSV trae una columna vacia sin nombre (Unnamed: 4); se descarta.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def construir_maestro() -> pd.DataFrame:
    gdf = cargar_parcelas()
    rend = cargar_rendimiento()

    maestro = gdf.merge(rend, on="ID_POLIGONO", how="inner", suffixes=("_shp", ""))

    # Centroides en lat/lon para el muestreo de rasters climaticos.
    # Se calculan en CRS metrico y se reproyectan, para que el centroide
    # sea geometricamente correcto (y sin warnings de geopandas).
    centroides = maestro.geometry.to_crs(config.CRS_METRICO).centroid.to_crs(maestro.crs)
    maestro["lon"] = centroides.x
    maestro["lat"] = centroides.y

    columnas = ["ID_POLIGONO", "CONJUNTO", "Estado", "Municipio",
                "AREA_HA", "RENDIMIENTO_T_HA", "lon", "lat"]
    tabla = pd.DataFrame(maestro[columnas])

    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(config.RUTA_SALIDAS / "parcelas_maestro.csv", index=False)
    return tabla


def main():
    t = construir_maestro()
    print(f"parcelas_maestro.csv: {len(t)} parcelas "
          f"({(t.CONJUNTO == 'ENTRENAMIENTO').sum()} entrenamiento, "
          f"{(t.CONJUNTO == 'PREDICCION').sum()} prediccion)")


if __name__ == "__main__":
    main()
