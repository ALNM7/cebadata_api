"""Configuracion central del pipeline del Reto AgroCebada 2026.

El pipeline NO forma parte de la API: sirve para reproducir el reporte y para
regenerar la carpeta artefactos/ que carga el backend. No depende de Django.

Variables de entorno opcionales:
  CEBADATA_DATASET    ruta a DATASET_RETO_AGRO_2026 (por defecto, en la raiz del repo)
  CEBADATA_SALIDAS    carpeta de salidas intermedias (por defecto, pipeline/salidas/)
  CEBADATA_ARTEFACTOS carpeta de artefactos para la API (por defecto, artefactos/)
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Carpeta DATASET_RETO_AGRO_2026 descomprimida (con los zips internos ya extraidos).
# No se versiona; hay que copiarla junto a manage.py o apuntar CEBADATA_DATASET.
RUTA_DATASET = Path(os.environ.get("CEBADATA_DATASET", RAIZ / "DATASET_RETO_AGRO_2026"))

# Carpeta donde el pipeline escribe sus salidas intermedias.
RUTA_SALIDAS = Path(os.environ.get("CEBADATA_SALIDAS", RAIZ / "pipeline" / "salidas"))
RUTA_MODELOS = RUTA_SALIDAS / "modelos"
RUTA_REFERENCIA_SIN_SUELO = RAIZ / "pipeline" / "referencia_sin_suelo"
RUTA_ARTEFACTOS = Path(os.environ.get("CEBADATA_ARTEFACTOS", RAIZ / "artefactos"))


def _dir_con(base, *candidatos, contiene=None):
    """Devuelve el primer candidato que exista (y que tenga `contiene`, si se pide).

    El zip del reto anida carpetas de forma distinta segun como se descomprima.
    """
    for c in candidatos:
        d = base / c
        if d.exists() and (contiene is None or (d / contiene).exists()):
            return d
    return base / candidatos[0]


# --- Archivos de entrada -------------------------------------------------
SHAPEFILE = RUTA_DATASET / "Parcelas_Reto_AGC_CONJUNTO_70_30" / "Parcelas_Reto_AGC_CONJUNTO.shp"
CSV_RENDIMIENTO = RUTA_DATASET / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
CSV_BASICO = RUTA_DATASET / "Conjunto_datos_BASICO_AgroCebada2026.csv"
CSV_PRO = RUTA_DATASET / "Conjunto_datos_PRO_AgroCebada.csv"

DIR_PRECIPITACION = _dir_con(
    RUTA_DATASET,
    "Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/Precipitacion_mensual",
    "Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/Precipitacion_mensual")
DIR_TEMPERATURA = _dir_con(
    RUTA_DATASET,
    "Reto_AgroCebada_Temperatura_2022_2025",
    "Reto_AgroCebada_Temperatura_2022_2025_RECORTADA_CHIRPS/Reto_AgroCebada_Temperatura_2022_2025")
DIR_TOPOGRAFIA = _dir_con(
    RUTA_DATASET,
    "Reto_AgroCebada_Topografia_INEGI_CEM4",
    "Reto_AgroCebada_Topografia_INEGI_CEM4/Reto_AgroCebada_Topografia_INEGI_CEM4",
    "Reto_AgroCebada_Topografia_INEGI_CEM4/Reto_AgroCebada_Topografia_INEGI_CEM4/Reto_AgroCebada_Topografia_INEGI_CEM4",
    contiene="Elevacion_INEGI_CEM4_120m.tif")

# --- Parametros metodologicos (documentados en el reporte) ---------------
# Umbral de nubosidad: una observacion se considera limpia si su porcentaje
# de nubosidad es estrictamente menor a este valor. Con 20 % el EDA mostro
# que practicamente no quedan indices nulos y se conserva una mediana de 32
# observaciones S2 por parcela dentro del ciclo.
UMBRAL_NUBOSIDAD = 20.0

# Ciclo de produccion al que corresponde el rendimiento (abril a octubre 2025)
CICLO_INICIO = "2025-04-01"
CICLO_FIN = "2025-10-31"

# Anios cubiertos por los rasters climaticos
ANIOS_CLIMA = [2022, 2023, 2024, 2025]

# CRS de trabajo para operaciones metricas (el de los rasters del CEM)
CRS_METRICO = "EPSG:6372"

TOTAL_PARCELAS = 197
N_ENTRENAMIENTO = 138
N_PREDICCION = 59

# --- Modelo ---------------------------------------------------------------
# El modelo que se sirve es la mezcla: promedio del ensamble (LightGBM + Random
# Forest + Elastic Net con todas las variables) y de un LightGBM que solo usa la
# longitud (el modelo del superciclo). El paso 07 sigue evaluando los tres
# candidatos para justificar la eleccion en el reporte, pero solo la mezcla se
# exporta a artefactos/.
CANDIDATOS = ["ensamble", "ubicacion", "mezcla"]
MODELO_FINAL = "mezcla"

# Regla para tratar dos RMSE como empatados (la misma que se uso con SoilGrids):
# diferencias menores a esto estan dentro del ruido de la validacion.
UMBRAL_RUIDO_RMSE = 0.010
