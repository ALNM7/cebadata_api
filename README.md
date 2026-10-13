# CebaData API

Proyecto Django con el pipeline de datos y modelos del Reto AgroCebada FIRA 2026 y la API que consume el dashboard ([DashboardCebaData](https://github.com/ALNM7/DashboardCebaData), hecho en Angular).

Predice el rendimiento de cebada (t/ha) en 197 parcelas del altiplano de Hidalgo, Puebla y Tlaxcala, ciclo abril a octubre de 2025. En 138 se midió el rendimiento y las otras 59 son las que el modelo predice.

## Qué hay en el repo

```
manage.py
cebadata/            proyecto Django (settings, urls)
pipeline/            app con el pipeline: pasos 01 a 09, modelos y comandos
  conf.py            rutas y parámetros (dataset, salidas, modelo final)
  modelado.py        los tres candidatos (ensamble, ubicación y mezcla) y su ajuste con Optuna
  pasos/             paso01 a paso09, uno por archivo
  comparar_suelo.py  veredicto sobre SoilGrids con la regla de ruido
api/                 app con los modelos de base de datos, el simulador y los endpoints
salidas/             tablas y modelo que escribe el pipeline (se versionan, pesan 2 MB)
referencia_sin_suelo/ resultados de la versión sin SoilGrids, para comparar_suelo
tests/               pruebas de la API y del pipeline
```

## Cómo correrlo

Necesitas Python 3.10 o mayor.

```bash
python3 -m venv .venv && source .venv/bin/activate     # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

python manage.py migrate
python manage.py cargar_resultados      # lee salidas/ y llena la base de datos
python manage.py runserver              # http://localhost:8000
```

Con eso la API ya responde, porque `salidas/` trae los resultados de la última corrida. Para probarla:

```bash
curl http://localhost:8000/api/salud
curl http://localhost:8000/api/parcelas/AGC_003
```

El dashboard (ng serve en el puerto 4200) pide los datos a `http://localhost:8000`. Si lo sirves desde otro origen, pásalo en `CORS_ALLOWED_ORIGINS` (varios separados por coma).

### Pruebas

```bash
python -m pytest
```

## Endpoints

| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/api/salud` | estado del servicio y si ya hay resultados cargados |
| GET | `/api/parcelas` | lista de las 197 parcelas con su predicción e intervalo |
| GET | `/api/parcelas/<id>` | `{ parcela, otros_enfoques, serie, clima, shap, shap_otras, base, vecinos }` |
| GET | `/api/modelo/metricas` | métricas, comparación de candidatos, dispersión, importancias, robustez y hallazgos |
| POST | `/api/simulador` | recibe `{ id, delta_lluvia_mayo_pct, delta_tmin_abril_c }` y devuelve el escenario |

`/api/parcelas` y `/api/parcelas/<id>` aceptan `?modelo=ensamble|ubicacion|mezcla`. Sin parámetro usan el modelo congelado. El simulador acepta `modelo` en el cuerpo.

En las parcelas medidas, `rendimiento_pred` es la predicción fuera de muestra (validación cruzada), no un ajuste sobre datos que el modelo ya vio. En las 59 ocultas es la del modelo final. El intervalo es la predicción más/menos el cuantil 90 del error absoluto de la validación estratificada de ese candidato.

## Los tres candidatos

El paso 07 compara tres enfoques bajo las mismas reglas y las mismas 138 parcelas:

| | Qué es |
|---|---|
| **A. Ensamble** | LightGBM, Random Forest y Elastic Net con 121 variables (clima, satélite, topografía, ubicación) |
| **B. Ubicación** | LightGBM que solo usa la longitud. Es lo que encontró el superciclo de modelos clásicos |
| **C. Mezcla** | Promedio de las predicciones de A y B |

Dos validaciones, las dos anidadas (Optuna solo ve el fold de entrenamiento):

- **Estratificada por municipio** (5 folds x 3 repeticiones). Mide lo que se va a calificar, porque las 59 ocultas están en los mismos municipios que las de entrenamiento.
- **LOMO** (un municipio completo fuera en cada ronda). Mide qué tan bien funciona en una zona nueva de la región.

Resultados de la última corrida (RMSE en t/ha, `salidas/comparacion_candidatos.csv`):

| Candidato | Estratificada | MAE | LOMO |
|---|---|---|---|
| A. Ensamble | 0.508 | 0.362 | 0.722 |
| B. Ubicación | 0.525 | 0.364 | 0.539 |
| C. Mezcla | 0.508 | 0.354 | 0.595 |

Cómo leerlo:

- Donde se califica, A y C empatan en RMSE (la diferencia es mucho menor que 0.010) y C tiene el mejor MAE. B queda 0.017 por detrás.
- En zonas nuevas B es el mejor, C queda a 0.056 y A se cae (0.722). La mezcla recorta mucho el problema de A, pero no lo elimina.
- Si solo se predijera la media de las demás zonas, el RMSE en LOMO sería 1.153, no 0.846. Ese baseline infla la mejora, por eso conviene citar siempre el 0.846 de la validación estratificada.
- El sesgo de B en LOMO es +0.077 t/ha: en promedio sobreestima.

### Qué modelo queda congelado

Lo fija `CEBADATA_MODELO_FINAL` (`ensamble`, `ubicacion` o `mezcla`; por defecto `mezcla`). El paso 07 imprime además una recomendación con una regla fija: se consideran empatados los candidatos a menos de 0.010 t/ha del mejor RMSE estratificado, y entre ellos se elige el de menor RMSE en LOMO. Con los resultados actuales recomienda la mezcla. Si el modelo que configuras no coincide con la recomendación, el paso lo avisa.

Para cambiar de modelo basta repetir los pasos 07 a 09 con otra variable:

```bash
CEBADATA_MODELO_FINAL=ensamble python manage.py run_pipeline --desde 7 --cargar
```

### Qué muestra el dashboard por parcela

La explicación por parcela (cascada tipo SHAP) viene de la parte A. El modelo de ubicación solo tiene una variable y no tiene nada que contar. Lo que falta para llegar a la predicción final (variables fuera de las 8 principales, ajuste de los otros modelos del ensamble y la mezcla con la ubicación) se agrupa en "otras variables". En las parcelas medidas el SHAP sale de un LightGBM entrenado sin esa parcela, igual que su predicción fuera de muestra.

### El simulador

El simulador cambia la lluvia de mayo (de -40 % a +40 %) y la temperatura mínima de abril (de -2 a +2 °C) en la fila de la parcela, recalcula las variables del ciclo que dependen de esos meses (lluvia total, anomalía, meses secos, grados-día) y vuelve a predecir con el modelo guardado. La predicción del escenario es la predicción mostrada de la parcela más el cambio que calcula el modelo. Avisa cuando el escenario sale del rango de entrenamiento y ensancha el intervalo.

Ojo con la lectura: es la sensibilidad que aprendió el modelo, no una relación causal comprobada. En esta región la lluvia de mayo también funciona como indicador de zona. Además, con la mezcla solo la mitad de la predicción (la parte A) reacciona al clima, así que el efecto sale más chico que con el ensamble solo.

## El pipeline

Los pasos corren en orden con un solo comando:

```bash
python manage.py run_pipeline                       # pasos 01 a 09
python manage.py run_pipeline --desde 7 --cargar    # solo modelado, y luego carga la base
```

| Paso | Qué hace |
|---|---|
| 01 | Tabla maestra de parcelas (ID, conjunto, estado, municipio, área, rendimiento, centroide) |
| 02 | Separa sensores y marca observaciones limpias y del ciclo 2025 |
| 03 | Clima mensual y topografía por parcela |
| 04 | 22 validaciones de integridad; truena si algo no cuadra |
| 05 | Tabla de modelado: fenología por índice, clima, historia y estáticos |
| 05b | (opcional, requiere internet) suelo de SoilGrids: `python manage.py paso_soilgrids` |
| 06 | Baselines con CV repetida estratificada por municipio |
| 07 | Poda de redundancia, Optuna anidado, los tres candidatos con las dos validaciones, modelo final y predicciones |
| 08 | SHAP global, ablación por bloques y robustez espacial |
| 09 | SHAP por parcela para la ficha del dashboard |

Los pasos 01 a 04 necesitan la carpeta `DATASET_RETO_AGRO_2026/` del reto. No se versiona: cópiala junto a `manage.py` o apunta `CEBADATA_DATASET` a donde la tengas. Los CSV y el shapefile van sueltos; los zips de clima y topografía deben estar extraídos en carpetas con el mismo nombre del zip. `geopandas` y `rasterio` traen dependencias binarias; con un `pip` reciente se instalan sin compilar. Si falla en Windows, usa conda: `conda install -c conda-forge geopandas rasterio`.

Desde el paso 05 solo hace falta lo que ya está en `salidas/`, pero el paso 05 vuelve a leer los CSV satelitales (`satelital_s2.csv` y `satelital_planet.csv`), que se generan en el paso 02 y no se versionan por su tamaño. Los pasos 06 a 09 sí corren con lo que trae el repo.

### Curvas satelitales de la ficha

La curva del índice por parcela sale de `salidas/satelital_s2.csv` y `salidas/satelital_planet.csv`. Si no existen, `cargar_resultados` lo avisa y la ficha muestra "Sin observaciones". Para llenarla corre los pasos 01 a 03 con el dataset y vuelve a cargar los resultados. Planet solo trae NDVI, EVI y LAI; NDDI y NDWI solo existen en Sentinel-2.

### Datos externos: SoilGrids (probado y descartado)

El suelo de SoilGrids 2.0 (Poggio et al. 2021, https://doi.org/10.5194/soil-7-217-2021) se probó con una regla fija que se fijó antes de correrlo: se adopta solo si baja el RMSE del ensamble al menos 0.010 t/ha y la ablación pareada lo confirma. No la pasó (+0.0013 en la validación anidada, -0.0023 en la ablación pareada, que está dentro del ruido). El paso queda cableado por si se quiere repetir:

```bash
python manage.py paso_soilgrids
python manage.py run_pipeline --desde 5 --cargar
python manage.py comparar_suelo
```

`comparar_suelo` ya no dice "el suelo ayuda" cuando la diferencia es menor a 0.010: dice que está dentro del ruido. Entre corridas hay variaciones de casi 0.01 por sí solas (la ablación de todo pasó de 0.508 a 0.515 sin cambiar nada importante).

Los datos externos de INEGI (edafología y nitrógeno) que usó el superciclo tampoco pasan esa regla (de 0.550 a 0.542 son 0.008 t/ha) y no se usan.

## Decisiones metodológicas (para el reporte técnico)

- El identificador del shapefile viene truncado como `ID_POLIGON`; se renombra a `ID_POLIGONO` para unificar la llave.
- El CSV de rendimiento trae una columna vacía sin nombre; se descarta.
- Umbral de nubosidad de 20 %: por debajo de él, menos del 1 % de los índices viene nulo y toda parcela conserva al menos 26 observaciones S2 dentro del ciclo. El umbral es un parámetro en `pipeline/conf.py`.
- El clima se muestrea en el centroide de la parcela porque el píxel de CHIRPS/CHIRTS (unos 5 km) es mucho más grande que cualquier parcela; la topografía (píxel de 120 m) sí se promedia zonalmente con `all_touched=True`.
- Ninguna fila satelital se elimina: el filtrado queda como bandera para que cada experimento decida.
- Planet y Sentinel-2 se mantienen como fuentes separadas, como advierte el diccionario del reto.
- La validación es anidada: Optuna solo ve el fold de entrenamiento, así que las cifras no están infladas por la búsqueda de hiperparámetros.

## Límites que hay que decir tal cual

- El modelo aprendió de una sola temporada.
- La longitud solo funciona porque aquí el rendimiento cambia de oeste a este dentro de una caja de unos 45 km. Fuera de esta región (otro estado o valle) ningún candidato es confiable; para usarlo ahí hacen falta datos de rendimiento de esa zona para reentrenar.
- En Tlaxcala el rendimiento solo toma tres valores (2.5, 3.0 y 3.5), lo que parece una estimación por zona y no una medición por parcela.
- Es una herramienta de apoyo y no sustituye la evaluación crediticia.

## Variables de entorno

| Variable | Para qué | Por defecto |
|---|---|---|
| `CEBADATA_MODELO_FINAL` | candidato congelado | `mezcla` |
| `CEBADATA_DATASET` | ruta a `DATASET_RETO_AGRO_2026` | junto a `manage.py` |
| `CEBADATA_SALIDAS` | carpeta de salidas | `salidas/` |
| `DJANGO_DEBUG` | `1` o `0` | `1` |
| `DJANGO_SECRET_KEY` | obligatoria con `DJANGO_DEBUG=0` | |
| `DJANGO_ALLOWED_HOSTS` | hosts separados por coma | `localhost,127.0.0.1` |
| `CORS_ALLOWED_ORIGINS` | orígenes del front separados por coma | `http://localhost:4200,http://127.0.0.1:4200` |
