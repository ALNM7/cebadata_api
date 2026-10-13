# CebaData API

API de solo lectura del Reto AgroCebada FIRA 2026 y pipeline para reproducir el reporte. Sirve al dashboard ([DashboardCebaData](https://github.com/ALNM7/DashboardCebaData), hecho en Angular).

Predice el rendimiento de cebada (t/ha) en 197 parcelas del altiplano de Hidalgo, Puebla y Tlaxcala, ciclo abril a octubre de 2025. En 138 se midió el rendimiento y las otras 59 son las que el modelo predice.

## Cómo está organizado

```
artefactos/     lo único que lee la API: modelo, tablas y metadata (versionado, 2 MB)
api/            app de Django: carga los artefactos al arrancar y sirve los endpoints
cebadata/       proyecto Django (sin base de datos)
tests/          pruebas de la API (pytest)
pipeline/       entrenamiento y exportación; la API no lo importa ni lo necesita
```

El backend en tiempo de ejecución no corre el pipeline. Entrena una vez fuera de la API, exporta `artefactos/` y la API los carga una sola vez al arrancar.

## Modelo único: la mezcla

El modelo que se sirve es la **mezcla**: el promedio de dos predicciones.

- el **ensamble** (promedio de LightGBM, Random Forest y Elastic Net con 121 variables de clima, satélite, topografía y ubicación), y
- un **LightGBM que solo usa la longitud**, que es lo que encontró el superciclo de modelos clásicos del compañero.

Pesos: cada miembro del ensamble pesa 1/6 y el modelo de longitud pesa 1/2.

### Por qué la mezcla

El paso 07 del pipeline comparó tres enfoques con las mismas 138 parcelas y dos validaciones anidadas (Optuna solo ve el fold de entrenamiento). La estratificada por municipio mide lo que se va a calificar; LOMO deja un municipio completo fuera y mide una zona nueva de la región.

| Enfoque | Variables | RMSE donde se califica | MAE | RMSE zona nueva (LOMO) | Sesgo LOMO |
|---|---|---|---|---|---|
| Ensamble | 121 | 0.508 | 0.362 | 0.722 | -0.102 |
| Solo ubicación | 1 | 0.525 | 0.364 | 0.539 | +0.077 |
| **Mezcla** | 122 | **0.508** | **0.354** | 0.595 | -0.013 |

Donde se califica, ensamble y mezcla empatan en RMSE (la diferencia es menor a 0.010 t/ha, dentro del ruido) y la mezcla tiene el mejor MAE. En zona nueva la ubicación sola es la mejor, la mezcla queda 0.056 por encima y el ensamble se cae. La mezcla es la que mejor equilibra las dos pruebas: no pierde donde se califica y recorta casi la mitad de la caída del ensamble en zonas nuevas. Esta tabla viaja también en `metricas_dashboard.json` para el panel de metodología del dashboard.

Referencias: predecir la media da 0.846 en la validación estratificada. En LOMO la media de las demás zonas da 1.153 porque al sacar Chignahuapan (la mitad de los datos) el promedio del resto cae mucho; por eso conviene citar el 0.846.

### Límites que hay que decir tal cual

- El modelo aprendió de una sola temporada.
- La longitud solo funciona porque aquí el rendimiento cambia de oeste a este dentro de una caja de unos 45 km. Fuera de esta región ninguno es confiable; para usarlo ahí hacen falta rendimientos de esa zona para reentrenar.
- Con la mezcla solo la parte del ensamble reacciona al clima, así que en el simulador el efecto sale la mitad de grande que con el ensamble solo.
- La lluvia de mayo también puede estar actuando como indicador de zona, así que su peso no prueba por sí solo que la lluvia cause el rendimiento.
- En Tlaxcala el rendimiento solo toma tres valores (2.5, 3.0 y 3.5), lo que parece una estimación por zona.
- Es una herramienta de apoyo y no sustituye la evaluación crediticia.

## Levantar la API

Necesitas Python 3.13 (el de `.python-version`). Las versiones de numpy, pandas, scikit-learn, lightgbm y joblib están fijadas porque son las del entrenamiento: los `.joblib` se cargan con ellas.

```bash
python3 -m venv .venv && source .venv/bin/activate     # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

python manage.py runserver                              # http://localhost:8000
```

```bash
curl http://localhost:8000/api/salud
curl http://localhost:8000/api/parcelas/AGC_003
```

Al arrancar valida `artefactos/`: versión del formato, checksums sha256 de los `.joblib` y que scikit-learn y lightgbm instalados sean los del entrenamiento. Si algo no cuadra no arranca y dice qué. Para servir otra carpeta de artefactos: `CEBADATA_ARTEFACTOS=/ruta`.

El dashboard (`ng serve` en el 4200) pide los datos a `http://localhost:8000`. Si lo sirves desde otro origen, pásalo en `CORS_ALLOWED_ORIGINS` (varios separados por coma).

### Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Incluyen la **prueba de paridad**: con los artefactos cargados la API reproduce las predicciones de las 59 parcelas (contra la tabla con 1e-9 y contra `pipeline/salidas/predicciones_59_parcelas.csv` redondeando a 3 decimales), y verifica que los pins de `requirements.txt`, los del pipeline y los de `metadata.json` coincidan, y que la API no importe el pipeline.

## Endpoints

| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/api/salud` | estado, modelo y fecha de los artefactos |
| GET | `/api/parcelas` | las 197 parcelas con predicción e intervalo |
| GET | `/api/parcelas/<id>` | `{ parcela, serie, clima, shap, shap_otras, base, vecinos }` |
| GET | `/api/modelo/metricas` | métricas de la mezcla (estratificada y LOMO), dispersión, importancias, robustez, hallazgos y la tabla de comparación para la metodología |
| POST | `/api/simulador` | recibe `{ id, delta_lluvia_mayo_pct, delta_tmin_abril_c }` |

Las predicciones salen de `artefactos/tablas/predicciones.csv`, no se recalculan. En las parcelas medidas es la predicción fuera de muestra (validación cruzada); en las 59 ocultas, la del modelo final. El intervalo es la predicción más/menos el cuantil 90 del error absoluto de CV de la mezcla (0.749 t/ha).

### Simulador

Sobre una parcela existente cambia la lluvia de mayo (de -40 % a +40 %) y la temperatura mínima de abril (de -2 a +2 °C), recalcula las variables del ciclo que dependen de esos meses (meses secos y anomalía de lluvia) y predice con la mezcla. La respuesta trae la base, el escenario con su intervalo (que se ensancha con el tamaño del cambio), la diferencia y `fuera_de_rango` con las variables que salieron del rango de entrenamiento. La predicción del escenario es la predicción mostrada más el cambio que calcula el modelo, así la base coincide siempre con el mapa y la ficha.

## Regenerar los artefactos

Solo hace falta si cambia el modelo o los datos. Se hace fuera de la API, con el entorno del pipeline:

```bash
python3 -m venv .venv-pipeline && source .venv-pipeline/bin/activate
pip install -r pipeline/requirements-pipeline.txt

# dataset del reto: descomprimido en DATASET_RETO_AGRO_2026/ (raíz del repo) o apuntado con
# CEBADATA_DATASET; los zips internos de clima y topografía van extraídos en carpetas del mismo nombre
python -m pipeline.run_pipeline                  # pasos 01 a 10 (unos 8 minutos)
python -m pipeline.run_pipeline --desde 10       # solo reexportar artefactos/ con lo que ya hay en pipeline/salidas
python -m pytest pipeline/tests -p no:django     # pruebas del pipeline
```

Después revisa `git diff artefactos/`, corre las pruebas de la API y haz commit. Si cambias una de las cinco librerías de ML, cámbiala igual en `requirements.txt` y en `pipeline/requirements-pipeline.txt` (una prueba lo verifica) y regenera.

## Qué contiene `artefactos/`

```
metadata.json            orden exacto de las 121 variables, pesos de la mezcla, versiones de librerías,
                         cuantil del intervalo, métricas, rango min/max de cada variable en entrenamiento,
                         etiquetas, zonas de entrenamiento y checksums
metricas_dashboard.json  lo que muestran "Desempeño del modelo" y "Hallazgos"
modelo/                  imputador, escalador, lgbm, rf, enet y lgbm_longitud (joblib)
tablas/                  parcelas, predicciones, features_modelo, shap_parcela, clima_mensual,
                         serie_satelital (curvas de Sentinel-2 y Planet) y vecinos
```

## El pipeline

`pipeline/` queda en el repo solo para reproducibilidad del reporte. Los pasos corren en orden con `python -m pipeline.run_pipeline`:

| Paso | Qué hace |
|---|---|
| 01 | Tabla maestra de parcelas (ID, conjunto, estado, municipio, área, rendimiento, centroide) |
| 02 | Separa sensores y marca observaciones limpias y del ciclo 2025 |
| 03 | Clima mensual y topografía por parcela |
| 04 | 22 validaciones de integridad; truena si algo no cuadra |
| 05 | Tabla de modelado: fenología por índice, clima, historia y estáticos |
| 05b | (opcional, requiere internet) suelo de SoilGrids: `python -m pipeline.pasos.paso05b_soilgrids` |
| 06 | Baselines con CV repetida estratificada por municipio |
| 07 | Poda de redundancia, Optuna anidado, comparación de los tres enfoques con las dos validaciones, modelo final y predicciones |
| 08 | SHAP global, ablación por bloques y robustez espacial |
| 09 | SHAP por parcela para la ficha |
| 10 | Exporta `artefactos/` |

Con el dataset del reto la corrida reproduce exactamente los números de arriba (los CSV de los pasos 01 a 05 quedaron idénticos a los que ya estaban versionados en `pipeline/salidas/`).

### Datos externos: SoilGrids (probado y descartado)

El suelo de SoilGrids 2.0 (Poggio et al. 2021, https://doi.org/10.5194/soil-7-217-2021) se probó con una regla fijada antes de correrlo: se adopta solo si baja el RMSE del ensamble al menos 0.010 t/ha y la ablación pareada lo confirma. No la pasó (+0.0013 en la validación anidada, -0.0023 en la ablación pareada, dentro del ruido). Los datos de INEGI (edafología y nitrógeno) que usó el superciclo tampoco la pasan (de 0.550 a 0.542 son 0.008 t/ha). Para repetir la prueba:

```bash
python -m pipeline.pasos.paso05b_soilgrids
python -m pipeline.run_pipeline --desde 5
python -m pipeline.comparar_suelo
```

### Decisiones metodológicas (para el reporte técnico)

- El identificador del shapefile viene truncado como `ID_POLIGON`; se renombra a `ID_POLIGONO`.
- El CSV de rendimiento trae una columna vacía sin nombre; se descarta.
- Umbral de nubosidad de 20 %: por debajo, menos del 1 % de los índices viene nulo y toda parcela conserva al menos 26 observaciones S2 dentro del ciclo. Es un parámetro en `pipeline/conf.py`.
- El clima se muestrea en el centroide de la parcela porque el píxel de CHIRPS/CHIRTS (unos 5 km) es mucho más grande que cualquier parcela; la topografía (píxel de 120 m) se promedia zonalmente con `all_touched=True`.
- Ninguna fila satelital se elimina: el filtrado queda como bandera.
- Planet y Sentinel-2 se mantienen como fuentes separadas.

## Despliegue en Render o Railway

El backend es ligero (sin base de datos, sin pipeline, un worker de gunicorn con unos 160 MB).

- **Render:** conecta el repo; `render.yaml` ya trae el servicio. Ajusta `DJANGO_ALLOWED_HOSTS` al dominio del servicio y `CORS_ALLOWED_ORIGINS` al del dashboard.
- **Railway:** `railway.json` y `Procfile` ya traen el comando de arranque y el healthcheck `/api/salud`. Define las mismas variables.

| Variable | Para qué | Por defecto |
|---|---|---|
| `DJANGO_SECRET_KEY` | obligatoria con `DJANGO_DEBUG=0` | |
| `DJANGO_DEBUG` | `1` o `0` | `1` |
| `DJANGO_ALLOWED_HOSTS` | hosts separados por coma | `localhost,127.0.0.1` |
| `CORS_ALLOWED_ORIGINS` | orígenes del front separados por coma | `http://localhost:4200,http://127.0.0.1:4200` |
| `CEBADATA_ARTEFACTOS` | carpeta de artefactos | `artefactos/` |
| `WEB_CONCURRENCY` | workers de gunicorn | `1` |

El arranque usa `--preload`: los artefactos se cargan una vez en el proceso maestro y los workers los heredan. Los modelos se fijan en `n_jobs=1` al cargarlos porque con varios hilos de OpenMP el worker se bloqueaba tras el fork.
