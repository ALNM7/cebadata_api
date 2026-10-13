"""Paso 05b (opcional, requiere internet): propiedades de suelo de SoilGrids.

Consulta la API publica de SoilGrids (ISRIC, https://soilgrids.org) en el
centroide de cada parcela y guarda pH, carbono organico, arcilla, arena y
capacidad de intercambio cationico de la capa 0 a 5 cm. Es dato publico y
las bases lo permiten documentando la fuente:

  Poggio et al. (2021). SoilGrids 2.0: producing soil information for the
  globe with quantified spatial uncertainty. SOIL, 7, 217-240.
  https://doi.org/10.5194/soil-7-217-2021

Como correrlo (en una maquina con internet normal):
  python paso05b_soilgrids.py

El script es REANUDABLE: guarda el avance en suelo_parcela.csv cada 10
parcelas, y si se interrumpe (o la API falla a la mitad) basta volver a
correrlo; solo consulta las parcelas que falten. Al terminar, correr de
nuevo el pipeline (python ../correr_pipeline.py): el paso 05 integra el
suelo automaticamente y los pasos 07 y 08 cuantifican si mejora el modelo.
"""
import time

import pandas as pd
import requests

from pipeline import conf as config

PROPIEDADES = ["phh2o", "soc", "clay", "sand", "cec"]
URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
RUTA_SALIDA = config.RUTA_SALIDAS / "suelo_parcela.csv"
MAX_REINTENTOS = 3


def consultar(lon: float, lat: float) -> dict:
    params = [("lon", round(lon, 6)), ("lat", round(lat, 6)),
              ("depth", "0-5cm"), ("value", "mean")]
    params += [("property", p) for p in PROPIEDADES]
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    fila = {}
    for capa in r.json()["properties"]["layers"]:
        valor = capa["depths"][0]["values"]["mean"]
        # SoilGrids entrega enteros escalados; d_factor los lleva a unidades fisicas
        factor = capa["unit_measure"].get("d_factor", 1) or 1
        fila[f"suelo_{capa['name']}"] = valor / factor if valor is not None else None
    return fila


def cargar_avance() -> pd.DataFrame:
    if RUTA_SALIDA.exists():
        avance = pd.read_csv(RUTA_SALIDA)
        print(f"Avance previo encontrado: {len(avance)} parcelas ya consultadas.")
        return avance
    return pd.DataFrame(columns=["ID_POLIGONO"])


def main():
    parcelas = pd.read_csv(config.RUTA_SALIDAS / "parcelas_maestro.csv")
    avance = cargar_avance()
    hechas = set(avance["ID_POLIGONO"])
    pendientes = parcelas[~parcelas["ID_POLIGONO"].isin(hechas)]
    print(f"Parcelas pendientes: {len(pendientes)} de {len(parcelas)}")

    filas = avance.to_dict("records")
    for n, (_, p) in enumerate(pendientes.iterrows(), start=1):
        fila = None
        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                fila = consultar(p["lon"], p["lat"])
                break
            except Exception as e:
                espera = 10 * intento
                print(f"  {p['ID_POLIGONO']}: intento {intento} fallo ({e}); "
                      f"espera {espera} s")
                time.sleep(espera)
        if fila is None:
            print(f"  {p['ID_POLIGONO']}: sin respuesta tras {MAX_REINTENTOS} intentos; "
                  "se guarda el avance y puedes volver a correr el script mas tarde.")
            break
        fila["ID_POLIGONO"] = p["ID_POLIGONO"]
        filas.append(fila)
        if n % 10 == 0 or n == len(pendientes):
            pd.DataFrame(filas).to_csv(RUTA_SALIDA, index=False)
            print(f"  avance guardado: {len(filas)}/{len(parcelas)}")
        time.sleep(1)  # cortesia con la API publica

    suelo = pd.DataFrame(filas)
    suelo.to_csv(RUTA_SALIDA, index=False)
    completas = len(suelo)
    nulos = int(suelo.drop(columns="ID_POLIGONO").isna().sum().sum()) if completas else 0
    print(f"\nsuelo_parcela.csv: {completas}/{len(parcelas)} parcelas, {nulos} valores nulos")
    if completas == len(parcelas):
        print("Completo. Ahora corre de nuevo: python ../correr_pipeline.py")


if __name__ == "__main__":
    main()
