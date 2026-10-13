"""Corre el pipeline completo (o una parte) en orden.

Uso, desde la raiz del repo:
    python -m pipeline.run_pipeline                    # pasos 01 a 10
    python -m pipeline.run_pipeline --desde 7          # solo modelado y exportacion
    python -m pipeline.run_pipeline --desde 10         # solo regenerar artefactos/

Los pasos 01 a 04 necesitan la carpeta DATASET_RETO_AGRO_2026 (ver pipeline/conf.py).
El paso 05b (SoilGrids, opcional, requiere internet) se corre aparte:
    python -m pipeline.pasos.paso05b_soilgrids
"""
import argparse
import importlib
import sys
import time

PASOS = {
    1: "paso01_parcelas",
    2: "paso02_satelital",
    3: "paso03_clima_topografia",
    4: "paso04_validaciones",
    5: "paso05_features",
    6: "paso06_baselines",
    7: "paso07_optimizacion",
    8: "paso08_interpretacion",
    9: "paso09_explicaciones",
    10: "paso10_exportar_artefactos",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", type=int, default=1, choices=sorted(PASOS))
    ap.add_argument("--hasta", type=int, default=max(PASOS), choices=sorted(PASOS))
    args = ap.parse_args()
    if args.desde > args.hasta:
        ap.error("--desde no puede ser mayor que --hasta")

    inicio = time.time()
    for n in range(args.desde, args.hasta + 1):
        nombre = PASOS[n]
        print(f"\n=== {nombre} " + "=" * (44 - len(nombre)), flush=True)
        try:
            importlib.import_module(f"pipeline.pasos.{nombre}").main()
        except Exception as e:  # noqa: BLE001
            print(f"\nEl paso {nombre} fallo ({type(e).__name__}: {e}). Se detiene el pipeline.")
            sys.exit(1)
    print(f"\nPipeline en {time.time() - inicio:.0f} s.")


if __name__ == "__main__":
    main()
