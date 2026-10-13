"""Corre el pipeline completo (o una parte) en orden.

Uso:
    python manage.py run_pipeline                    # pasos 01 a 09
    python manage.py run_pipeline --desde 7          # solo modelado: 07, 08 y 09
    python manage.py run_pipeline --desde 5 --hasta 6
    python manage.py run_pipeline --cargar           # y despues carga los resultados en la base

Los pasos 01 a 04 necesitan la carpeta DATASET_RETO_AGRO_2026 (ver pipeline/conf.py).
Desde el paso 05 basta con lo que hay en salidas/. El paso 05b (SoilGrids, opcional,
requiere internet) se corre aparte: python manage.py paso_soilgrids
"""
import importlib
import time

from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError

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
}


class Command(BaseCommand):
    help = "Corre el pipeline de datos y modelos (pasos 01 a 09)"

    def add_arguments(self, parser):
        parser.add_argument("--desde", type=int, default=1, choices=sorted(PASOS))
        parser.add_argument("--hasta", type=int, default=9, choices=sorted(PASOS))
        parser.add_argument("--cargar", action="store_true",
                            help="al terminar, carga los resultados en la base de datos")

    def handle(self, *args, desde, hasta, cargar, **opciones):
        if desde > hasta:
            raise CommandError("--desde no puede ser mayor que --hasta")
        inicio = time.time()
        for n in range(desde, hasta + 1):
            nombre = PASOS[n]
            self.stdout.write(f"\n=== {nombre} " + "=" * (44 - len(nombre)))
            modulo = importlib.import_module(f"pipeline.pasos.{nombre}")
            try:
                modulo.main()
            except Exception as e:
                raise CommandError(f"El paso {nombre} fallo ({type(e).__name__}: {e}). Se detiene el pipeline.") from e
        self.stdout.write(self.style.SUCCESS(f"\nPipeline en {time.time() - inicio:.0f} s. Salidas en salidas/"))
        if cargar:
            call_command("cargar_resultados")
