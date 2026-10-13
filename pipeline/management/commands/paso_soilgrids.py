"""Paso 05b opcional: baja el suelo de SoilGrids para las 197 parcelas (requiere internet)."""
from django.core.management.base import BaseCommand

from pipeline.pasos import paso05b_soilgrids


class Command(BaseCommand):
    help = "Consulta SoilGrids (reanudable) y guarda salidas/suelo_parcela.csv"

    def handle(self, *args, **opciones):
        paso05b_soilgrids.main()
