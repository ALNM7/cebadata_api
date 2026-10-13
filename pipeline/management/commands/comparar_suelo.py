"""Compara el modelo sin suelo (referencia guardada) contra la corrida actual con suelo."""
from django.core.management.base import BaseCommand

from pipeline import comparar_suelo


class Command(BaseCommand):
    help = "Veredicto sobre si SoilGrids mejora el modelo (regla: baja >= 0.010 t/ha y la ablacion lo confirma)"

    def handle(self, *args, **opciones):
        comparar_suelo.main()
