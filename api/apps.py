from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"
    verbose_name = "API de CebaData"

    def ready(self):
        # Se cargan una sola vez al arrancar el proceso. Con gunicorn --preload se cargan en el
        # proceso maestro y los workers los heredan. Si faltan o no cuadran, el arranque falla.
        from django.conf import settings

        from api import artefactos

        artefactos.cargar(settings.ARTEFACTOS_DIR)
