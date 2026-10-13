import pytest
from django.core.management import call_command

from api import simulador


@pytest.fixture(scope="session")
def django_db_setup(django_db_setup, django_db_blocker):
    """Carga una sola vez los resultados del pipeline (salidas/) en la base de pruebas."""
    with django_db_blocker.unblock():
        call_command("cargar_resultados", verbosity=0)
    simulador.limpiar_cache()


@pytest.fixture
def cliente(client):
    client.defaults["HTTP_HOST"] = "localhost"
    return client
