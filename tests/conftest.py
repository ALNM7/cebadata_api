import pytest
from rest_framework.test import APIClient


@pytest.fixture
def cliente():
    c = APIClient()
    c.defaults["HTTP_HOST"] = "localhost"
    return c


@pytest.fixture(scope="session")
def art():
    from api import artefactos

    return artefactos.obtener()
