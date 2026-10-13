"""Configuracion de Django para la API de CebaData.

La API es de solo lectura: sirve los artefactos congelados de artefactos/ (modelo,
tablas y metadata) y no usa base de datos, sesiones ni admin.

Variables de entorno:
  DJANGO_SECRET_KEY       llave secreta (obligatoria si DJANGO_DEBUG=0)
  DJANGO_DEBUG            1 (por defecto) o 0
  DJANGO_ALLOWED_HOSTS    hosts separados por coma (en Render/Railway: el dominio del servicio)
  CORS_ALLOWED_ORIGINS    origenes del front separados por coma (por defecto el de ng serve)
  CEBADATA_ARTEFACTOS     carpeta de artefactos (por defecto artefactos/ junto a manage.py)
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    if not DEBUG:
        raise RuntimeError("Define DJANGO_SECRET_KEY cuando DJANGO_DEBUG=0")
    SECRET_KEY = "django-insecure-solo-para-desarrollo-local"

ALLOWED_HOSTS = [h.strip() for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]

ARTEFACTOS_DIR = Path(os.environ.get("CEBADATA_ARTEFACTOS", BASE_DIR / "artefactos"))

INSTALLED_APPS = [
    "corsheaders",
    "rest_framework",
    "api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "cebadata.urls"
WSGI_APPLICATION = "cebadata.wsgi.application"

# Sin base de datos: todo sale de los artefactos.
DATABASES = {}

LANGUAGE_CODE = "es-mx"
TIME_ZONE = "America/Mexico_City"
USE_I18N = False
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "UNAUTHENTICATED_USER": None,
}

# El front (Angular) corre en otro origen: en desarrollo, ng serve en el 4200.
CORS_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "CORS_ALLOWED_ORIGINS", "http://localhost:4200,http://127.0.0.1:4200").split(",") if o.strip()]
CORS_URLS_REGEX = r"^/api/.*$"

# Detras del proxy de Render/Railway las peticiones llegan por https
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
