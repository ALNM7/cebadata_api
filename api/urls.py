from django.urls import path

from api import views

urlpatterns = [
    path("salud", views.salud, name="salud"),
    path("parcelas", views.parcelas, name="parcelas"),
    path("parcelas/<str:id_poligono>", views.parcela_detalle, name="parcela-detalle"),
    path("modelo/metricas", views.metricas, name="metricas"),
    path("simulador", views.simular, name="simulador"),
]
