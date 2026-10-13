from django.contrib import admin

from api.models import ClimaMensual, Corrida, Explicacion, Parcela, Prediccion, SerieSatelital


@admin.register(Parcela)
class ParcelaAdmin(admin.ModelAdmin):
    list_display = ("id_poligono", "estado", "municipio", "conjunto", "area_ha", "rendimiento_obs")
    list_filter = ("estado", "conjunto")
    search_fields = ("id_poligono", "municipio")


@admin.register(Prediccion)
class PrediccionAdmin(admin.ModelAdmin):
    list_display = ("parcela", "modelo", "valor", "inferior", "superior", "fuera_de_muestra")
    list_filter = ("modelo", "fuera_de_muestra")


@admin.register(Corrida)
class CorridaAdmin(admin.ModelAdmin):
    list_display = ("creada", "modelo_final", "activa")


admin.site.register([ClimaMensual, SerieSatelital, Explicacion])
