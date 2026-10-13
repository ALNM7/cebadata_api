from django.db import models


class Parcela(models.Model):
    """Una de las 197 parcelas del reto (138 medidas, 59 por predecir)."""

    class Conjunto(models.TextChoices):
        ENTRENAMIENTO = "ENTRENAMIENTO", "Entrenamiento"
        PREDICCION = "PREDICCION", "Predicción"

    id_poligono = models.CharField("ID", max_length=16, primary_key=True)
    estado = models.CharField(max_length=40)
    municipio = models.CharField(max_length=80)
    area_ha = models.FloatField()
    conjunto = models.CharField(max_length=14, choices=Conjunto.choices)
    rendimiento_obs = models.FloatField(null=True, blank=True)
    lat = models.FloatField()
    lon = models.FloatField()
    elevacion_m = models.FloatField()
    pendiente_grados = models.FloatField()

    class Meta:
        ordering = ["id_poligono"]

    def __str__(self):
        return self.id_poligono


class Prediccion(models.Model):
    """Prediccion de un candidato (ensamble, ubicacion o mezcla) para una parcela.

    En las parcelas medidas es la prediccion fuera de muestra (validacion
    cruzada); en las ocultas, la del modelo final.
    """

    parcela = models.ForeignKey(Parcela, on_delete=models.CASCADE, related_name="predicciones")
    modelo = models.CharField(max_length=12)
    valor = models.FloatField()
    inferior = models.FloatField()
    superior = models.FloatField()
    fuera_de_muestra = models.BooleanField(default=False)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["parcela", "modelo"], name="unica_prediccion")]


class ClimaMensual(models.Model):
    """Clima del ciclo 2025 (abril a octubre) en el centroide de la parcela."""

    parcela = models.ForeignKey(Parcela, on_delete=models.CASCADE, related_name="clima")
    mes = models.PositiveSmallIntegerField()
    prec_mm = models.FloatField()
    tmax_c = models.FloatField()
    tmin_c = models.FloatField()

    class Meta:
        ordering = ["mes"]
        constraints = [models.UniqueConstraint(fields=["parcela", "mes"], name="unico_clima_mes")]


class SerieSatelital(models.Model):
    """Observaciones limpias de 2025 de un indice, por sensor (Sentinel-2 o Planet)."""

    parcela = models.ForeignKey(Parcela, on_delete=models.CASCADE, related_name="serie")
    fecha = models.DateField()
    fuente = models.CharField(max_length=8)   # S2 | Planet
    indice = models.CharField(max_length=8)
    valor = models.FloatField()

    class Meta:
        ordering = ["fecha"]
        indexes = [models.Index(fields=["parcela", "fuente", "indice"])]


class Explicacion(models.Model):
    """Variables con mas efecto en la prediccion de una parcela (valores SHAP de la parte A)."""

    parcela = models.OneToOneField(Parcela, on_delete=models.CASCADE, related_name="explicacion")
    items = models.JSONField()          # [{variable, etiqueta, valor_variable, unidad, contribucion}]


class Corrida(models.Model):
    """Resultado de una corrida del pipeline: metricas, comparacion y hallazgos."""

    creada = models.DateTimeField(auto_now_add=True)
    modelo_final = models.CharField(max_length=12)
    base = models.FloatField(help_text="Rendimiento medio de las parcelas medidas (t/ha)")
    metricas = models.JSONField()
    rango_entrenamiento = models.JSONField()
    activa = models.BooleanField(default=True)

    class Meta:
        ordering = ["-creada"]
