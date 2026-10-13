"""Endpoints de la API de CebaData (solo lectura sobre los artefactos).

  GET  /api/salud
  GET  /api/parcelas
  GET  /api/parcelas/<id>
  GET  /api/modelo/metricas
  POST /api/simulador
  GET  /api/nueva-parcela/opciones
  POST /api/predecir
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response

from api import artefactos, nueva_parcela, simulador


def _error(mensaje, status):
    return Response({"error": mensaje}, status=status)


@api_view(["GET"])
def salud(request):
    art = artefactos.obtener()
    return Response({
        "estado": "ok", "modelo": art.metadata["modelo"]["nombre"], "parcelas": len(art.parcelas),
        "artefactos_generados": art.metadata["generado"], "version_artefactos": art.metadata["version_artefactos"],
    })


@api_view(["GET"])
def parcelas(request):
    art = artefactos.obtener()
    return Response([art.parcelas[i] for i in art.orden])


@api_view(["GET"])
def parcela_detalle(request, id_poligono):
    art = artefactos.obtener()
    p = art.parcelas.get(id_poligono)
    if p is None:
        return _error("Parcela no encontrada", 404)
    return Response({
        "parcela": p,
        "serie": art.serie.get(id_poligono, []),
        "clima": [{"mes": int(c["mes"]), "prec_mm": round(c["prec_mm"]), "tmax_c": round(c["tmax_c"], 1),
                   "tmin_c": round(c["tmin_c"], 1)} for c in art.clima.get(id_poligono, [])],
        "shap": art.shap.get(id_poligono, []),
        "shap_otras": art.shap_otras.get(id_poligono, 0.0),
        "base": round(art.base, 2),
        "vecinos": art.vecinos.get(id_poligono, []),
    })


@api_view(["GET"])
def metricas(request):
    return Response(artefactos.obtener().dashboard)


def _numero(cuerpo, clave):
    valor = cuerpo.get(clave, 0)
    if valor is None:
        return 0.0
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError(f"{clave} debe ser un numero")
    return float(valor)


@api_view(["POST"])
def simular(request):
    art = artefactos.obtener()
    cuerpo = request.data
    if not isinstance(cuerpo, dict):
        return _error("Cuerpo invalido: se espera un objeto JSON", 400)
    try:
        dl = _numero(cuerpo, "delta_lluvia_mayo_pct")
        dt = _numero(cuerpo, "delta_tmin_abril_c")
    except ValueError as e:
        return _error(f"Cuerpo invalido: {e}", 400)
    id_poligono = cuerpo.get("id")
    if id_poligono not in art.parcelas:
        return _error("Parcela no encontrada", 404)
    return Response(simulador.simular(art, id_poligono, dl, dt))


@api_view(["GET"])
def opciones_nueva_parcela(request):
    return Response(nueva_parcela.opciones(artefactos.obtener()))


@api_view(["POST"])
def predecir(request):
    """Estimacion para una parcela nueva: lat, lon, municipio, area_ha y clima opcional."""
    try:
        return Response(nueva_parcela.predecir(artefactos.obtener(), request.data))
    except nueva_parcela.ErrorEntrada as e:
        return _error(str(e), 400)
