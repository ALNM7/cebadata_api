"""Endpoints de la API de CebaData.

Son los mismos que espera el dashboard (ver README):

  GET  /api/salud
  GET  /api/parcelas[?modelo=ensamble|ubicacion|mezcla]
  GET  /api/parcelas/<id>[?modelo=...]
  GET  /api/modelo/metricas
  POST /api/simulador
"""
import json
import math

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from api import simulador
from api.models import Corrida, Parcela, Prediccion
from pipeline import conf


def _error(mensaje, status):
    return JsonResponse({"error": mensaje}, status=status)


def _corrida():
    return Corrida.objects.filter(activa=True).first()


def _sin_datos():
    return _error("Sin resultados cargados. Ejecuta `python manage.py cargar_resultados`.", 503)


def _modelo(request, corrida):
    pedido = request.GET.get("modelo")
    if pedido is None:
        return corrida.modelo_final
    return pedido if pedido in conf.CANDIDATOS else None


def _parcela_json(p, pr):
    return {
        "id": p.id_poligono, "estado": p.estado, "municipio": p.municipio, "area_ha": round(p.area_ha, 1),
        "conjunto": p.conjunto,
        "rendimiento_obs": None if p.rendimiento_obs is None else round(p.rendimiento_obs, 2),
        "rendimiento_pred": round(pr.valor, 2), "intervalo_inf": round(pr.inferior, 2),
        "intervalo_sup": round(pr.superior, 2), "lat": round(p.lat, 5), "lon": round(p.lon, 5),
        "elevacion_m": round(p.elevacion_m), "pendiente_grados": round(p.pendiente_grados, 1),
    }


def _predicciones(modelo):
    return {x.parcela_id: x for x in Prediccion.objects.filter(modelo=modelo)}


@require_GET
def salud(request):
    corrida = _corrida()
    return JsonResponse({
        "estado": "ok", "resultados_cargados": corrida is not None,
        "modelo_final": corrida.modelo_final if corrida else None,
        "parcelas": Parcela.objects.count(),
    })


@require_GET
def parcelas(request):
    corrida = _corrida()
    if corrida is None:
        return _sin_datos()
    modelo = _modelo(request, corrida)
    if modelo is None:
        return _error(f"Modelo desconocido. Opciones: {', '.join(conf.CANDIDATOS)}", 400)
    preds = _predicciones(modelo)
    return JsonResponse([_parcela_json(p, preds[p.id_poligono]) for p in Parcela.objects.all()], safe=False)


def _haversine(lat1, lon1, lat2, lon2):
    R, p = 6371.0, math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


@require_GET
def parcela_detalle(request, id_poligono):
    corrida = _corrida()
    if corrida is None:
        return _sin_datos()
    modelo = _modelo(request, corrida)
    if modelo is None:
        return _error(f"Modelo desconocido. Opciones: {', '.join(conf.CANDIDATOS)}", 400)
    try:
        p = Parcela.objects.get(pk=id_poligono)
    except Parcela.DoesNotExist:
        return _error("Parcela no encontrada", 404)

    preds = _predicciones(modelo)
    todas = list(Parcela.objects.all())
    vecinos = []
    for q in sorted((q for q in todas if q.id_poligono != p.id_poligono),
                    key=lambda q: _haversine(p.lat, p.lon, q.lat, q.lon))[:5]:
        vecinos.append({
            "id": q.id_poligono, "municipio": q.municipio,
            "distancia_km": round(_haversine(p.lat, p.lon, q.lat, q.lon), 2),
            "rendimiento": round(q.rendimiento_obs if q.rendimiento_obs is not None else preds[q.id_poligono].valor, 2),
            "tipo": "observado" if q.rendimiento_obs is not None else "predicho",
        })

    pr = preds[p.id_poligono]
    explicacion = getattr(p, "explicacion", None)
    items = explicacion.items if explicacion else []
    otras = round(pr.valor - corrida.base - sum(i["contribucion"] for i in items), 3)

    return JsonResponse({
        "parcela": _parcela_json(p, pr),
        "modelo": modelo,
        "otros_enfoques": {x.modelo: round(x.valor, 2) for x in p.predicciones.all()},
        "serie": [{"fecha": s.fecha.isoformat(), "fuente": s.fuente, "indice": s.indice, "valor": s.valor}
                  for s in p.serie.all()],
        "clima": [{"mes": c.mes, "prec_mm": round(c.prec_mm), "tmax_c": round(c.tmax_c, 1), "tmin_c": round(c.tmin_c, 1)}
                  for c in p.clima.all()],
        "shap": items,
        "shap_otras": otras,
        "base": round(corrida.base, 2),
        "vecinos": vecinos,
    })


@require_GET
def metricas(request):
    corrida = _corrida()
    if corrida is None:
        return _sin_datos()
    return JsonResponse({**corrida.metricas, "rango_entrenamiento": corrida.rango_entrenamiento})


@csrf_exempt
@require_http_methods(["POST"])
def simular(request):
    corrida = _corrida()
    if corrida is None:
        return _sin_datos()
    try:
        cuerpo = json.loads(request.body or b"{}")
        dl = float(cuerpo.get("delta_lluvia_mayo_pct") or 0)
        dt = float(cuerpo.get("delta_tmin_abril_c") or 0)
    except (ValueError, TypeError, AttributeError):
        return _error("Cuerpo invalido: se espera JSON con id, delta_lluvia_mayo_pct y delta_tmin_abril_c", 400)
    id_poligono = cuerpo.get("id")
    modelo = cuerpo.get("modelo") or corrida.modelo_final
    if modelo not in conf.CANDIDATOS:
        return _error(f"Modelo desconocido. Opciones: {', '.join(conf.CANDIDATOS)}", 400)
    pr = Prediccion.objects.filter(parcela_id=id_poligono, modelo=modelo).first()
    if pr is None:
        return _error("Parcela no encontrada", 404)
    try:
        resultado = simulador.simular(
            id_poligono, {"pred": pr.valor, "inf": pr.inferior, "sup": pr.superior}, dl, dt,
            corrida.rango_entrenamiento, modelo)
    except simulador.ModeloNoDisponible as e:
        return _error(str(e), 503)
    return JsonResponse(resultado)
