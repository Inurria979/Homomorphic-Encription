"""
EXPORTADOR DEL SITIO ESTÁTICO
=============================
Vuelca a ficheros JSON exactamente lo que devuelve la API (web/backend/api.py),
para poder publicar la web en un hosting SOLO-ESTÁTICO (GitHub Pages) sin
servidor Python.

Es posible porque los resultados son de **solo lectura**: la BD no cambia cuando
alguien visita la web, así que las respuestas se pueden precalcular. Se importan
las propias funciones de `api.py` en vez de reimplementar las consultas, de modo
que el JSON servido es idéntico al que devolvería FastAPI.

Uso (desde la raíz del repo):
    .venv/bin/python web/exportar_estatico.py [--salida DIR]

Por defecto escribe en `web/frontend/public/datos/`, que Vite copia tal cual al
compilar. La estructura resultante imita las rutas de la API:

    datos/estado.json              <- /api/estado
    datos/comparativa.json         <- /api/comparativa
    datos/barrido.json             <- /api/barrido
    datos/experimentos.json        <- /api/experimentos
    datos/experimento/<id>.json    <- /api/experimento/<id>
    datos/tabla/<nombre>.json      <- /api/tabla/<nombre>
"""

import argparse
import json
import os
import sys

# api.py y datos.py viven en web/backend y se importan entre ellos por nombre
# ("import datos"), así que ese directorio tiene que estar en el path.
_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_AQUI, "backend"))

import api  # noqa: E402  (necesita el sys.path de arriba)

# Tablas que expone /api/tabla/<nombre>.
TABLAS = ["comparativa", "barrido_ckks", "experimentos", "predicciones",
          "semillas", "metricas_clase"]


def _escribir(destino, ruta_relativa, contenido):
    ruta = os.path.join(destino, ruta_relativa)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(ruta)


def exportar(destino):
    """Genera todos los JSON. Devuelve (nº de ficheros, bytes totales)."""
    os.makedirs(destino, exist_ok=True)
    n = total = 0

    estado = api.estado()
    total += _escribir(destino, "estado.json", estado); n += 1

    if not estado.get("hay_datos"):
        print("⚠️  No hay datos en la BD: solo se ha escrito estado.json")
        return n, total

    total += _escribir(destino, "comparativa.json", api.comparativa()); n += 1
    total += _escribir(destino, "barrido.json", api.barrido()); n += 1

    experimentos = api.experimentos()
    total += _escribir(destino, "experimentos.json", experimentos); n += 1

    # Detalle de cada experimento (lo que consume el explorador).
    for exp in experimentos:
        exp_id = exp.get("id")
        if exp_id is None:
            continue
        total += _escribir(destino, f"experimento/{int(exp_id)}.json",
                           api.experimento(int(exp_id)))
        n += 1

    # Tablas descargables.
    for nombre in TABLAS:
        try:
            total += _escribir(destino, f"tabla/{nombre}.json", api.tabla(nombre))
            n += 1
        except Exception as e:   # una tabla vacía no debe romper la exportación
            print(f"   (aviso) tabla '{nombre}' omitida: {e}")

    return n, total


def main():
    ap = argparse.ArgumentParser(description="Exporta la API a JSON estáticos.")
    ap.add_argument("--salida", default=os.path.join(_AQUI, "frontend", "public", "datos"),
                    help="Directorio de salida (default: web/frontend/public/datos)")
    args = ap.parse_args()

    n, total = exportar(args.salida)
    print(f"✅ {n} ficheros JSON ({total/1024:.0f} KB) en {args.salida}")


if __name__ == "__main__":
    main()
