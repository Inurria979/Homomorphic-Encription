#!/usr/bin/env bash
# =====================================================================
# Lanza el visualizador web de resultados FHE (React + FastAPI).
# Compila el frontend si hace falta y sirve todo (web + API) en un puerto.
#
#   ./web/lanzar.sh            arranca en http://localhost:8000
#   ./web/lanzar.sh --build    fuerza recompilar el frontend antes de arrancar
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # raíz del repo, sea cual sea el CWD

PUERTO="${PUERTO:-8000}"
FORZAR_BUILD=0
[[ "${1:-}" == "--build" ]] && FORZAR_BUILD=1

# --- 1. Frontend: instalar dependencias y compilar si es necesario ---
if [[ ! -d web/frontend/node_modules ]]; then
    echo "▶ Instalando dependencias del frontend (npm install)…"
    ( cd web/frontend && npm install )
fi
# El frontend se compila de dos formas distintas: para servirlo aquí (rutas en la
# raíz y datos vía API) o para GitHub Pages (rutas bajo /<repo>/ y datos en JSON
# estáticos). Si en dist quedó un build de Pages, el navegador pediría los assets
# a /<repo>/... y la página saldría EN BLANCO, así que se detecta y se recompila.
DIST_INDEX=web/frontend/dist/index.html
if [[ -f $DIST_INDEX ]] && grep -q 'src="/[^"]*/assets/' "$DIST_INDEX"; then
    echo "▶ El frontend compilado es de GitHub Pages: se recompila para local…"
    FORZAR_BUILD=1
fi
if [[ ! -d web/frontend/dist || $FORZAR_BUILD -eq 1 ]]; then
    echo "▶ Compilando el frontend (npm run build)…"
    # Sin VITE_BASE ni VITE_DATOS_ESTATICOS: rutas en la raíz y datos vía API.
    ( cd web/frontend && VITE_BASE=/ VITE_DATOS_ESTATICOS=0 npm run build )
fi

# --- 2. Backend: FastAPI sirve la API y el frontend compilado ---
echo "▶ Sirviendo en http://localhost:${PUERTO}  (Ctrl+C para parar)"
exec .venv/bin/python -m uvicorn api:app \
    --app-dir web/backend \
    --host localhost \
    --port "${PUERTO}"
