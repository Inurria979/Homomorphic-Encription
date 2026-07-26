// Cliente ligero de la API.
//
// Funciona en dos modos:
//  - CON BACKEND (desarrollo y ejecución local): llama a /api/... del servidor
//    FastAPI. En dev, Vite redirige /api al puerto 8000.
//  - ESTÁTICO (GitHub Pages): no hay servidor Python, así que lee los JSON
//    precalculados por web/exportar_estatico.py. Se activa compilando con
//    VITE_DATOS_ESTATICOS=1. Como los resultados son de solo lectura, ambos
//    modos devuelven exactamente lo mismo.
//
// BASE_URL respeta el subdirectorio en el que se publique el sitio
// (p. ej. /FHE-Breast-Cancer-Detection/ en GitHub Pages).

const ESTATICO = import.meta.env.VITE_DATOS_ESTATICOS === '1'
const BASE = import.meta.env.BASE_URL || '/'
const DATOS = `${BASE.replace(/\/$/, '')}/datos`

async function get(ruta) {
  const r = await fetch(ruta)
  if (!r.ok) throw new Error(`${ruta} -> ${r.status}`)
  return r.json()
}

const ruta = (endpoint, estatico) => (ESTATICO ? `${DATOS}/${estatico}` : `/api/${endpoint}`)

export const api = {
  estado: () => get(ruta('estado', 'estado.json')),
  comparativa: () => get(ruta('comparativa', 'comparativa.json')),
  barrido: () => get(ruta('barrido', 'barrido.json')),
  experimentos: () => get(ruta('experimentos', 'experimentos.json')),
  experimento: (id) => get(ruta(`experimento/${id}`, `experimento/${id}.json`)),
  tabla: (nombre) => get(ruta(`tabla/${nombre}`, `tabla/${nombre}.json`)),
}
