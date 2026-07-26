# Visualizador web de resultados FHE

Ensayo interactivo —con formato de **artículo de investigación** (estilo
Distill)— sobre **el coste de la privacidad**: la comparación entre la inferencia
en claro (*plana*) y sobre datos cifrados (*homomórfica*, CKKS/TenSEAL) de todos
los experimentos guardados en la base de datos.

- **Frontend:** React + Vite + [ECharts](https://echarts.apache.org) (gráficas
  interactivas: hover, zoom, filtros, transiciones animadas).
- **Backend:** FastAPI, que expone la BD como JSON y sirve el frontend compilado.

Lee `resultados/resultados.db` en **modo solo-lectura**: no modifica nada del
pipeline ni de los datos.

## Cómo lanzarlo

Desde la raíz del repositorio:

```bash
./web/lanzar.sh            # compila el frontend si hace falta y sirve todo
```

Abre <http://localhost:8000> en el navegador. `Ctrl+C` para parar.
Para forzar recompilación del frontend: `./web/lanzar.sh --build`.

> Requisitos: `.venv` con FastAPI/uvicorn (ya instalados) y Node.js (para
> compilar el frontend). La primera vez, el script hace `npm install` solo.

## Qué se ve

Un único documento de scroll, con navegación lateral e índice, dividido en
secciones. Los **filtros de dataset** y los **selectores de métrica/experimento**
recalculan las gráficas al vuelo.

| Sección | Contenido |
|---|---|
| **Resumen** | Abstract + hallazgos clave (fidelidad, ralentización, precisión perdida). |
| **§1 Fidelidad** | Dispersión precisión plana vs homomórfica (diagonal = fidelidad) y pérdida por profundidad. |
| **§2 Coste temporal** | Tiempo (log), factor de ralentización y tiempo vs nº de muestras. |
| **§3 Recursos** | RAM pico plana vs homomórfica. |
| **§4 Métrica a métrica** | Cualquier métrica ML enfrentada + radar por experimento. |
| **§5 Explorador** | Matrices de confusión enfrentadas, métricas por clase, semillas y parámetros CKKS. |
| **§6 Datos y métodos** | Nota metodológica + tablas descargables en CSV. |

## Estructura del código

```
web/
├─ backend/
│  ├─ api.py        # FastAPI: endpoints JSON + sirve el frontend de dist/
│  └─ datos.py      # acceso solo-lectura a la BD -> DataFrames + métricas derivadas
├─ frontend/
│  ├─ src/
│  │  ├─ App.jsx        # orquesta las secciones del artículo
│  │  ├─ api.js         # cliente de la API
│  │  ├─ theme.js       # paleta y estilo base de ECharts
│  │  ├─ charts.js      # constructores de opciones ECharts (sin estado)
│  │  ├─ EChart.jsx     # wrapper React sobre ECharts
│  │  ├─ ui.jsx         # piezas presentacionales (Section, Figure, Head)
│  │  ├─ index.css      # estética de artículo de investigación
│  │  └─ components/    # Toc, ExperimentExplorer, DataTables
│  └─ dist/         # build de producción (lo sirve FastAPI; git-ignorado)
├─ lanzar.sh
└─ README.md
```

`datos.py` es puro (no importa FastAPI); pruébalo suelto con
`.venv/bin/python web/backend/datos.py`. `charts.js`/`theme.js` no dependen de
React.

## Endpoints de la API

| Ruta | Devuelve |
|---|---|
| `GET /api/estado` | ¿hay datos?, nº de experimentos, lista de datasets. |
| `GET /api/comparativa` | una fila por experimento (plana vs homomórfica + derivadas). |
| `GET /api/experimento/{id}` | detalle: matrices, métricas por clase, semillas, CKKS. |
| `GET /api/tabla/{nombre}` | tabla cruda (`experimentos`, `predicciones`, …). |

## Desarrollo

Con recarga en caliente (dos terminales):

```bash
# terminal 1 — backend
.venv/bin/python -m uvicorn api:app --app-dir web/backend --port 8000 --reload
# terminal 2 — frontend (Vite en :5173, redirige /api al backend)
cd web/frontend && npm run dev
```
