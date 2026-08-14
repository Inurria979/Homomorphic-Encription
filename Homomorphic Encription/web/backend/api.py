"""
API DEL VISUALIZADOR FHE (FastAPI)
==================================
Sirve en JSON los resultados de la BD (a través de datos.py, en modo lectura) y,
en producción, los ficheros estáticos del frontend React compilado
(web/frontend/dist). El frontend consume estos endpoints.

Arrancar (desde la raíz del repo):
    .venv/bin/python -m uvicorn api:app --app-dir web/backend --port 8000
o simplemente:
    ./web/lanzar.sh
"""

import json
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import datos as D

app = FastAPI(title="FHE · Coste de la Privacidad", version="2.0")

# En desarrollo el frontend corre en el dev-server de Vite (5173) y llama a esta
# API en otro puerto: hay que permitir CORS desde ahí.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "frontend", "dist")


# ----------------------------------------------------------------------
# Utilidades de serialización (NaN -> null, tipos numpy -> nativos)
# ----------------------------------------------------------------------

def _registros(df):
    """DataFrame -> lista de dicts con NaN convertidos a null (vía pandas to_json)."""
    return json.loads(df.to_json(orient="records"))


def _valor(v):
    """Escala un valor suelto a algo serializable en JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


# ----------------------------------------------------------------------
# Carga perezosa y cacheada de los datos
# ----------------------------------------------------------------------
_cache = {}


def _datos():
    if not _cache:
        if not D.bd_existe():
            raise HTTPException(status_code=503,
                                detail="No hay base de datos de resultados.")
        _cache["exp"] = D.cargar_experimentos()
        _cache["pred"] = D.cargar_predicciones()
        _cache["comp"] = D.tabla_comparativa(_cache["pred"])
        _cache["sem"] = D.cargar_semillas()
        # Las métricas por clase se recortan a las predicciones que han quedado
        # (ver datos.solo_optimas), o una red barrida aportaría varias tandas
        _cache["mc"] = D.cargar_metricas_clase()
        _cache["mc"] = _cache["mc"][_cache["mc"]["prediccion_id"].isin(_cache["pred"]["id"])]
        _cache["barrido"] = D.tabla_barrido_ckks(_cache["pred"])
    return _cache


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/api/estado")
def estado():
    """¿Hay datos? Metadatos ligeros para la cabecera."""
    if not D.bd_existe():
        return {"hay_datos": False}
    d = _datos()
    comp = d["comp"]
    barrido = d["barrido"]
    return {
        "hay_datos": True,
        "n_experimentos": int(len(comp)),
        "n_predicciones": int(len(d["pred"])),
        "n_semillas": int(len(d["sem"])),
        "datasets": sorted(comp["dataset_nombre"].unique().tolist()),
        # Barrido de parámetros CKKS: cuántas configuraciones de cifrado se probaron
        "n_configs_ckks": int(len(barrido)),
        "hay_barrido": bool(len(barrido)),
    }


@app.get("/api/barrido")
def barrido():
    """
    Barrido de parámetros CKKS: una fila por configuración de cifrado probada
    (grado del polinomio × escala) en los experimentos que se barrieron.
    """
    return _registros(_datos()["barrido"])


@app.get("/api/comparativa")
def comparativa():
    """
    Tabla comparativa (una fila por experimento) con las métricas plana vs
    homomórfica y las derivadas del coste de la privacidad. Es la fuente de
    casi todas las gráficas del frontend.
    """
    d = _datos()
    return _registros(d["comp"])


@app.get("/api/experimentos")
def experimentos():
    return _registros(_datos()["exp"])


@app.get("/api/experimento/{exp_id}")
def experimento(exp_id: int):
    """Detalle completo de un experimento para la vista de exploración."""
    d = _datos()
    exp = d["exp"]
    fila_exp = exp[exp["id"] == exp_id]
    if fila_exp.empty:
        raise HTTPException(status_code=404, detail="Experimento no encontrado.")

    pred = d["pred"]
    preds_exp = pred[pred["experimento_id"] == exp_id]

    METRICAS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
                "precision_macro", "recall_macro", "mcc", "cohen_kappa",
                "roc_auc", "log_loss", "tiempo_wall_s", "tiempo_inferencia",
                "tiempo_por_muestra_ms", "throughput", "ram_pico_proceso_gb",
                "hilos_pico", "eficiencia_cpu", "cpu_time_total_s", "n_muestras"]
    # Parámetros de cifrado, incluidos el rango de las pre-activaciones y los
    # coeficientes del polinomio ajustado (columnas nuevas de la BD).
    CKKS = ["poly_modulus_degree", "coeff_mod_bit_sizes", "global_scale_bits",
            "grado_taylor", "rango_activacion", "poly_coeffs",
            "batch_size_efectivo", "num_workers_efectivos", "tiempo_contexto",
            "tiempo_cifrado", "tiempo_prediccion", "tiempo_descifrado"]

    def _fila(r, con_ckks):
        d = {
            "matriz_confusion": _matriz(r["matriz_confusion"]),
            "metricas": {k: _valor(r[k]) for k in METRICAS},
            "ckks": {k: _valor(r[k]) for k in CKKS} if con_ckks else None,
        }
        if con_ckks:
            d["config_ckks"] = r.get("config_ckks") or ""
            d["prediccion_id"] = _valor(r["id"])
            d["rango"] = D.parse_lista(r["rango_activacion"])
            d["coeficientes"] = D.parse_lista(r["poly_coeffs"])
        return d

    predicciones = {}
    plana = preds_exp[preds_exp["tipo"] == "plana"]
    if not plana.empty:
        predicciones["plana"] = _fila(plana.iloc[0], False)

    # Con el barrido CKKS hay VARIAS homomórficas: se devuelven todas (lista
    # `homomorficas`) y además una representativa en `homomorfica` para que las
    # vistas que solo comparan plana vs cifrada sigan funcionando.
    homo = preds_exp[preds_exp["tipo"] == "homomorfica"]
    homomorficas = []
    if not homo.empty:
        homo = homo.sort_values(["grado_taylor", "global_scale_bits"])
        homomorficas = [_fila(r, True) for _, r in homo.iterrows()]
        predicciones["homomorfica"] = _fila(D._representativa(homo), True)

    mc = d["mc"]
    mc_exp = mc[mc["experimento_id"] == exp_id]
    sem = d["sem"]
    sem_exp = sem[sem["experimento_id"] == exp_id]

    info = {k: _valor(fila_exp.iloc[0][k]) for k in exp.columns}
    info["distribucion_clases"] = D.parse_distribucion(fila_exp.iloc[0]["distribucion_clases"])

    return {
        "info": info,
        "predicciones": predicciones,
        "homomorficas": homomorficas,          # todas las configuraciones CKKS probadas
        "metricas_clase": _registros(mc_exp),
        "semillas": _registros(sem_exp.sort_values("val_accuracy", ascending=False)),
    }


@app.get("/api/tabla/{nombre}")
def tabla(nombre: str):
    """Tablas crudas para la sección de datos descargables."""
    d = _datos()
    mapa = {
        "experimentos": d["exp"],
        "predicciones": d["pred"],
        "semillas": d["sem"],
        "metricas_clase": d["mc"],
        "comparativa": d["comp"],
        "barrido_ckks": d["barrido"],
    }
    if nombre not in mapa:
        raise HTTPException(status_code=404, detail="Tabla desconocida.")
    return _registros(mapa[nombre])


def _matriz(texto):
    m = D.parse_matriz_confusion(texto)
    return m.tolist() if m is not None else None


# ----------------------------------------------------------------------
# Servir el frontend compilado (si existe). Debe ir DESPUÉS de las rutas /api.
# ----------------------------------------------------------------------
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def _sin_build():
        return JSONResponse(
            {"mensaje": "Frontend sin compilar. Ejecuta './web/lanzar.sh' o, en "
                        "desarrollo, 'npm run dev' dentro de web/frontend."},
            status_code=200)
