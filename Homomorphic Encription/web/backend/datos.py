"""
CAPA DE ACCESO A DATOS DE LA WEB
================================
Lee la base de datos SQLite de resultados (la ÚNICA fuente de verdad del
proyecto, ver invariante 8 del CLAUDE.md) y la devuelve como DataFrames de
pandas listos para graficar. No escribe nada: abre la BD en modo lectura.

Es pura (no importa Streamlit) para poder probarla sin arrancar el servidor.
Las columnas derivadas ("coste de la privacidad") se calculan aquí una sola vez:
  - delta_accuracy       = precisión plana − homomórfica (pp perdidos al cifrar)
  - factor_ralentizacion = tiempo homomórfico / tiempo plano (cuántas veces más lento)
  - overhead_ram_gb      = RAM pico homomórfica − plana

Convenciones de unidades (heredadas de la BD, ver ResultadosStore):
  - accuracy, f1, precision, recall, roc_auc... -> PORCENTAJE (0-100)
  - mcc, cohen_kappa -> rango natural (-1..1);  log_loss -> natural (>=0)
  - tiempos en segundos;  RAM en GB.
"""

import json
import os
import sqlite3

import numpy as np
import pandas as pd

# Ruta a la BD: este fichero vive en web/backend/, la BD cuelga de resultados/
# en la raíz del repo (tres niveles arriba).
_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_BD_POR_DEFECTO = os.path.join(_RAIZ, "resultados", "resultados.db")

# Etiqueta legible para el tipo de predicción.
ETIQUETA_TIPO = {"plana": "Plana (sin cifrar)", "homomorfica": "Homomórfica (CKKS)"}

# --- Nombres presentables -------------------------------------------------
# Los directorios ("BreastCancer_PCA5_rg", "CKKS_BC_6-4-2") son identificadores
# técnicos, feos en una gráfica. Aquí se derivan nombres legibles a partir de los
# METADATOS del experimento (dataset, arquitectura, PCA, regularización), no del
# nombre de la carpeta: así son consistentes aunque cambie la convención.
DATASETS_CORTOS = {
    "Breast Cancer Wisconsin": ("Cáncer de mama", "Cáncer"),
    "Credit Approval": ("Aprobación de crédito", "Crédito"),
    "Obesity Levels (Eating Habits)": ("Niveles de obesidad", "Obesidad"),
    "CDC Diabetes Health Indicators": ("Diabetes (CDC)", "Diabetes"),
}


def _dataset_nombres(dataset_nombre):
    """('Breast Cancer Wisconsin') -> ('Cáncer de mama', 'Cáncer')."""
    nombre = (dataset_nombre or "").strip()
    if nombre in DATASETS_CORTOS:
        return DATASETS_CORTOS[nombre]
    # Desconocido: se usa tal cual, recortando para el nombre corto.
    return (nombre or "Dataset", (nombre or "Dataset").split()[0])


def _arquitectura_compacta(arquitectura):
    """'30 -> 16 -> 8 -> 2' -> '30→16→8→2'."""
    if not arquitectura:
        return ""
    return str(arquitectura).replace(" -> ", "→").replace("->", "→").replace(" ", "")


def nombres_experimento(fila):
    """
    Devuelve (nombre, titulo, subtitulo) presentables para un experimento.

      nombre    : corto, para ejes de gráficas   -> 'Cáncer 5→2 · L2'
      titulo    : descriptivo, para cabeceras     -> 'Cáncer de mama · red 5→2'
      subtitulo : detalles                        -> 'PCA a 5 componentes · regularización L2'

    `fila` es un dict/Series con dataset_nombre, arquitectura, pca_features,
    regularizacion y directorio.
    """
    largo, corto = _dataset_nombres(fila.get("dataset_nombre"))
    arq = _arquitectura_compacta(fila.get("arquitectura"))
    directorio = str(fila.get("directorio") or "")

    pca = fila.get("pca_features") or 0
    try:
        pca = int(pca)
    except (TypeError, ValueError):
        pca = 0
    rg = fila.get("regularizacion") or 0
    try:
        rg = float(rg)
    except (TypeError, ValueError):
        rg = 0.0

    # Distintivos que hacen único el nombre corto (hay experimentos que solo se
    # diferencian por la regularización o por ser del barrido de cifrado).
    marcas = []
    if rg > 0:
        marcas.append("L2")
    if directorio.startswith("CKKS_"):
        marcas.append("barrido")

    nombre = " · ".join([f"{corto} {arq}".strip()] + marcas)
    titulo = f"{largo} · red {arq}" if arq else largo

    detalles = []
    detalles.append(f"PCA a {pca} componentes" if pca else "todas las variables")
    if rg > 0:
        detalles.append(f"regularización L2 ({rg:g})")
    if directorio.startswith("CKKS_"):
        detalles.append("barrido de parámetros CKKS")
    return nombre, titulo, " · ".join(detalles)


def _conectar(ruta_bd):
    """Abre la BD en modo solo-lectura para no arriesgar los datos."""
    uri = f"file:{ruta_bd}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def bd_existe(ruta_bd=RUTA_BD_POR_DEFECTO):
    return os.path.isfile(ruta_bd)


# ----------------------------------------------------------------------
# Carga cruda de cada tabla
# ----------------------------------------------------------------------

def _añadir_nombres(df):
    """Añade las columnas nombre / titulo / subtitulo a un DataFrame de experimentos."""
    if df.empty:
        for c in ("nombre", "titulo", "subtitulo"):
            df[c] = []
        return df
    nombres = df.apply(lambda f: nombres_experimento(f), axis=1)
    df["nombre"] = [n[0] for n in nombres]
    df["titulo"] = [n[1] for n in nombres]
    df["subtitulo"] = [n[2] for n in nombres]
    return df


def cargar_experimentos(ruta_bd=RUTA_BD_POR_DEFECTO):
    with _conectar(ruta_bd) as con:
        df = pd.read_sql_query("SELECT * FROM experimentos ORDER BY id", con)
    return _añadir_nombres(df)


def cargar_predicciones(ruta_bd=RUTA_BD_POR_DEFECTO):
    """
    Predicciones enriquecidas con datos del experimento (dataset, arquitectura,
    directorio) para poder filtrar y etiquetar sin más JOINs.
    """
    with _conectar(ruta_bd) as con:
        df = pd.read_sql_query(
            """
            SELECT p.*,
                   e.dataset_nombre, e.arquitectura, e.directorio,
                   e.pca_features, e.num_clases AS exp_num_clases,
                   e.n_test, e.ratio_balance, e.regularizacion,
                   e.hidden_layers, e.input_size
            FROM predicciones p
            JOIN experimentos e ON p.experimento_id = e.id
            ORDER BY p.experimento_id, p.tipo
            """,
            con,
        )
    df["tipo_legible"] = df["tipo"].map(ETIQUETA_TIPO).fillna(df["tipo"])
    # Nº de capas ocultas = profundidad de la red (para explicar la degradación del cifrado).
    df["profundidad"] = df["hidden_layers"].apply(_contar_capas)
    df = _añadir_nombres(df)
    # Etiqueta de la configuración CKKS de cada fila homomórfica: con el barrido
    # hay VARIAS por experimento y hay que poder distinguirlas ("g5 · 2^31").
    df["config_ckks"] = df.apply(_etiqueta_config, axis=1)
    return solo_optimas(df)


def solo_optimas(df):
    """
    Deja UNA sola predicción cifrada por experimento: la de mayor escala, que es
    la configuración óptima que calcula PrediccionHomomorfica (maximiza la escala
    dentro del presupuesto de bits que permite la seguridad).

    La BD conserva además corridas con otras configuraciones -barridos de grado x
    escala, controles para comprobar si un anillo mayor compensa- que sirven para
    el análisis pero no para la web, donde cada red debe aparecer una vez con su
    configuración definitiva. Las predicciones planas no se tocan.
    """
    if df.empty or "tipo" not in df.columns:
        return df
    cifradas = df[df["tipo"] == "homomorfica"]
    if cifradas.empty:
        return df
    # La última de cada experimento tras ordenar = mayor escala; a igualdad, mayor
    # anillo, y a igualdad, la más reciente.
    mejores = (cifradas
               .sort_values(["global_scale_bits", "poly_modulus_degree", "id"])
               .groupby("experimento_id", as_index=False)
               .tail(1))
    return df[(df["tipo"] != "homomorfica") | df["id"].isin(mejores["id"])]


def _etiqueta_config(fila):
    """'g5 · 2^31' para las homomórficas; vacío para las planas."""
    if fila.get("tipo") != "homomorfica":
        return ""
    grado = fila.get("grado_taylor")
    escala = fila.get("global_scale_bits")
    partes = []
    if grado is not None and str(grado) != "nan":
        partes.append(f"grado {grado}")
    if escala is not None and not (isinstance(escala, float) and np.isnan(escala)):
        partes.append(f"2^{int(escala)}")
    return " · ".join(partes)


def cargar_semillas(ruta_bd=RUTA_BD_POR_DEFECTO):
    with _conectar(ruta_bd) as con:
        df = pd.read_sql_query(
            "SELECT * FROM entrenamientos_semilla ORDER BY experimento_id, semilla", con)
    return df


def cargar_metricas_clase(ruta_bd=RUTA_BD_POR_DEFECTO):
    with _conectar(ruta_bd) as con:
        df = pd.read_sql_query(
            "SELECT * FROM metricas_clase ORDER BY experimento_id, tipo, clase", con)
    df["tipo_legible"] = df["tipo"].map(ETIQUETA_TIPO).fillna(df["tipo"])
    return df


# ----------------------------------------------------------------------
# Vistas derivadas
# ----------------------------------------------------------------------

# Métricas que tienen sentido comparar plana vs homomórfica lado a lado.
METRICAS_COMPARABLES = [
    ("accuracy", "Accuracy (%)", True),
    ("balanced_accuracy", "Balanced accuracy (%)", True),
    ("f1_macro", "F1 macro (%)", True),
    ("f1_weighted", "F1 weighted (%)", True),
    ("precision_macro", "Precision macro (%)", True),
    ("recall_macro", "Recall macro (%)", True),
    ("mcc", "MCC (-1..1)", False),
    ("cohen_kappa", "Cohen kappa (-1..1)", False),
]


def tabla_comparativa(df_pred=None, ruta_bd=RUTA_BD_POR_DEFECTO):
    """
    Una fila por experimento con las columnas de plana y homomórfica
    enfrentadas y las métricas derivadas del coste de la privacidad.
    """
    if df_pred is None:
        df_pred = cargar_predicciones(ruta_bd)

    # Columnas que queremos enfrentar plana vs homomórfica.
    metricas = [
        "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro", "mcc", "cohen_kappa",
        "roc_auc", "log_loss", "tiempo_wall_s", "tiempo_inferencia",
        "tiempo_por_muestra_ms", "throughput", "ram_pico_proceso_gb",
        "hilos_pico", "eficiencia_cpu", "cpu_time_total_s",
    ]
    fijas = [
        "experimento_id", "dataset_nombre", "arquitectura", "directorio",
        "pca_features", "exp_num_clases", "n_muestras", "profundidad",
        "regularizacion", "nombre", "titulo", "subtitulo",
    ]
    # De la homomórfica también interesan los parámetros de cifrado usados.
    ckks = ["grado_taylor", "global_scale_bits", "poly_modulus_degree",
            "rango_activacion", "poly_coeffs"]

    filas = []
    for exp_id, grupo in df_pred.groupby("experimento_id"):
        base = grupo.iloc[0]
        fila = {c: base[c] for c in fijas}
        for tipo in ("plana", "homomorfica"):
            sub = grupo[grupo["tipo"] == tipo]
            if sub.empty:
                continue
            # Con el barrido CKKS hay VARIAS homomórficas por experimento; para la
            # comparativa se toma una representativa (la configuración de referencia)
            # y el resto se consulta en la vista del barrido.
            r = _representativa(sub) if tipo == "homomorfica" else sub.iloc[0]
            sufijo = "_plana" if tipo == "plana" else "_homo"
            for m in metricas:
                fila[m + sufijo] = r[m]
            if tipo == "homomorfica":
                fila["n_configs_ckks"] = int(len(sub))
                for c in ckks:
                    fila[c] = r[c] if c in r else None
        filas.append(fila)

    df = pd.DataFrame(filas)
    if df.empty:
        return df

    # --- Métricas derivadas (el "coste de la privacidad") ---
    df["delta_accuracy"] = df["accuracy_plana"] - df["accuracy_homo"]
    df["delta_f1_macro"] = df["f1_macro_plana"] - df["f1_macro_homo"]
    df["delta_mcc"] = df["mcc_plana"] - df["mcc_homo"]
    df["overhead_ram_gb"] = df["ram_pico_proceso_gb_homo"] - df["ram_pico_proceso_gb_plana"]

    with np.errstate(divide="ignore", invalid="ignore"):
        df["factor_ralentizacion"] = df["tiempo_wall_s_homo"] / df["tiempo_wall_s_plana"]
        df["factor_throughput"] = df["throughput_plana"] / df["throughput_homo"]
    df["factor_ralentizacion"] = df["factor_ralentizacion"].replace([np.inf, -np.inf], np.nan)
    df["factor_throughput"] = df["factor_throughput"].replace([np.inf, -np.inf], np.nan)

    # Etiqueta de los ejes: el nombre presentable en vez del directorio técnico.
    df["etiqueta"] = df["nombre"]
    df = df.sort_values("experimento_id").reset_index(drop=True)
    return df


# Configuración CKKS de referencia para la comparativa general (la "óptima" que usa
# la batería). Debe coincidir con configuracion_ckks.GRADO_OPTIMO / ESCALA_OPTIMA.
GRADO_REFERENCIA, ESCALA_REFERENCIA = "3", 31


def _representativa(sub):
    """
    Elige UNA predicción homomórfica entre las N de un experimento (barrido CKKS).
    Prioriza la configuración de referencia para que la comparativa entre
    experimentos sea homogénea; si no está, la primera disponible.
    """
    if len(sub) == 1:
        return sub.iloc[0]
    exacta = sub[(sub["grado_taylor"].astype(str) == GRADO_REFERENCIA)
                 & (sub["global_scale_bits"] == ESCALA_REFERENCIA)]
    if not exacta.empty:
        return exacta.iloc[0]
    solo_grado = sub[sub["grado_taylor"].astype(str) == GRADO_REFERENCIA]
    return (solo_grado if not solo_grado.empty else sub).iloc[0]


def tabla_barrido_ckks(df_pred=None, ruta_bd=RUTA_BD_POR_DEFECTO):
    """
    Todas las predicciones homomórficas de los experimentos que tienen MÁS DE UNA
    (es decir, los del barrido de parámetros CKKS), con la precisión plana de su
    experimento como referencia. Una fila por configuración de cifrado probada.

    Es la vista que permite responder: ¿cómo afecta el grado del polinomio y la
    escala a la precisión y al coste?
    """
    if df_pred is None:
        df_pred = cargar_predicciones(ruta_bd)
    if df_pred.empty:
        return pd.DataFrame()

    homo = df_pred[df_pred["tipo"] == "homomorfica"]
    if homo.empty:
        return pd.DataFrame()
    # Solo experimentos con varias configuraciones (los barridos).
    cuenta = homo.groupby("experimento_id")["id"].count()
    ids = cuenta[cuenta > 1].index
    barrido = homo[homo["experimento_id"].isin(ids)].copy()
    if barrido.empty:
        return pd.DataFrame()

    # Precisión plana de cada experimento, como línea base de fidelidad.
    planas = (df_pred[df_pred["tipo"] == "plana"]
              .set_index("experimento_id")["accuracy"].to_dict())
    barrido["accuracy_plana"] = barrido["experimento_id"].map(planas)
    barrido["delta_accuracy"] = barrido["accuracy_plana"] - barrido["accuracy"]

    barrido["grado"] = pd.to_numeric(barrido["grado_taylor"], errors="coerce")
    barrido["n_primos"] = barrido["coeff_mod_bit_sizes"].apply(lambda t: len(parse_lista(t)))
    barrido["rango"] = barrido["rango_activacion"].apply(_texto_rango)

    columnas = [
        "id", "experimento_id", "nombre", "titulo", "directorio", "dataset_nombre",
        "arquitectura", "profundidad", "config_ckks", "grado", "global_scale_bits",
        "poly_modulus_degree", "n_primos", "coeff_mod_bit_sizes", "rango_activacion",
        "rango", "poly_coeffs", "accuracy", "accuracy_plana", "delta_accuracy",
        "f1_macro", "mcc", "tiempo_wall_s", "tiempo_inferencia", "throughput",
        "ram_pico_proceso_gb", "tiempo_contexto", "tiempo_cifrado",
        "tiempo_prediccion", "tiempo_descifrado", "n_muestras",
    ]
    columnas = [c for c in columnas if c in barrido.columns]
    return (barrido[columnas]
            .sort_values(["experimento_id", "grado", "global_scale_bits"])
            .reset_index(drop=True))


def _texto_rango(texto):
    """'[-8.69, 5.29]' -> '[−8.69, 5.29]' listo para mostrar."""
    lista = parse_lista(texto)
    if isinstance(lista, list) and len(lista) == 2:
        try:
            return f"[{float(lista[0]):.2f}, {float(lista[1]):.2f}]"
        except (TypeError, ValueError):
            return ""
    return ""


# ----------------------------------------------------------------------
# Utilidades de parseo de columnas JSON/texto
# ----------------------------------------------------------------------

def parse_matriz_confusion(texto):
    """'[[72, 0], [2, 40]]' -> np.array 2D. Devuelve None si no se puede."""
    if not texto or (isinstance(texto, float) and np.isnan(texto)):
        return None
    try:
        return np.array(json.loads(texto), dtype=int)
    except Exception:
        return None


def parse_distribucion(texto):
    """'{\"0\": 357, \"1\": 212}' -> {0: 357, 1: 212}. Tolerante a None y NaN."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return {}
    if not texto:
        return {}
    try:
        d = json.loads(texto)
        return {int(k): int(v) for k, v in d.items()}
    except Exception:
        return {}


def parse_lista(texto):
    """
    '[60, 31, ...]' -> list. Tolerante a None y a NaN.

    OJO con el NaN: es "truthy", así que un `if not texto` NO lo filtra y el valor
    acabaría colándose en la respuesta JSON (que no admite NaN). Las columnas
    opcionales de la BD llegan como NaN vía pandas cuando están a NULL —por
    ejemplo poly_coeffs en redes sin capas ocultas, que no tienen activación
    cifrada—, así que se descartan explícitamente.
    """
    if texto is None:
        return []
    if isinstance(texto, float) and np.isnan(texto):
        return []
    if isinstance(texto, (list, tuple)):
        return list(texto)
    if not texto:
        return []
    try:
        return json.loads(texto)
    except Exception:
        return [texto]


def _contar_capas(hidden_layers_texto):
    """Nº de capas ocultas a partir del campo hidden_layers ('[16, 10]' -> 2)."""
    lista = parse_lista(hidden_layers_texto)
    if isinstance(lista, list):
        return len([x for x in lista if isinstance(x, (int, float))])
    return 0


# ----------------------------------------------------------------------
# Prueba rápida por línea de comandos
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if not bd_existe():
        print("No hay BD en", RUTA_BD_POR_DEFECTO)
        raise SystemExit(1)
    exp = cargar_experimentos()
    pred = cargar_predicciones()
    comp = tabla_comparativa(pred)
    print(f"experimentos: {len(exp)}  predicciones: {len(pred)}  comparativa: {len(comp)} filas")
    print("\nCoste de la privacidad (resumen):")
    cols = ["directorio", "accuracy_plana", "accuracy_homo", "delta_accuracy",
            "factor_ralentizacion", "overhead_ram_gb"]
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(comp[cols].to_string(index=False))
