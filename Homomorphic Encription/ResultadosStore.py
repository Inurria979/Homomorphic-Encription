"""
ALMACENAMIENTO DE RESULTADOS (SQLite + Excel)
=============================================
La base de datos SQLite es la fuente de verdad; el Excel se regenera completo
desde la BD tras cada escritura, así ambos están siempre en sincronía.

Estructura de tablas:
    experimentos      -> un registro por entrenamiento (dataset, arquitectura,
                         hiperparámetros, tamaños de dataset, balance de clases)
    entrenamientos_semilla -> resultado de cada semilla probada
    predicciones      -> cada evaluación (plana u homomórfica) con un conjunto
                         AMPLIO de métricas de ML (accuracy, F1, MCC, kappa,
                         ROC-AUC...) y de SISTEMA (CPU, hilos, RAM, tiempos)
    metricas_clase    -> desglose por clase de cada predicción (formato largo)

Las tasas (accuracy, F1, precision...) se guardan en PORCENTAJE (0-100).
MCC y kappa en su rango natural (-1..1); log-loss en su valor bruto.

El esquema evoluciona por migración: _migrar() añade con ALTER TABLE cualquier
columna que falte en una BD antigua, así nunca se pierden datos anteriores.
"""

import json
import os
import sqlite3
import statistics
from datetime import datetime
from glob import glob


DATASET_NOMBRES = {
    17: "Breast Cancer Wisconsin",
    891: "CDC Diabetes Health Indicators",
    544: "Obesity Levels (Eating Habits)",
    27: "Credit Approval",
}


def _num(valor, tipo=float):
    """Convierte numpy/None a tipos nativos de Python para sqlite."""
    if valor is None:
        return None
    return tipo(valor)


def _pct(valor):
    """Tasa en fracción (0-1) -> porcentaje (0-100), respetando None."""
    return None if valor is None else float(valor) * 100


def _json_o_none(valor):
    if valor is None:
        return None
    if hasattr(valor, 'tolist'):
        valor = valor.tolist()
    return json.dumps(valor)


def _normalizar_dir(directorio):
    """'./Diabetes/', 'Diabetes' y '/abs/ruta/Diabetes' se registran igual."""
    return os.path.basename(os.path.abspath(directorio))


# Columnas nuevas por tabla (para la migración de BD antiguas). El CREATE de
# _crear_tablas ya las incluye todas; esto solo cubre bases de datos previas.
_COLUMNAS_EXPERIMENTOS = {
    'n_total': 'INTEGER', 'n_train': 'INTEGER', 'n_val': 'INTEGER',
    'n_test': 'INTEGER', 'n_features_original': 'INTEGER',
    'distribucion_clases': 'TEXT', 'ratio_balance': 'REAL',
    # Estadísticos del muestreo Monte Carlo
    'val_accuracy_media': 'REAL', 'val_accuracy_desviacion': 'REAL',
    'epocas_reentrenamiento': 'INTEGER',
}
_COLUMNAS_SEMILLAS = {
    'epoca_optima': 'INTEGER',
}
_COLUMNAS_PREDICCIONES = {
    # ML
    'balanced_accuracy': 'REAL', 'error_rate': 'REAL',
    'precision_macro': 'REAL', 'precision_micro': 'REAL', 'precision_weighted': 'REAL',
    'recall_macro': 'REAL', 'recall_micro': 'REAL', 'recall_weighted': 'REAL',
    'f1_macro': 'REAL', 'f1_micro': 'REAL', 'f1_weighted': 'REAL',
    'roc_auc': 'REAL', 'mcc': 'REAL', 'cohen_kappa': 'REAL', 'log_loss': 'REAL',
    'n_correctas': 'INTEGER', 'n_incorrectas': 'INTEGER', 'num_clases': 'INTEGER',
    'metricas_clase_json': 'TEXT',
    # Cifrado / paralelismo
    'poly_modulus_degree': 'INTEGER', 'coeff_mod_bit_sizes': 'TEXT',
    'global_scale_bits': 'INTEGER', 'grado_taylor': 'TEXT',
    'seguridad_efectiva': 'INTEGER',
    'rango_activacion': 'TEXT', 'poly_coeffs': 'TEXT',
    'batch_size': 'INTEGER', 'num_workers': 'INTEGER',
    'num_workers_efectivos': 'INTEGER', 'batch_size_efectivo': 'INTEGER',
    'tiempo_contexto': 'REAL', 'tiempo_cifrado': 'REAL',
    'tiempo_prediccion': 'REAL', 'tiempo_descifrado': 'REAL',
    # Sistema
    'ram_total_wsl_gb': 'REAL', 'ram_limite_gb': 'REAL', 'ram_pico_gb': 'REAL',
    'cpu_count_logico': 'INTEGER', 'cpu_count_fisico': 'INTEGER',
    'torch_threads': 'INTEGER', 'hilos_pico': 'INTEGER',
    'tiempo_wall_s': 'REAL', 'cpu_time_usuario_s': 'REAL',
    'cpu_time_sistema_s': 'REAL', 'cpu_time_total_s': 'REAL', 'eficiencia_cpu': 'REAL',
    'ram_inicial_gb': 'REAL', 'ram_media_gb': 'REAL', 'ram_pico_proceso_gb': 'REAL',
    'ram_disponible_min_gb': 'REAL', 'swap_pico_gb': 'REAL',
}


class ResultadosStore:

    def __init__(self, db_path=None, excel_path=None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.db_path = db_path or os.path.join(base, 'resultados', 'resultados.db')
        self.excel_path = excel_path or os.path.join(base, 'resultados', 'resultados.xlsx')

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._crear_tablas()

    # ------------------------------------------------------------------
    # Esquema
    # ------------------------------------------------------------------

    def _crear_tablas(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS experimentos (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha               TEXT,
            dataset_id          INTEGER,
            dataset_nombre      TEXT,
            directorio          TEXT NOT NULL,
            arquitectura        TEXT,
            input_size          INTEGER,
            hidden_layers       TEXT,
            num_clases          INTEGER,
            pca_features        INTEGER,
            learning_rate       REAL,
            regularizacion      REAL,
            num_semillas        INTEGER,
            mejor_semilla       INTEGER,
            mejor_val_accuracy  REAL,
            val_accuracy_media  REAL,
            val_accuracy_desviacion REAL,
            epocas_reentrenamiento  INTEGER,
            n_total             INTEGER,
            n_train             INTEGER,
            n_val               INTEGER,
            n_test              INTEGER,
            n_features_original INTEGER,
            distribucion_clases TEXT,
            ratio_balance       REAL,
            notas               TEXT
        );

        CREATE TABLE IF NOT EXISTS entrenamientos_semilla (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            experimento_id  INTEGER NOT NULL REFERENCES experimentos(id),
            semilla         INTEGER,
            val_accuracy    REAL,
            val_loss        REAL,
            epoca_optima    INTEGER
        );

        CREATE TABLE IF NOT EXISTS predicciones (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            experimento_id        INTEGER REFERENCES experimentos(id),
            fecha                 TEXT,
            tipo                  TEXT NOT NULL CHECK (tipo IN ('plana', 'homomorfica')),
            n_muestras            INTEGER,
            tiempo_inferencia     REAL,
            tiempo_por_muestra_ms REAL,
            throughput            REAL,
            accuracy              REAL,
            balanced_accuracy     REAL,
            error_rate            REAL,
            precision_macro       REAL,
            precision_micro       REAL,
            precision_weighted    REAL,
            recall_macro          REAL,
            recall_micro          REAL,
            recall_weighted       REAL,
            f1_macro              REAL,
            f1_micro              REAL,
            f1_weighted           REAL,
            sensibilidad          REAL,
            especificidad         REAL,
            ppv                   REAL,
            npv                   REAL,
            roc_auc               REAL,
            mcc                   REAL,
            cohen_kappa           REAL,
            log_loss              REAL,
            n_correctas           INTEGER,
            n_incorrectas         INTEGER,
            num_clases            INTEGER,
            tp                    INTEGER,
            tn                    INTEGER,
            fp                    INTEGER,
            fn                    INTEGER,
            matriz_confusion      TEXT,
            metricas_clase_json   TEXT,
            poly_modulus_degree   INTEGER,
            coeff_mod_bit_sizes   TEXT,
            global_scale_bits     INTEGER,
            grado_taylor          TEXT,
            seguridad_efectiva    INTEGER,
            rango_activacion      TEXT,
            poly_coeffs           TEXT,
            batch_size            INTEGER,
            batch_size_efectivo   INTEGER,
            num_workers           INTEGER,
            num_workers_efectivos INTEGER,
            tiempo_contexto       REAL,
            tiempo_cifrado        REAL,
            tiempo_prediccion     REAL,
            tiempo_descifrado     REAL,
            ram_total_wsl_gb      REAL,
            ram_limite_gb         REAL,
            ram_pico_gb           REAL,
            cpu_count_logico      INTEGER,
            cpu_count_fisico      INTEGER,
            torch_threads         INTEGER,
            hilos_pico            INTEGER,
            tiempo_wall_s         REAL,
            cpu_time_usuario_s    REAL,
            cpu_time_sistema_s    REAL,
            cpu_time_total_s      REAL,
            eficiencia_cpu        REAL,
            ram_inicial_gb        REAL,
            ram_media_gb          REAL,
            ram_pico_proceso_gb   REAL,
            ram_disponible_min_gb REAL,
            swap_pico_gb          REAL,
            notas                 TEXT
        );

        CREATE TABLE IF NOT EXISTS metricas_clase (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            prediccion_id   INTEGER NOT NULL REFERENCES predicciones(id),
            experimento_id  INTEGER,
            tipo            TEXT,
            clase           INTEGER,
            precision_pct   REAL,
            recall_pct      REAL,
            f1_pct          REAL,
            especificidad_pct REAL,
            soporte         INTEGER
        );
        """)
        self.conn.commit()
        self._migrar()

    def _columnas(self, tabla):
        return [fila[1] for fila in self.conn.execute(f"PRAGMA table_info({tabla})")]

    def _migrar(self):
        """Añade a bases de datos antiguas las columnas que les falten."""
        for tabla, nuevas in (('experimentos', _COLUMNAS_EXPERIMENTOS),
                              ('predicciones', _COLUMNAS_PREDICCIONES),
                              ('entrenamientos_semilla', _COLUMNAS_SEMILLAS)):
            existentes = set(self._columnas(tabla))
            for nombre, tipo in nuevas.items():
                if nombre not in existentes:
                    self.conn.execute(f"ALTER TABLE {tabla} ADD COLUMN {nombre} {tipo}")
        self.conn.commit()

    def _insertar(self, tabla, datos):
        """
        INSERT genérico: filtra `datos` a las columnas que existen en la tabla,
        así el esquema puede crecer sin tocar cada llamada. Devuelve el id.
        """
        cols = set(self._columnas(tabla))
        pares = [(k, v) for k, v in datos.items() if k in cols]
        nombres = ", ".join(k for k, _ in pares)
        marcadores = ", ".join("?" for _ in pares)
        cur = self.conn.execute(
            f"INSERT INTO {tabla} ({nombres}) VALUES ({marcadores})",
            [v for _, v in pares])
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Experimentos
    # ------------------------------------------------------------------

    def guardar_experimento(self, directorio, dataset_id=None, dataset_nombre=None,
                            input_size=None, hidden_layers=None, num_clases=None,
                            arquitectura=None, pca_features=0, learning_rate=None,
                            regularizacion=0.0, num_semillas=None, mejor_semilla=None,
                            mejor_val_accuracy=None, semillas_resultados=None,
                            epocas_reentrenamiento=None,
                            n_total=None, n_train=None, n_val=None, n_test=None,
                            n_features_original=None, distribucion_clases=None,
                            ratio_balance=None, fecha=None, notas=None, exportar=True):
        """
        Registra un experimento y sus semillas. Devuelve el id del experimento.
        semillas_resultados: lista de dicts con 'seed', 'val_accuracy' y opc.
        'val_loss' y 'best_epoch'.

        La media y la desviación de la accuracy de validación se calculan aquí a
        partir de las semillas: son los estadísticos del muestreo Monte Carlo y
        describen la estabilidad del entrenamiento. `mejor_val_accuracy` es solo
        el criterio de selección y está sesgado al alza, así que no debe usarse
        como estimación de rendimiento.
        """
        if dataset_nombre is None and dataset_id is not None:
            dataset_nombre = DATASET_NOMBRES.get(dataset_id, f'UCI id {dataset_id}')
        if num_semillas is None and semillas_resultados:
            num_semillas = len(semillas_resultados)

        accs = [a for a in (_num(r.get('val_accuracy'))
                            for r in (semillas_resultados or [])) if a is not None]
        val_media = statistics.mean(accs) if accs else None
        val_desviacion = statistics.pstdev(accs) if len(accs) > 1 else (0.0 if accs else None)

        experimento_id = self._insertar('experimentos', {
            'fecha': fecha or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_id': _num(dataset_id, int), 'dataset_nombre': dataset_nombre,
            'directorio': _normalizar_dir(directorio), 'arquitectura': arquitectura,
            'input_size': _num(input_size, int), 'hidden_layers': _json_o_none(hidden_layers),
            'num_clases': _num(num_clases, int), 'pca_features': _num(pca_features, int),
            'learning_rate': _num(learning_rate), 'regularizacion': _num(regularizacion),
            'num_semillas': _num(num_semillas, int), 'mejor_semilla': _num(mejor_semilla, int),
            'mejor_val_accuracy': _num(mejor_val_accuracy),
            'val_accuracy_media': val_media,
            'val_accuracy_desviacion': val_desviacion,
            'epocas_reentrenamiento': _num(epocas_reentrenamiento, int),
            'n_total': _num(n_total, int), 'n_train': _num(n_train, int),
            'n_val': _num(n_val, int), 'n_test': _num(n_test, int),
            'n_features_original': _num(n_features_original, int),
            'distribucion_clases': _json_o_none(distribucion_clases),
            'ratio_balance': _num(ratio_balance), 'notas': notas,
        })

        for res in (semillas_resultados or []):
            self._insertar('entrenamientos_semilla', {
                'experimento_id': experimento_id, 'semilla': _num(res.get('seed'), int),
                'val_accuracy': _num(res.get('val_accuracy')), 'val_loss': _num(res.get('val_loss')),
                'epoca_optima': _num(res.get('best_epoch'), int),
            })

        self.conn.commit()
        if exportar:
            self.exportar_excel()
        print(f"💾 Experimento #{experimento_id} guardado en BD: {self.db_path}")
        return experimento_id

    def obtener_experimento(self, directorio):
        """Devuelve el id del último experimento registrado para un directorio, o None."""
        fila = self.conn.execute(
            "SELECT id FROM experimentos WHERE directorio = ? ORDER BY id DESC LIMIT 1",
            (_normalizar_dir(directorio),)).fetchone()
        return fila[0] if fila else None

    def obtener_o_registrar_experimento(self, directorio, dataset_id=None, dataset_nombre=None,
                                        learning_rate=None, regularizacion=None,
                                        pca_features=None, exportar=True):
        """
        Para ejecuciones sin entrenamiento: recupera el experimento ya
        registrado para ese directorio o lo reconstruye desde los ficheros
        model_config.json / Nfeatures_*_experiment_info.json.
        """
        experimento_id = self.obtener_experimento(directorio)
        if experimento_id is not None:
            return experimento_id

        config = {}
        config_path = os.path.join(directorio, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)

        info = {}
        candidatos = sorted(glob(os.path.join(directorio, 'Nfeatures_*_experiment_info.json')))
        if candidatos:
            with open(candidatos[-1]) as f:
                info = json.load(f)

        return self.guardar_experimento(
            directorio=directorio, dataset_id=dataset_id, dataset_nombre=dataset_nombre,
            input_size=config.get('input_size'), hidden_layers=config.get('hidden_layers'),
            num_clases=config.get('num_classes'), arquitectura=config.get('architecture'),
            pca_features=info.get('features', pca_features), learning_rate=learning_rate,
            regularizacion=regularizacion, num_semillas=info.get('num_seeds_tried'),
            mejor_semilla=config.get('best_seed'), mejor_val_accuracy=config.get('best_val_accuracy'),
            epocas_reentrenamiento=(info.get('epocas_reentrenamiento')
                                    or config.get('epocas_reentrenamiento')),
            semillas_resultados=[{'seed': r.get('seed'), 'val_accuracy': r.get('val_accuracy'),
                                  'best_epoch': r.get('best_epoch')}
                                 for r in info.get('all_results', [])],
            fecha=info.get('timestamp'),
            notas='Registrado a partir de los ficheros del directorio', exportar=exportar)

    # ------------------------------------------------------------------
    # Predicciones
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_a_columnas(metrics):
        """Traduce el dict rico de Metricas (fracciones) a columnas de la BD (pct)."""
        if not metrics:
            return {}, None
        cols = {
            'accuracy': _pct(metrics.get('accuracy')),
            'balanced_accuracy': _pct(metrics.get('balanced_accuracy')),
            'error_rate': _pct(metrics.get('error_rate')),
            'precision_macro': _pct(metrics.get('precision_macro')),
            'precision_micro': _pct(metrics.get('precision_micro')),
            'precision_weighted': _pct(metrics.get('precision_weighted')),
            'recall_macro': _pct(metrics.get('recall_macro')),
            'recall_micro': _pct(metrics.get('recall_micro')),
            'recall_weighted': _pct(metrics.get('recall_weighted')),
            'f1_macro': _pct(metrics.get('f1_macro')),
            'f1_micro': _pct(metrics.get('f1_micro')),
            'f1_weighted': _pct(metrics.get('f1_weighted')),
            'sensibilidad': _pct(metrics.get('sensitivity')),
            'especificidad': _pct(metrics.get('specificity')),
            'ppv': _pct(metrics.get('ppv')),
            'npv': _pct(metrics.get('npv')),
            'roc_auc': _pct(metrics.get('roc_auc')),
            'mcc': _num(metrics.get('mcc')),
            'cohen_kappa': _num(metrics.get('cohen_kappa')),
            'log_loss': _num(metrics.get('log_loss')),
            'n_correctas': _num(metrics.get('n_correct'), int),
            'n_incorrectas': _num(metrics.get('n_incorrect'), int),
            'num_clases': _num(metrics.get('num_classes'), int),
            'tp': _num(metrics.get('tp'), int), 'tn': _num(metrics.get('tn'), int),
            'fp': _num(metrics.get('fp'), int), 'fn': _num(metrics.get('fn'), int),
            'matriz_confusion': _json_o_none(metrics.get('conf_matrix')),
            'metricas_clase_json': _json_o_none(metrics.get('per_class')),
        }
        return cols, metrics.get('per_class')

    def guardar_prediccion(self, tipo, metrics=None, sistema=None, experimento_id=None,
                           n_muestras=None, tiempo_inferencia=None, poly_modulus_degree=None,
                           coeff_mod_bit_sizes=None, global_scale_bits=None,
                           grado_taylor=None, seguridad_efectiva=None,
                           rango_activacion=None, poly_coeffs=None,
                           batch_size=None, num_workers=None,
                           tiempo_contexto=None, tiempo_cifrado=None,
                           tiempo_prediccion=None, tiempo_descifrado=None,
                           ram_total_wsl_gb=None, ram_limite_gb=None, ram_pico_gb=None,
                           num_workers_efectivos=None, batch_size_efectivo=None,
                           fecha=None, notas=None, exportar=True, **metricas_sueltas):
        """
        Registra una predicción ('plana' u 'homomorfica'). Devuelve su id.

        metrics : dict rico de Metricas.calcular_metricas (tasas en fracción).
        sistema : dict de MonitorSistema.resumen() (métricas de CPU/RAM/hilos).
        metricas_sueltas: por compatibilidad con importar_resultados.py, que
                          pasa métricas ya en porcentaje como kwargs sueltos.
        """
        cols_metricas, por_clase = self._metrics_a_columnas(metrics)

        # Métricas sueltas (importador): ya vienen en pct y con nombres de columna
        for k, v in metricas_sueltas.items():
            if k == 'matriz_confusion':
                cols_metricas[k] = _json_o_none(v)
            elif k in ('tp', 'tn', 'fp', 'fn', 'n_correctas', 'n_incorrectas', 'num_clases'):
                cols_metricas[k] = _num(v, int)
            else:
                cols_metricas[k] = _num(v)

        tiempo_por_muestra_ms = throughput = None
        if tiempo_inferencia and n_muestras:
            tiempo_por_muestra_ms = float(tiempo_inferencia) / float(n_muestras) * 1000
            throughput = float(n_muestras) / float(tiempo_inferencia)

        datos = {
            'experimento_id': _num(experimento_id, int),
            'fecha': fecha or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'tipo': tipo, 'n_muestras': _num(n_muestras, int),
            'tiempo_inferencia': _num(tiempo_inferencia),
            'tiempo_por_muestra_ms': tiempo_por_muestra_ms, 'throughput': throughput,
            'poly_modulus_degree': _num(poly_modulus_degree, int),
            'coeff_mod_bit_sizes': _json_o_none(coeff_mod_bit_sizes),
            'global_scale_bits': _num(global_scale_bits, int),
            'grado_taylor': str(grado_taylor) if grado_taylor is not None else None,
            'seguridad_efectiva': _num(seguridad_efectiva, int),
            'rango_activacion': _json_o_none(rango_activacion),
            'poly_coeffs': _json_o_none(poly_coeffs),
            'batch_size': _num(batch_size, int), 'num_workers': _num(num_workers, int),
            'num_workers_efectivos': _num(num_workers_efectivos, int),
            'batch_size_efectivo': _num(batch_size_efectivo, int),
            'tiempo_contexto': _num(tiempo_contexto), 'tiempo_cifrado': _num(tiempo_cifrado),
            'tiempo_prediccion': _num(tiempo_prediccion), 'tiempo_descifrado': _num(tiempo_descifrado),
            'ram_total_wsl_gb': _num(ram_total_wsl_gb), 'ram_limite_gb': _num(ram_limite_gb),
            'ram_pico_gb': _num(ram_pico_gb), 'notas': notas,
        }
        datos.update(cols_metricas)
        if sistema:
            for k, v in sistema.items():
                datos[k] = _num(v, int) if k in ('cpu_count_logico', 'cpu_count_fisico',
                                                 'torch_threads', 'hilos_pico') else _num(v)

        prediccion_id = self._insertar('predicciones', datos)

        for c in (por_clase or []):
            self._insertar('metricas_clase', {
                'prediccion_id': prediccion_id, 'experimento_id': _num(experimento_id, int),
                'tipo': tipo, 'clase': _num(c.get('clase'), int),
                'precision_pct': _pct(c.get('precision')), 'recall_pct': _pct(c.get('recall')),
                'f1_pct': _pct(c.get('f1')), 'especificidad_pct': _pct(c.get('especificidad')),
                'soporte': _num(c.get('soporte'), int),
            })

        self.conn.commit()
        if exportar:
            self.exportar_excel()
        print(f"💾 Predicción '{tipo}' #{prediccion_id} guardada en BD: {self.db_path}")
        return prediccion_id

    # ------------------------------------------------------------------
    # Exportación a Excel
    # ------------------------------------------------------------------

    def _tabla(self, sql):
        cur = self.conn.execute(sql)
        cabeceras = [c[0] for c in cur.description]
        return cabeceras, cur.fetchall()

    def exportar_excel(self):
        """Regenera el Excel completo (datos + gráficas + matrices) desde la BD."""
        try:
            from openpyxl import Workbook
        except ImportError:
            print("⚠️  openpyxl no está instalado: se omite el Excel (la BD sí se ha guardado).")
            print("    Instálalo con: pip install openpyxl")
            return None

        from ExcelReport import construir_excel
        try:
            construir_excel(self, self.excel_path)
        except PermissionError:
            print(f"⚠️  No se pudo escribir {self.excel_path} (¿abierto en Excel?). "
                  "La BD sí está actualizada; ciérralo y vuelve a exportar.")
            return None
        print(f"📗 Excel actualizado: {self.excel_path}")
        return self.excel_path

    def cerrar(self):
        self.conn.close()
