"""
IMPORTADOR DE RESULTADOS EXISTENTES
===================================
Recorre las carpetas de experimentos ya generadas (las que contienen
model_config.json) y vuelca a la BD SQLite + Excel:

  - model_config.json / Nfeatures_*_experiment_info.json  -> tabla experimentos + semillas
  - pred                                                  -> predicción 'plana'
  - pred_hom*  (excepto .png)                             -> predicciones 'homomorfica'

Uso:
    python importar_resultados.py                # escanea todas las carpetas del proyecto
    python importar_resultados.py Diabetes ...   # solo las carpetas indicadas

Es idempotente: si un directorio ya está registrado en la BD se salta.
"""

import os
import re
import sys

from ResultadosStore import ResultadosStore

RAIZ = os.path.dirname(os.path.abspath(__file__))

# Inferencia del dataset a partir del nombre de la carpeta
PREFIJOS_DATASET = [
    (('breast', 'bc'), (17, 'Breast Cancer Wisconsin')),
    (('directorio_test',), (17, 'Breast Cancer Wisconsin')),
    (('diabetes',), (891, 'CDC Diabetes Health Indicators')),
    (('obesity',), (544, 'Obesity Levels (Eating Habits)')),
    (('credit',), (27, 'Credit Approval')),
]


def inferir_dataset(nombre_dir):
    nombre = os.path.basename(nombre_dir.rstrip('/')).lower()
    for prefijos, (ds_id, ds_nombre) in PREFIJOS_DATASET:
        if nombre.startswith(prefijos):
            return ds_id, ds_nombre
    return None, None


def _buscar(patron, texto, tipo=float):
    m = re.search(patron, texto)
    return tipo(m.group(1)) if m else None


def parsear_matriz(texto):
    """Extrae la matriz de confusión (binaria o multiclase) del fichero de texto."""
    # Binaria: "Real  Benigno     71       1" / "      Maligno      3      39"
    m = re.search(r"Real\s+\S+\s+(\d+)\s+(\d+)\s*\n\s+\S+\s+(\d+)\s+(\d+)", texto)
    if m:
        a, b, c, d = (int(x) for x in m.groups())
        return [[a, b], [c, d]]

    # Multiclase: líneas "Cl0   0     0     54 ..."
    filas = re.findall(r"^Cl\d+\s+((?:\d+\s+)+)$", texto, flags=re.MULTILINE)
    if filas:
        return [[int(v) for v in fila.split()] for fila in filas]
    return None


def parsear_pred(texto):
    """Extrae métricas y tiempos de un fichero pred / pred_hom."""
    datos = {
        'fecha': _buscar(r"Fecha:\s*(.+)", texto, str),
        'tiempo_inferencia': _buscar(r"Tiempo total de inferencia:\s*([\d.]+)\s*segundos", texto),
        'n_muestras': _buscar(r"Muestras procesadas:\s*([\d.]+)", texto),
        'accuracy': _buscar(r"Accuracy:\s*([\d.]+)%", texto),
        'matriz_confusion': parsear_matriz(texto),
        # Binarias
        'sensibilidad': _buscar(r"Sensibilidad[^:]*:\s*([\d.]+)%", texto),
        'especificidad': _buscar(r"Especificidad[^:]*:\s*([\d.]+)%", texto),
        'ppv': _buscar(r"Valor Predictivo Positivo \(PPV\):\s*([\d.]+)%", texto),
        'npv': _buscar(r"Valor Predictivo Negativo \(NPV\):\s*([\d.]+)%", texto),
        'tp': _buscar(r"Verdaderos Positivos \(TP\):\s*(\d+)", texto, int),
        'tn': _buscar(r"Verdaderos Negativos \(TN\):\s*(\d+)", texto, int),
        'fp': _buscar(r"Falsos Positivos \(FP\):\s*(\d+)", texto, int),
        'fn': _buscar(r"Falsos Negativos \(FN\):\s*(\d+)", texto, int),
    }
    if datos['n_muestras'] is not None:
        datos['n_muestras'] = int(datos['n_muestras'])

    # Multiclase: macro recall / precisión / NPV clase 0
    if datos['sensibilidad'] is None:
        datos['sensibilidad'] = _buscar(r"Recall \(Promedio\):\s*([\d.]+)%", texto)
        datos['ppv'] = datos['ppv'] or _buscar(r"Precisión \(PPV Promedio\):\s*([\d.]+)%", texto)
        datos['npv'] = datos['npv'] or _buscar(r"NPV \(Referencia Clase 0\):\s*([\d.]+)%", texto)

    # Anotaciones manuales tras el pie del fichero -> notas + parámetros de cifrado
    partes = re.split(r"={60,}\s*\n", texto)
    notas = partes[-1].strip() if len(partes) > 1 else ''
    if notas:
        datos['notas'] = notas
        datos['poly_modulus_degree'] = _buscar(r"poly_modulus_degree\s*=\s*(\d+)", notas, int)
        datos['batch_size'] = _buscar(r"batch_size de (\d+)", notas, int)
        datos['num_workers'] = _buscar(r"max workers de (\d+)", notas, int)
        datos['grado_taylor'] = _buscar(r"(?:polu|poly|polinomio)\s*degree\s*=?\s*(\d+)", notas, str)
    return datos


def importar_directorio(store, directorio):
    nombre = os.path.basename(directorio.rstrip('/'))

    if store.obtener_experimento(directorio):
        print(f"⏭️  {nombre}: ya registrado, se salta.")
        return

    dataset_id, dataset_nombre = inferir_dataset(nombre)
    experimento_id = store.obtener_o_registrar_experimento(
        directorio=directorio, dataset_id=dataset_id,
        dataset_nombre=dataset_nombre, exportar=False)
    print(f"📂 {nombre}: experimento #{experimento_id}")

    for fichero in sorted(os.listdir(directorio)):
        ruta = os.path.join(directorio, fichero)
        if not os.path.isfile(ruta) or fichero.endswith('.png'):
            continue
        if fichero == 'pred':
            tipo = 'plana'
        elif fichero.startswith('pred_hom'):
            tipo = 'homomorfica'
        else:
            continue

        with open(ruta, encoding='utf-8', errors='replace') as f:
            texto = f.read()
        datos = parsear_pred(texto)
        if datos['accuracy'] is None:
            print(f"   ⚠️  {fichero}: no se pudo parsear, se salta.")
            continue

        # El grado del polinomio a veces solo está en el nombre del fichero (pred_hom_grado3)
        if tipo == 'homomorfica' and not datos.get('grado_taylor'):
            m = re.search(r"grado(\d+)", fichero)
            if m:
                datos['grado_taylor'] = m.group(1)

        nota_origen = f"Importado de {nombre}/{fichero}"
        datos['notas'] = f"{nota_origen} | {datos['notas']}" if datos.get('notas') else nota_origen

        store.guardar_prediccion(tipo=tipo, experimento_id=experimento_id,
                                 exportar=False, **datos)
        print(f"   ✅ {fichero} -> predicción '{tipo}' "
              f"(accuracy {datos['accuracy']:.2f}%)")


def main():
    if len(sys.argv) > 1:
        directorios = [os.path.abspath(d) for d in sys.argv[1:]]
    else:
        directorios = sorted(
            os.path.join(RAIZ, d) for d in os.listdir(RAIZ)
            if os.path.isfile(os.path.join(RAIZ, d, 'model_config.json')))

    if not directorios:
        print("No se encontraron carpetas con model_config.json")
        return

    store = ResultadosStore()
    for d in directorios:
        importar_directorio(store, d)
    store.exportar_excel()
    store.cerrar()


if __name__ == "__main__":
    main()
