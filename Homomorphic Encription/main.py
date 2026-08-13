"""
PIPELINE PRINCIPAL
==================
Punto de entrada de un experimento: entrena (opcional) con submuestreo aleatorio
repetido de Monte Carlo y evalúa el modelo resultante dos veces -en claro y sobre
datos cifrados homomórficamente (TenSEAL/CKKS)-, guardando métricas de ML y de
sistema en la base de datos y el Excel.

Uso: python main.py <carpeta> [--train] [--dataset ID] [--pca N] ...
     (ver los argumentos en main() más abajo)
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from DataProcessor import DataProcessor
from Trainer import ConfigurableNN
from PrediccionPlana import PrediccionPlana
from PrediccionHomomorfica import PrediccionHomomorfica
from PrediccionHomomorficaParalela import PrediccionHomomorficaParalela
from configuracion_ckks import ESCALA_OPTIMA
from scripts import train_with_multiple_seeds, save_final_model_and_plots
from ResultadosStore import ResultadosStore
from MonitorMemoria import MonitorMemoria
from MonitorSistema import MonitorSistema
from Metricas import distribucion_clases, ratio_balance


# Tamaño del conjunto del que se extraen las semillas del muestreo Monte Carlo.
# Solo acota el valor de las semillas; el número de entrenamientos lo fija --seeds.
RANGO_SEMILLAS = 10_000




def end_to_end(model_size=5, learning_rate=0.001, n_seeds=50, hidden_layers=[], num_classes=2, path="", dataset_id=17, pca_features=0,batch_size=0, num_workers=0, rg=0, train=False, dataset=None, fraccion_ram=0.75, solo_homomorfica=False, sin_homomorfica=False, ckks=None):
    """Construye el modelo según model_size y lanza el pipeline completo."""
    # Arquitecturas predefinidas para los dos casos más habituales; el resto
    # se construye a medida con hidden_layers/num_classes
    if model_size == 5:
        model = ConfigurableNN(input_size=5, hidden_layers=[], num_classes=2)
    elif model_size == 30:
        # Breast Cancer con las 30 características originales
        model = ConfigurableNN(input_size=model_size, hidden_layers=[16, 8], num_classes=2)
    else:
        model = ConfigurableNN(input_size=model_size, hidden_layers=hidden_layers, num_classes=num_classes)

    # Se puede pasar un DataProcessor ya cargado para no volver a descargar
    # el dataset de UCI (útil al encadenar varios experimentos)
    if dataset is None:
        dataset = DataProcessor(dataset_id)
    store = ResultadosStore()
    end_to_end_mc(model, learning_rate, n_seeds,path, dataset, pca_features=pca_features ,batch_size=batch_size, num_workers=num_workers, rg=rg, train=train, store=store, fraccion_ram=fraccion_ram, solo_homomorfica=solo_homomorfica, sin_homomorfica=sin_homomorfica, ckks=ckks)


def end_to_end_mc(model, learning_rate, n_seeds, path, dataset, batch_size, num_workers, rg=0, pca_features=0, train = False, store=None, fraccion_ram=0.75, solo_homomorfica=False, sin_homomorfica=False, ckks=None):
    """
    Submuestreo aleatorio repetido (Monte Carlo) + evaluación.

    Con cada semilla se remezclan train y val (el test se mantiene intacto) y se
    entrena un modelo. El muestreo aporta dos cosas: la semilla de mejor accuracy
    de validación, que fija la inicialización del modelo definitivo, y la mediana
    de las épocas óptimas, que es el número de épocas del reentrenamiento con
    train+val. Después se evalúa en claro y homomórficamente, guardando
    experimento y predicciones en la BD/Excel.
    """
    # Semillas SIN repetición: se extraen sin reemplazo. Con reemplazo se
    # entrenarían modelos idénticos más de una vez, lo que desperdicia cómputo y
    # sesga los estadísticos del muestreo (val_accuracy_media y
    # val_accuracy_desviacion contarían el mismo modelo varias veces, de modo que
    # la desviación saldría artificialmente baja).
    # El rango se ensancha a n_seeds si hiciera falta, para no pedir nunca una
    # muestra mayor que la población.
    seeds = random.sample(range(max(RANGO_SEMILLAS, n_seeds)), n_seeds)

    # Información del dataset (tamaños y balance de clases) para guardar en la BD.
    # El split es fijo: 20% test, 10% val, 70% train sobre el total.
    n_total = len(dataset.X)
    n_test = int(round(n_total * 0.2))
    n_val = int(round(n_total * 0.1))
    n_train = n_total - n_test - n_val
    info_dataset = dict(
        n_total=n_total, n_train=n_train, n_val=n_val, n_test=n_test,
        n_features_original=int(dataset.X.shape[1]) if hasattr(dataset.X, 'shape') else None,
        distribucion_clases=distribucion_clases(dataset.y),
        ratio_balance=ratio_balance(dataset.y),
    )

    experimento_id = None
    if train:
        best_model, best_seed, best_test_dataset, results, epocas_finales = train_with_multiple_seeds(model=model, processor=dataset,output_dir=path, learning_rate=learning_rate, seeds=seeds,
                                test_size=0.2, val_size=0.1, pca_features=pca_features, rg=rg)

        save_final_model_and_plots(
            model=best_model,
            seed=best_seed,
            results=results,
            test_dataset=best_test_dataset,
            output_dir=path

        )

        if store:
            config = best_model.get_config()
            experimento_id = store.guardar_experimento(
                directorio=path,
                dataset_id=dataset.dataset_id,
                dataset_nombre=dataset.dataset_nombre,
                input_size=config['input_size'],
                hidden_layers=config['hidden_layers'],
                num_clases=config['num_classes'],
                arquitectura=config['architecture'],
                pca_features=pca_features,
                learning_rate=learning_rate,
                regularizacion=rg,
                mejor_semilla=best_seed,
                mejor_val_accuracy=max(r['val_accuracy'] for r in results),
                semillas_resultados=results,
                epocas_reentrenamiento=epocas_finales,
                **info_dataset
            )
    elif store:
        # Sin entrenamiento: recuperamos (o reconstruimos desde los json del
        # directorio) el experimento para poder enlazar las predicciones
        experimento_id = store.obtener_o_registrar_experimento(
            directorio=path,
            dataset_id=dataset.dataset_id,
            dataset_nombre=dataset.dataset_nombre,
            learning_rate=learning_rate,
            regularizacion=rg,
            pca_features=pca_features
        )

    # La plana se salta en el barrido CKKS (--solo-homomorfica): ya está guardada
    # del entrenamiento y no depende del cifrado.
    if not solo_homomorfica:
        predict_plain(path=path, store=store, experimento_id=experimento_id)

    # El entrenamiento del barrido (--sin-homomorfica) no cifra: cada config CKKS
    # se lanza después en su propio subproceso.
    if not sin_homomorfica:
        #el tamaño de test val y train está fijo por eso pongo directamnte 0.2
        predict_encrypt(path=path, dataset_len=len(dataset.X) * 0.2,
                        batch_size=batch_size, num_workers=num_workers,
                        fraccion_ram=fraccion_ram, store=store,
                        experimento_id=experimento_id, **(ckks or {}))


def predict_plain(path, store=None, experimento_id=None):
    model_path = f'{path}/models/best_model.pth'
    config_path = f'{path}/model_config.json'
    save_path = f'{path}/pred'

    print("Empezamos la predicción plana")
    pp = PrediccionPlana(data_dir=path, model_path=model_path, config_path=config_path)

    # Medimos recursos de sistema durante la inferencia
    ms = MonitorSistema().iniciar()
    predictions , probabilities, inference_time = pp.predict_plain()
    ms.detener()

    n_samples = len(pp.y_test)
    # Con las probabilidades del modelo obtenemos también ROC-AUC y log-loss
    metrics = pp.calculate_metrics(predictions, y_proba=probabilities)
    pp.save_results_txt(metrics, inference_time, n_samples, save_path)
    pp.plot_confusion_matrix(metrics['conf_matrix'], save_path,
                             titulo='Matriz de Confusión - Predicción Plana')

    if store:
        store.guardar_prediccion(
            tipo='plana',
            experimento_id=experimento_id,
            metrics=metrics,
            sistema=ms.resumen(),
            n_muestras=n_samples,
            tiempo_inferencia=inference_time
        )

def predict_encrypt(path, dataset_len, degree=3, poly_modulus_degree=None,
                    scale_bits=None, coeff_mod_bit_sizes=None,
                    batch_size=10, num_workers=12, fraccion_ram=0.75,
                    store=None, experimento_id=None):
    """
    Predicción homomórfica con parámetros CKKS configurables.

    degree               : grado del polinomio que aproxima la ReLU (ajustado por
                           mínimos cuadrados al rango real medido; 2, 3, 5, 7...).
    poly_modulus_degree,
    coeff_mod_bit_sizes,
    scale_bits           : parámetros CKKS. Si no se dan, se AUTO-derivan de la
                           profundidad de la red (ver configuracion_ckks): redes
                           llanas -> cadenas cortas y poly pequeño; redes
                           profundas / grado 5 -> cadenas largas y poly mayor.

    Cada llamada guarda UNA predicción homomórfica; el barrido llama varias
    veces con configuraciones distintas (todas quedan en la BD, cada una con
    sus propios parámetros).
    """
    n_samples = dataset_len

    # 0 significa "auto": evita ThreadPoolExecutor(max_workers=0) y trocear en 0
    if not batch_size or batch_size <= 0:
        batch_size = 10
    if not num_workers or num_workers <= 0:
        num_workers = os.cpu_count()

    model_path = f'{path}/models/best_model.pth'
    config_path = f'{path}/model_config.json'

    # Vigilancia de RAM: límite = fraccion_ram de la RAM de WSL, tope 12 GB.
    # Evita que la predicción paralela desborde la memoria y tumbe WSL.
    monitor = MonitorMemoria(fraccion_limite=fraccion_ram, tope_absoluto_gb=12.0).iniciar()
    # Telemetría de sistema (CPU, hilos, RAM media) de todo el proceso cifrado
    ms = MonitorSistema().iniciar()
    print(f"🧠 RAM WSL: {monitor.ram_total / 1024**3:.1f} GB | "
          f"límite de uso: {monitor.limite / 1024**3:.1f} GB")

    # Test pequeño: todo de una vez (serial). Grande: por lotes con control de RAM.
    paralela = n_samples >= 100
    if paralela:
        ph = PrediccionHomomorficaParalela(data_dir=path, model_path=model_path,
                                           config_path=config_path, degree=degree, verbose=True)
    else:
        ph = PrediccionHomomorfica(data_dir=path, model_path=model_path,
                                   config_path=config_path, degree=degree, verbose=True)

    # --- Configuración CKKS ---
    # Si el barrido pasa los parámetros, se respetan; si no, los calcula la propia
    # PrediccionHomomorfica a partir de sus capas lineales y su grado, llevando la
    # escala al máximo que cabe en el presupuesto de bits.
    explicita = None not in (poly_modulus_degree, coeff_mod_bit_sizes)
    ini = time.time()
    if explicita:
        ph.setup_encryption_context(
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            global_scale=2 ** (scale_bits if scale_bits else ESCALA_OPTIMA))
    else:
        ph.setup_encryption_context()
    context_time = time.time() - ini

    # La configuración que de verdad se ha usado la deja la clase en ph.ckks
    poly_modulus_degree = ph.ckks['poly_modulus_degree']
    coeff_mod_bit_sizes = ph.ckks['coeff_mod_bit_sizes']
    global_scale_bits = ph.ckks['global_scale_bits']

    print(f"🔧 CKKS: grado poly {ph.degree} | poly_modulus_degree {poly_modulus_degree} | "
          f"escala 2^{global_scale_bits} | coeff_mod {coeff_mod_bit_sizes} "
          f"| {sum(coeff_mod_bit_sizes)} bits -> seguridad "
          f"{ph.ckks.get('seguridad_efectiva')} | "
          f"{'explícita' if explicita else 'calculada por la red'}")

    if paralela:
        ini = time.time()
        final_pred = ph.predict_parallel(batch_size=batch_size, max_workers=num_workers, monitor=monitor)
        inference_time_enc = time.time() - ini
        print(f'Tiempo en todo el proceso homomorfico {inference_time_enc} con '
              f'batch_size de {batch_size} y max workers de {num_workers}')
        params_cifrado = dict(
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            global_scale_bits=global_scale_bits,
            grado_taylor=ph.degree,
            seguridad_efectiva=ph.ckks.get('seguridad_efectiva'),
            rango_activacion=ph.rango_activacion,
            poly_coeffs=ph.taylor_coeffs,
            batch_size=batch_size,
            num_workers=num_workers,
            num_workers_efectivos=ph.info_recursos.get('num_workers_efectivos'),
            batch_size_efectivo=ph.info_recursos.get('batch_size_efectivo'),
            tiempo_contexto=context_time,
        )
    else:
        ini = time.time()
        encrypted_samples = ph.encrypt_data()
        enc_time = time.time() - ini

        ini = time.time()
        encrypted_pred = ph.predict_on_encrypted_batch(enc_x=encrypted_samples)
        pred_time = time.time() - ini

        ini = time.time()
        final_pred = ph.decrypt_predictions(encrypted_tensor=encrypted_pred)
        decrypt_time = time.time() - ini

        inference_time_enc = context_time + enc_time + pred_time + decrypt_time
        print(f'Tiempo en todo el proceso homomorfico {inference_time_enc}')
        params_cifrado = dict(
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            global_scale_bits=global_scale_bits,
            grado_taylor=ph.degree,
            seguridad_efectiva=ph.ckks.get('seguridad_efectiva'),
            rango_activacion=ph.rango_activacion,
            poly_coeffs=ph.taylor_coeffs,
            tiempo_contexto=context_time,
            tiempo_cifrado=enc_time,
            tiempo_prediccion=pred_time,
            tiempo_descifrado=decrypt_time,
            num_workers_efectivos=1,
        )
    save_path_enc = f'{path}/pred_hom'
    # Convertir y_test a lista de numpy (si es un tensor)
    y_true = ph.y_test.cpu().numpy() if hasattr(ph.y_test, 'cpu') else np.array(ph.y_test)

    # Asegurar que final_pred sea un array de numpy
    y_pred = np.array(final_pred)

    #Calcular aciertos
    correctos = (y_true == y_pred)
    accuracy = np.mean(correctos) * 100

    # Mostrar índices donde falló
    indices_error = np.where(y_true != y_pred)[0]

    n_samples = len(y_true)

    # Cerramos las dos vigilancias y recogemos picos/telemetría para la BD
    monitor.detener()
    ms.detener()
    info_ram = monitor.resumen()
    info_sistema = ms.resumen()
    print(f"🧠 Pico de RAM del proceso: {info_ram['ram_pico_gb']} GB "
          f"(límite {info_ram['ram_limite_gb']} GB) | "
          f"hilos pico {info_sistema['hilos_pico']} | "
          f"eficiencia CPU {info_sistema['eficiencia_cpu']}")

    metrics_enc = ph.calculate_metrics(y_pred=y_pred)
    ph.save_results_txt(inference_time=inference_time_enc, n_samples=n_samples, save_path=save_path_enc, metrics=metrics_enc,
                        titulo="RESULTADOS DE PREDICCIÓN HOMOMÓRFICA (CON CIFRADO)",
                        descripcion="Predicción sobre datos cifrados (TenSEAL - CKKS)")
    ph.plot_confusion_matrix(metrics_enc['conf_matrix'], save_path_enc,
                             titulo='Matriz de Confusión - Predicción Homomórfica')

    if store:
        store.guardar_prediccion(
            tipo='homomorfica',
            experimento_id=experimento_id,
            metrics=metrics_enc,
            sistema=info_sistema,
            n_muestras=n_samples,
            tiempo_inferencia=inference_time_enc,
            **info_ram,
            **params_cifrado
        )



def main():

    parser = argparse.ArgumentParser(
        description="Pipeline de Entrenamiento y Evaluación Plana y Homomórfica (TenSEAL)."
    )

    # Argumento OBLIGATORIO: Path
    # Si no se pone, el script se detiene y avisa al usuario.
    parser.add_argument("path", type=str, 
                        help="Ruta (carpeta) donde se guardarán los modelos, logs y resultados.")

    # Argumentos OPCIONALES con valores por defecto
    parser.add_argument("--model_size", type=int, default=5,
                        help="Componentes PCA / Tamaño de entrada (default: 5)")

    parser.add_argument("--lr", type=float, default=0.001, dest="learning_rate",
                        help="Learning rate (default: 0.001)")
    
    parser.add_argument("--seeds", type=int, default=50, dest="n_seeds",
                        help="Número de semillas del muestreo Monte Carlo (default: 50)")
    
    parser.add_argument("--dataset", type=int, default=17, dest="dataset_id",
                        help="ID del dataset UCI: 17=Breast Cancer, 891=Diabetes, "
                             "544=Obesidad, 27=Credit Approval (default: 17)")

    parser.add_argument("--h1", type=int, default = 0, dest="h1",
                        help="Tamaño de la capa intermedia 1")
    parser.add_argument("--h2", type=int, default = 0, dest="h2", 
                        help="Tamaño de la capa intermedia 2")
    parser.add_argument("--nc", type=int, default=2, dest="nc",
                        help="Número de clases y tamaño de la capa fina")
    parser.add_argument("--bz", type=int, default=0, dest="bz",
                        help="Batch Size para predicción paralela encriptada")
    parser.add_argument("--nw", type=int, default=0, dest="nw",
                        help="Número de hilos en predicción paralela encriptada")
    parser.add_argument("--rg", type=float, default=0, dest="rg",
                        help="Coeficiente de regularización" )
    parser.add_argument("--pca", type=int,default=0 ,dest= "pca",
                        help="Número de características a reducir. Tiene que ser igual que el tamaño del model (primera capa)")
    parser.add_argument("--ram", type=float, default=0.75, dest="ram",
                        help="Fracción de la RAM de WSL usable por la predicción homomórfica (default: 0.75, tope absoluto 12 GB)")
    parser.add_argument("--semilla", type=int, default=None, dest="semilla",
                        help="Semilla maestra: fija random/torch antes de generar las semillas "
                             "del Monte Carlo, para que la ejecución sea reproducible")
    # OJO: antes era type=bool, que convertía cualquier texto no vacío a True
    # ("--train False" activaba el entrenamiento). Ahora es un flag: se pone
    # "--train" a secas para entrenar y se omite para solo predecir.
    parser.add_argument("--train", action='store_true', dest="train",
                        help="Activar entrenamiento si es primera vez, por defecto sin entrenar")

    # --- Barrido de parámetros CKKS (cifrado) ---
    parser.add_argument("--degree", type=int, default=7, dest="degree",
                        help="Grado del polinomio que aproxima la ReLU en cifrado, ajustado "
                             "por mínimos cuadrados al rango real. Por defecto 7: consume los "
                             "mismos niveles multiplicativos que el 5, ceil(log2(d+1)), y "
                             "aproxima mejor (2, 3, 5, 7...)")
    parser.add_argument("--scale-bits", type=int, default=None, dest="scale_bits",
                        help="Bits de la escala global CKKS (2^scale). Si se omite, se auto-deriva")
    parser.add_argument("--poly-mod", type=int, default=None, dest="poly_mod",
                        help="poly_modulus_degree CKKS. Si se omite, se elige el menor seguro")
    parser.add_argument("--coeff-mod", type=str, default=None, dest="coeff_mod",
                        help="coeff_mod_bit_sizes como lista JSON, p.ej. '[60,31,31,60]'. Si se omite, se auto-deriva")
    parser.add_argument("--solo-homomorfica", action='store_true', dest="solo_homomorfica",
                        help="Ejecuta SOLO la predicción homomórfica (para el barrido CKKS sobre un modelo ya entrenado)")
    parser.add_argument("--sin-homomorfica", action='store_true', dest="sin_homomorfica",
                        help="Ejecuta entrenamiento + plana pero NO cifrado (para preparar el barrido)")
    args = parser.parse_args()

    # --solo-homomorfica implica no reentrenar (usa el modelo ya guardado).
    if args.solo_homomorfica:
        args.train = False

    ckks = {'degree': args.degree, 'scale_bits': args.scale_bits,
            'poly_modulus_degree': args.poly_mod}
    if args.coeff_mod:
        ckks['coeff_mod_bit_sizes'] = json.loads(args.coeff_mod)

    if args.semilla is not None:
        random.seed(args.semilla)
        torch.manual_seed(args.semilla)

    # Forzar a PyTorch a usar todos los hilos
    num_cores = os.cpu_count() # Debería devolver 12
    torch.set_num_threads(num_cores)
    try:
        torch.set_num_interop_threads(num_cores)
    except RuntimeError:
        # Solo puede fijarse una vez y antes de cualquier operación paralela
        pass

    print(f"[*] PyTorch configurado para usar {num_cores} hilos en CPU.")

    hidden_layers = []
    if args.h1 != 0:
        hidden_layers.append(args.h1)

    if args.h2 != 0:
        hidden_layers.append(args.h2)

    
    end_to_end(
        model_size=args.model_size,
        learning_rate=args.learning_rate,
        n_seeds=args.n_seeds,
        path=args.path,
        dataset_id=args.dataset_id,
        num_classes=args.nc,
        hidden_layers=hidden_layers,
        num_workers=args.nw,
        batch_size=args.bz,
        pca_features=args.pca,
        rg=args.rg,
        train=args.train,
        fraccion_ram=args.ram,
        solo_homomorfica=args.solo_homomorfica,
        sin_homomorfica=args.sin_homomorfica,
        ckks=ckks,
    )


if __name__ == "__main__":
    main()

        

