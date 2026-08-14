import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from PrediccionHomomorfica import PrediccionHomomorfica
from MonitorMemoria import MB, GB, rss_bytes, liberar_memoria, PresupuestoRAMExcedido

# Colchón sobre la memoria medida en la calibración (es una estimación).
#
# Bajado de 1.3 a 1.1 midiendo sobre la red 30->16->8->2 a N=32768, que es el caso
# límite del proyecto: la sonda mide ~3640 MB por muestra, frente a un margen que
# en esta máquina oscila entre 3900 y 4500 MB según lo que haya abierto. O sea que
# UNA muestra ocupa ya el ~85 % del margen disponible, y cualquier colchón por
# encima de ~1.15 la deja fuera:
#     1.3 -> 4740 MB (no cabe nunca)   1.2 -> 4360 MB (no cabe con margen 4300)
#     1.1 -> 4000 MB (cabe con 4300, justo con 3900)
# El consumo REAL de esa corrida es 4,24 GB de contexto + 3,65 de la muestra = 7,9
# de los 11,96 GB de la máquina: lo que sobraba era colchón sobre una estimación,
# no memoria. NO se toca RESERVA_SISTEMA (1 GB), que es la otra protección y la
# que evita dejar sin memoria al resto del sistema.
FACTOR_SEGURIDAD = 1.1
# Un solo lote nunca debería comerse más que esta fracción del margen libre
FRACCION_MAX_LOTE = 0.4
# Fracción del margen que puede ocupar el conjunto de hilos simultáneos
FRACCION_MAX_HILOS = 0.85

class PrediccionHomomorficaParalela(PrediccionHomomorfica):

    def __init__(self, data_dir, model_path, config_path, degree=5, verbose=True):
        super().__init__(data_dir, model_path, config_path, degree, verbose)
        # Se rellenan durante predict_parallel para poder guardarlos en la BD
        self.info_recursos = {}
        self._monitor = None
        self._mem_lote = None


    def worker_inference(self, X_batch):
        """
        Función que ejecutará cada hilo de forma independiente.
        """
        try:
            # Freno de memoria: si el proceso está cerca del límite, este hilo
            # espera antes de crear nuevos tensores cifrados. Si ni esperando
            # ni forzando la liberación hay margen, esperar_margen lanza
            # PresupuestoRAMExcedido (ver más abajo: NO se atrapa aquí).
            if self._monitor is not None and self._mem_lote:
                self._monitor.esperar_margen(self._mem_lote)

            # 1. Encriptar (Pico de RAM inicial del hilo)
            enc_x = self.encrypt_data_batch(X_batch)

            # 2. Predecir (Cálculo pesado)
            enc_output = self.predict_on_encrypted_batch(enc_x)

            # 3. Desencriptar (Liberación de peso)
            plain_output = self.decrypt_predictions(enc_output)

            # Limpieza explícita dentro del hilo
            del enc_x
            del enc_output

            return plain_output
        except PresupuestoRAMExcedido:
            # No se atrapa como un error cualquiera: debe propagarse para que
            # predict_parallel aborte la predicción en vez de seguir con
            # menos muestras de las debidas (ver nota en MonitorMemoria).
            raise
        except Exception as e:
            print(f"❌ Error en un hilo: {e}")
            return None


    def _calibrar(self, X_test, batch_size, max_workers, monitor):
        """
        Ejecuta UNA muestra en solitario midiendo su pico de RAM y, con esa
        medida, decide el tamaño de lote y el número de hilos que caben bajo
        el límite de memoria. La red 30->16->8->2, por ejemplo, consume varios
        GB por lote de 10: ahí no basta con bajar hilos, hay que bajar el lote.

        Devuelve (resultado_sonda, batch_efectivo, workers_efectivos).
        """
        base = rss_bytes()
        monitor.reiniciar_ventana()
        res_sonda = self.worker_inference(X_test[0:1])
        mem_muestra = max(monitor.pico_ventana() - base, 8 * MB)
        liberar_memoria()

        margen = monitor.margen_bytes()
        coste_muestra = mem_muestra * FACTOR_SEGURIDAD

        # 1. Lote: que uno solo no pase de FRACCION_MAX_LOTE del margen
        batch_efectivo = max(1, min(batch_size,
                                    int(margen * FRACCION_MAX_LOTE // coste_muestra)))
        # 2. Hilos: que todos los lotes simultáneos quepan en FRACCION_MAX_HILOS
        coste_lote = batch_efectivo * coste_muestra
        workers_efectivos = max(1, min(max_workers,
                                       int(margen * FRACCION_MAX_HILOS // coste_lote)))

        self._mem_lote = coste_lote

        if self.verbose:
            print(f"🧠 Calibración: ~{mem_muestra / MB:.0f} MB por muestra | "
                  f"margen: {margen / GB:.1f} GB | "
                  f"lote {batch_size}→{batch_efectivo} | "
                  f"hilos {max_workers}→{workers_efectivos}", flush=True)
        if batch_efectivo * workers_efectivos < batch_size:
            print(f"⚠️  Modelo pesado en memoria: la predicción irá más lenta "
                  f"para no desbordar la RAM de WSL", flush=True)

        return res_sonda, batch_efectivo, workers_efectivos

    def _hilos_para_esta_ola(self, workers_calibrados, monitor):
        """
        Recalcula, ANTES de cada oleada, cuántos hilos caben en el margen
        actual (no solo en el calibrado al principio). Si la RAM no bajó lo
        esperado tras la oleada anterior, aquí se reduce la concurrencia en
        vez de confiar ciegamente en la calibración inicial.
        """
        if not self._mem_lote:
            return workers_calibrados
        margen = monitor.margen_bytes()
        posibles = max(1, int(margen * FRACCION_MAX_HILOS // self._mem_lote))
        return max(1, min(workers_calibrados, posibles))

    def predict_parallel(self, batch_size=2, max_workers=2, X_test=None, monitor=None):
        """
        Lanza la inferencia en paralelo con control de memoria.

        monitor: MonitorMemoria opcional. Si se pasa, se calibra el coste de
        RAM con una muestra sonda y se procesa en OLEADAS: antes de cada
        oleada se recalcula cuántos hilos caben en el margen actual (por si
        la memoria no bajó lo esperado), y cada hilo espera margen antes de
        cifrar. Si pese a esperar y forzar la liberación de memoria sigue sin
        haber margen, se aborta con PresupuestoRAMExcedido en vez de superar
        el límite (ver MonitorMemoria: ya no existe un "continuar de todos
        modos", que fue lo que dejó crecer la RAM sin freno la última vez).
        X_test: opcional, subconjunto a predecir; por defecto todo self.X_test
        """
        if X_test is None:
            X_test = self.X_test
        if self.verbose:
            print(
                f"🧬 Iniciando experimento: {max_workers} hilos | Batch: {batch_size}")

        start_time = time.time()

        resultados_finales = []
        batch_efectivo = batch_size
        workers_efectivos = max_workers
        self._monitor = monitor
        self._mem_lote = None

        restantes = X_test
        if monitor is not None and len(X_test) > 1:
            res_sonda, batch_efectivo, workers_efectivos = self._calibrar(
                X_test, batch_size, max_workers, monitor)
            if res_sonda is not None:
                resultados_finales.append(res_sonda)
            restantes = X_test[1:]

        # Dividimos lo pendiente en trozos del tamaño de lote efectivo
        chunks = [restantes[i:i + batch_efectivo]
                for i in range(0, len(restantes), batch_efectivo)]

        procesados = 0
        pos = 0
        try:
            while pos < len(chunks):
                hilos_ola = (self._hilos_para_esta_ola(workers_efectivos, monitor)
                            if monitor is not None else workers_efectivos)
                tanda = chunks[pos:pos + hilos_ola]

                with ThreadPoolExecutor(max_workers=len(tanda)) as executor:
                    futures = [executor.submit(self.worker_inference, c) for c in tanda]
                    for future in futures:
                        res = future.result()
                        if res is not None:
                            resultados_finales.append(res)

                pos += len(tanda)
                procesados += len(tanda)

                # Devolvemos al sistema la memoria de la oleada terminada: sin
                # esto glibc la retiene y el margen medido nunca se recupera
                liberar_memoria()
                if (procesados % 5 == 0 or pos >= len(chunks)) and self.verbose:
                    print(f"📊 Lote {pos}/{len(chunks)} procesado "
                          f"(RSS {rss_bytes() / GB:.2f} GB)...")
        except PresupuestoRAMExcedido as e:
            print(f"🛑 Predicción abortada para no arriesgar la RAM del sistema: {e}",
                  flush=True)
            raise

        total_time = time.time() - start_time
        if self.verbose:
            print(f"⏱️ Tiempo total paralelo: {total_time:.2f} segundos")

        self.info_recursos = {
            'num_workers_efectivos': workers_efectivos,
            'batch_size_efectivo': batch_efectivo,
        }
        if monitor is not None:
            self.info_recursos.update(monitor.resumen())

        return np.concatenate(resultados_finales)
