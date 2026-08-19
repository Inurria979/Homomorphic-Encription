"""
CONTROL DE MEMORIA PARA LA PREDICCIÓN HOMOMÓRFICA
=================================================
La predicción cifrada en paralelo puede desbordar la RAM de WSL (cada hilo
mantiene en memoria tensores CKKS de cientos de MB) y tumbar la máquina.

Este módulo vigila la RSS del proceso leyendo /proc (sin dependencias) y
aplica un límite: una fracción de la RAM total de WSL (75% por defecto) con
un tope absoluto de 12 GB. Con ello:

  - se calibra cuánta RAM consume un lote real y se reduce el número de hilos
    para que el conjunto quepa bajo el límite,
  - cada hilo espera antes de cifrar si en ese momento no hay margen (hasta
    un máximo de 6 horas, avisando en el log cada 5 minutos), y solo si pasado
    ese tiempo la RAM sigue sin bajar se aborta el experimento en curso,
  - se registra el pico de RAM para guardarlo en la BD/Excel.
"""

import ctypes
import gc
import os
import threading
import time

GB = 1024 ** 3
MB = 1024 ** 2

_PAGINA = os.sysconf('SC_PAGE_SIZE')

# Reserva mínima que siempre dejamos libre al sistema, además del límite propio
RESERVA_SISTEMA = 1 * GB


class PresupuestoRAMExcedido(RuntimeError):
    """
    Se lanza cuando, pese a esperar hasta el máximo permitido (por defecto
    6 horas) y forzar la liberación de memoria, el proceso sigue sin margen.
    A propósito NO hay una vía de "continuar de todos modos": la primera
    versión de este módulo tenía ese escape (para evitar deadlocks) y fue
    precisamente lo que dejó que la RAM subiera sin freno hasta tumbar WSL.
    Es preferible abortar el experimento en curso (se recupera con
    --continuar) a arriesgar la máquina entera. Pero antes de llegar a ese
    punto se espera un tiempo generoso: en la inmensa mayoría de los casos
    el margen se libera solo (otro hilo termina, gc/malloc_trim hacen
    efecto) mucho antes del límite.
    """
    pass


def _formato_horas(segundos):
    segundos = int(segundos)
    h, resto = divmod(segundos, 3600)
    m, s = divmod(resto, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _leer_meminfo(campo):
    """Lee un campo de /proc/meminfo y lo devuelve en bytes."""
    with open('/proc/meminfo') as f:
        for linea in f:
            if linea.startswith(campo + ':'):
                return int(linea.split()[1]) * 1024
    raise RuntimeError(f"Campo {campo} no encontrado en /proc/meminfo")


def mem_total_bytes():
    """RAM total visible para WSL."""
    return _leer_meminfo('MemTotal')


def mem_disponible_bytes():
    """RAM aún disponible en todo WSL (cuenta lo que usan otros procesos)."""
    return _leer_meminfo('MemAvailable')


def rss_bytes():
    """Memoria residente actual de ESTE proceso (todos sus hilos)."""
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * _PAGINA


def liberar_memoria():
    """
    Recolecta basura y fuerza a glibc a devolver al sistema la memoria ya
    liberada. Sin malloc_trim, los buffers de TenSEAL liberados se quedan
    retenidos en el arena de malloc y la RSS (y la RAM de WSL) no baja.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class MonitorMemoria:
    """
    Vigila la RSS del proceso en un hilo de muestreo y ofrece:
      - margen_bytes(): cuánto se puede crecer sin pasar el límite
      - esperar_margen(): bloquea a un hilo hasta que haya hueco
      - resumen(): pico y límites para guardar en la BD
    """

    def __init__(self, fraccion_limite=0.75, tope_absoluto_gb=12.0, intervalo=0.2):
        self.ram_total = mem_total_bytes()
        self.limite = min(int(self.ram_total * fraccion_limite),
                          int(tope_absoluto_gb * GB))
        self.intervalo = intervalo
        self.pico = rss_bytes()
        self._pico_ventana = self.pico
        self._parar = threading.Event()
        self._hilo = None
        # Coordina el aviso periódico entre los hilos que puedan estar
        # esperando margen a la vez, para no repetir la misma línea en el
        # log una vez por hilo cada vez que toca avisar
        self._lock_aviso = threading.Lock()
        self._ultimo_aviso = 0.0

    def iniciar(self):
        self.pico = self._pico_ventana = rss_bytes()
        self._parar.clear()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()
        return self

    def _muestrear(self):
        while not self._parar.wait(self.intervalo):
            r = rss_bytes()
            if r > self.pico:
                self.pico = r
            if r > self._pico_ventana:
                self._pico_ventana = r

    def detener(self):
        self._parar.set()
        if self._hilo:
            self._hilo.join(timeout=2)
        # Última lectura por si el muestreo se perdió el pico final
        r = rss_bytes()
        if r > self.pico:
            self.pico = r

    # ------------------------------------------------------------------
    # Ventana de medición (para calibrar el coste de un lote)
    # ------------------------------------------------------------------

    def reiniciar_ventana(self):
        self._pico_ventana = rss_bytes()

    def pico_ventana(self):
        return max(self._pico_ventana, rss_bytes())

    # ------------------------------------------------------------------
    # Margen y espera
    # ------------------------------------------------------------------

    def margen_bytes(self):
        """
        Cuánto puede crecer aún el proceso: lo que quede hasta nuestro límite,
        pero nunca más de lo que le queda libre a WSL (menos una reserva),
        por si otros procesos del PC también están consumiendo.
        """
        propio = self.limite - rss_bytes()
        sistema = mem_disponible_bytes() - RESERVA_SISTEMA
        return min(propio, sistema)

    def esperar_margen(self, bytes_necesarios, timeout=6 * 3600, intervalo_aviso=300):
        """
        Bloquea hasta que haya margen para bytes_necesarios, esperando como
        máximo `timeout` segundos (6 horas por defecto).

        Mientras espera, recuerda en el log cada `intervalo_aviso` segundos
        (5 min por defecto) que sigue bloqueado -con cuánto lleva esperando y
        cuánto le queda-, para que quede explícito que no se ha colgado en
        silencio si alguien revisa los logs mucho después. Si varios hilos
        están esperando a la vez, solo uno imprime el aviso en cada ventana.

        Si se agotan las 6 horas sin conseguir margen, se hace un último
        intento de liberar memoria agresivamente (gc + malloc_trim) y, si
        con eso tampoco hay margen, se lanza PresupuestoRAMExcedido con un
        mensaje bien visible: NUNCA se continúa por encima del límite,
        porque eso fue justo lo que tumbó WSL la última vez.
        """
        inicio = time.time()
        fin = inicio + timeout
        while self.margen_bytes() < bytes_necesarios:
            ahora = time.time()
            if ahora > fin:
                break
            with self._lock_aviso:
                if ahora - self._ultimo_aviso >= intervalo_aviso:
                    print(f"🧠 Esperando margen de RAM desde hace "
                          f"{_formato_horas(ahora - inicio)} "
                          f"(máximo {_formato_horas(timeout)}): "
                          f"RSS {rss_bytes() / GB:.2f} GB / "
                          f"límite {self.limite / GB:.2f} GB", flush=True)
                    self._ultimo_aviso = ahora
            time.sleep(self.intervalo)
        else:
            return True

        # Se agotó el tiempo máximo: último intento antes de rendirse
        print(f"🧠 Se agotaron las {_formato_horas(timeout)} de espera de RAM: "
              "forzando una última liberación de memoria...", flush=True)
        liberar_memoria()
        if self.margen_bytes() >= bytes_necesarios:
            return True

        mensaje = (
            f"⛔ TIMEOUT DE RAM TRAS {_formato_horas(timeout)} DE ESPERA: "
            f"RSS {rss_bytes() / GB:.2f} GB sigue sin margen para "
            f"{bytes_necesarios / MB:.0f} MB (límite {self.limite / GB:.2f} GB). "
            "Se aborta este experimento -la memoria no ha bajado pese a "
            "esperar y forzar su liberación- para no arriesgar la RAM del "
            "sistema. Revisa si hay otro proceso pesado corriendo a la vez.")
        print(mensaje, flush=True)
        raise PresupuestoRAMExcedido(mensaje)

    def resumen(self):
        """Datos para la BD/Excel (en GB)."""
        return {
            'ram_total_wsl_gb': round(self.ram_total / GB, 2),
            'ram_limite_gb': round(self.limite / GB, 2),
            'ram_pico_gb': round(self.pico / GB, 2),
        }
