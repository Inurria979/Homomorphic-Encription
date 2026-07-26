"""
TELEMETRÍA DE SISTEMA
=====================
Mide el consumo de recursos de una fase de trabajo (predicción plana u
homomórfica) para guardarlo junto a las métricas de ML. Usa psutil.

Uso típico:

    ms = MonitorSistema().iniciar()
    ... trabajo a medir ...
    ms.detener()
    info = ms.resumen()   # dict listo para la BD

Captura, muestreando en un hilo aparte:
  - RAM del proceso (RSS): inicial, media, pico
  - RAM disponible de WSL: mínima alcanzada (cuánto llegó a apretar el sistema)
  - Swap usado: pico
  - Número de hilos del proceso: pico
  - Tiempo de CPU (usuario/sistema) consumido -> permite calcular la
    eficiencia de paralelización (cpu_time / (wall * núcleos))
  - Datos estáticos del entorno: núcleos lógicos/físicos, hilos de PyTorch
"""

import threading
import time

import psutil

GB = 1024 ** 3


def _torch_threads():
    try:
        import torch
        return torch.get_num_threads()
    except Exception:
        return None


class MonitorSistema:

    def __init__(self, intervalo=0.2):
        self.intervalo = intervalo
        self.proc = psutil.Process()

        # --- Datos estáticos del entorno (no cambian durante la ejecución) ---
        self.cpu_logico = psutil.cpu_count(logical=True)
        self.cpu_fisico = psutil.cpu_count(logical=False)
        self.torch_threads = _torch_threads()
        self.ram_total_gb = psutil.virtual_memory().total / GB

        # --- Acumuladores del muestreo ---
        self._parar = threading.Event()
        self._hilo = None
        self._muestras_rss = []
        self.rss_inicial = 0
        self.rss_pico = 0
        self.hilos_pico = 0
        self.ram_disponible_min = None
        self.swap_pico = 0

        # --- Tiempos ---
        self._t_wall_ini = None
        self._t_wall_fin = None
        self._cpu_ini = None
        self._cpu_fin = None

    # ------------------------------------------------------------------

    def iniciar(self):
        self.rss_inicial = self.proc.memory_info().rss
        self.rss_pico = self.rss_inicial
        self._muestras_rss = [self.rss_inicial]
        self.hilos_pico = self.proc.num_threads()
        self.ram_disponible_min = psutil.virtual_memory().available
        self.swap_pico = psutil.swap_memory().used

        self._t_wall_ini = time.time()
        self._cpu_ini = self.proc.cpu_times()

        self._parar.clear()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()
        return self

    def _muestrear(self):
        while not self._parar.wait(self.intervalo):
            try:
                rss = self.proc.memory_info().rss
                self._muestras_rss.append(rss)
                if rss > self.rss_pico:
                    self.rss_pico = rss

                hilos = self.proc.num_threads()
                if hilos > self.hilos_pico:
                    self.hilos_pico = hilos

                disp = psutil.virtual_memory().available
                if disp < self.ram_disponible_min:
                    self.ram_disponible_min = disp

                swap = psutil.swap_memory().used
                if swap > self.swap_pico:
                    self.swap_pico = swap
            except Exception:
                # Un fallo puntual de lectura no debe tumbar la medición
                pass

    def detener(self):
        self._parar.set()
        if self._hilo:
            self._hilo.join(timeout=2)
        self._t_wall_fin = time.time()
        self._cpu_fin = self.proc.cpu_times()
        # Última lectura por si el pico ocurrió justo al final
        try:
            rss = self.proc.memory_info().rss
            self._muestras_rss.append(rss)
            if rss > self.rss_pico:
                self.rss_pico = rss
        except Exception:
            pass

    # ------------------------------------------------------------------

    def resumen(self):
        """Devuelve todas las métricas de sistema en un dict (unidades en el nombre)."""
        wall = (self._t_wall_fin or time.time()) - (self._t_wall_ini or time.time())
        cpu_user = cpu_sys = None
        eficiencia_cpu = None
        if self._cpu_ini and self._cpu_fin:
            cpu_user = self._cpu_fin.user - self._cpu_ini.user
            cpu_sys = self._cpu_fin.system - self._cpu_ini.system
            cpu_total = cpu_user + cpu_sys
            if wall > 0 and self.cpu_logico:
                # 1.0 = usó de media todos los núcleos al 100%
                eficiencia_cpu = cpu_total / (wall * self.cpu_logico)

        rss_medio = (sum(self._muestras_rss) / len(self._muestras_rss)
                     if self._muestras_rss else self.rss_inicial)

        return {
            'cpu_count_logico': self.cpu_logico,
            'cpu_count_fisico': self.cpu_fisico,
            'torch_threads': self.torch_threads,
            'hilos_pico': self.hilos_pico,
            'tiempo_wall_s': round(wall, 4),
            'cpu_time_usuario_s': round(cpu_user, 4) if cpu_user is not None else None,
            'cpu_time_sistema_s': round(cpu_sys, 4) if cpu_sys is not None else None,
            'cpu_time_total_s': (round(cpu_user + cpu_sys, 4)
                                 if cpu_user is not None else None),
            'eficiencia_cpu': round(eficiencia_cpu, 4) if eficiencia_cpu is not None else None,
            'ram_inicial_gb': round(self.rss_inicial / GB, 3),
            'ram_media_gb': round(rss_medio / GB, 3),
            'ram_pico_proceso_gb': round(self.rss_pico / GB, 3),
            'ram_disponible_min_gb': (round(self.ram_disponible_min / GB, 3)
                                      if self.ram_disponible_min is not None else None),
            'swap_pico_gb': round(self.swap_pico / GB, 3),
        }
