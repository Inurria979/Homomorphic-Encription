"""
BATERÍA DE EXPERIMENTOS
=======================
Repite los experimentos históricos

Cada experimento hace: entrenamiento multi-semilla -> predicción plana ->
predicción homomórfica, y guarda todo en resultados/resultados.db y
resultados/resultados.xlsx. Los modelos y gráficas van a experimentos_v2/<nombre>.

BARRIDO CKKS INCLUIDO
---------------------
Este script es el ÚNICO punto de entrada: lanza tanto los entrenamientos como
todas las predicciones homomórficas del estudio.

Las redes marcadas con `barrido=True` reciben una predicción homomórfica por cada
configuración CKKS de la rejilla grado x escala (ver
configuracion_ckks.generar_configuraciones), todas sobre el MISMO modelo ya
entrenado: mismos pesos, mismo test, solo cambia el cifrado. Es lo que sostiene la
comparación entre grados del polinomio, donde se ve que en las redes con capas
ocultas el grado -y no el cifrado- es lo que decide la exactitud.

En esas redes la corrida base entrena y predice en claro pero NO cifra
(--sin-homomorfica): su configuración automática es grado 3 con escala 2^31, que
ya está dentro de la rejilla, y cifrarla ahí duplicaría la corrida. Las redes sin
barrido sí cifran en la corrida base, con esa configuración automática.

Las redes sin capas ocultas no tienen activación polinómica, así que el grado no
interviene y la rejilla se colapsa a las escalas (lo hace generar_configuraciones).
Diabetes queda fuera del barrido: su test de 4000 muestras dispara cada corrida a
horas.

AISLAMIENTO POR SUBPROCESO
---------------------------
Cada experimento se ejecuta como un `python main.py ...` independiente, NO en
el mismo proceso que este orquestador. Esto es así tras un incidente real: con
los 12 experimentos corriendo en un único proceso Python de larga duración, la
RAM que consumían los hilos de TenSEAL no siempre volvía al sistema operativo
entre experimentos (arenas de memoria por hilo en glibc que malloc_trim no
alcanza), y el proceso arrastraba cada vez más memoria "pegada" de un
experimento al siguiente hasta tumbar WSL. Lanzar cada experimento en un
proceso propio garantiza que el sistema operativo recupera el 100% de su RAM
al terminar, pase lo que pase dentro. Si aun así un experimento se queda sin
margen de RAM, main.py aborta ese experimento con un error controlado
(PresupuestoRAMExcedido) en vez de arriesgar la máquina.

Si un experimento falla (o se queda sin RAM), se reintenta una vez con la
mitad de RAM permitida; si vuelve a fallar, se anota el error y se continúa
con el siguiente.

Uso:
    python ejecutar_experimentos.py --listar             # ver la lista
    python ejecutar_experimentos.py                      # ejecutar todos
    python ejecutar_experimentos.py BreastCancer_PCA5    # solo los indicados
    python ejecutar_experimentos.py --rapidos            # sin los de Diabetes (horas)
    python ejecutar_experimentos.py --continuar          # reanuda tras un corte
    python ejecutar_experimentos.py --sin-barrido        # solo la corrida base de cada red
    python ejecutar_experimentos.py --solo-barrido       # solo el barrido CKKS (no reentrena)
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

from configuracion_ckks import generar_configuraciones, num_capas_lineales
from ResultadosStore import ResultadosStore

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_SALIDA = os.path.join(RAIZ, 'experimentos_v2')
FICHERO_REGISTRO = os.path.join(DIR_SALIDA, 'registro_ejecucion.log')
PYTHON = sys.executable  # el intérprete del venv activo (.venv/bin/python)

# ---------------------------------------------------------------------------
# Definición de experimentos.
#   lento=True   -> dataset grande (Diabetes, ~4000 muestras de test): la
#                   predicción homomórfica puede tardar varias horas.
#   barrido=True -> además de la corrida base, se evalúa la rejilla completa de
#                   configuraciones CKKS (grado x escala) sobre el modelo ya
#                   entrenado, sin reentrenar.
# ---------------------------------------------------------------------------
EXPERIMENTOS = [
    # --- Breast Cancer (rápidos, ~114 muestras de test) ---
    dict(nombre='BreastCancer_PCA5', dataset_id=17, model_size=5, hidden=[], nc=2,
         pca=5, rg=0, barrido=True, descripcion='Repite Breast_Cancer: PCA a 5 -> red 5->2'),
    dict(nombre='BreastCancer_PCA5_rg', dataset_id=17, model_size=5, hidden=[], nc=2,
         pca=5, rg=1e-4, barrido=True,
         descripcion='Repite directorio_test_rg, ahora con regularización real'),
    dict(nombre='BreastCancer_PCA10', dataset_id=17, model_size=10, hidden=[], nc=2,
         pca=10, rg=0, barrido=True, descripcion='NUEVO: PCA a 10 -> red 10->2'),
    dict(nombre='BreastCancer_Full30', dataset_id=17, model_size=30, hidden=[16, 8], nc=2,
         pca=0, rg=0, barrido=True,
         descripcion='Repite BC_entero: 30 características -> 30->16->8->2'),

    # --- Credit Approval (~138 muestras de test) ---
    # Sin barrido: red sin capas ocultas, entra como control de otro dominio.
    dict(nombre='Credit_PCA5', dataset_id=27, model_size=5, hidden=[], nc=2,
         pca=5, rg=0, descripcion='Repite CreditApproval: PCA a 5 -> red 5->2'),

    # --- Obesidad (7 clases, ~423 muestras de test) ---
    dict(nombre='Obesity_16-7', dataset_id=544, model_size=16, hidden=[], nc=7,
         pca=0, rg=0, descripcion='Repite Obesity_SinCapasIntermedias: 16->7'),
    dict(nombre='Obesity_16-10-7', dataset_id=544, model_size=16, hidden=[10], nc=7,
         pca=0, rg=0, barrido=True, descripcion='Repite Obesity: 16->10->7'),
    dict(nombre='Obesity_16-10-7_rg', dataset_id=544, model_size=16, hidden=[10], nc=7,
         pca=0, rg=1e-4, barrido=True,
         descripcion='NUEVO: lo que Obesity_rg_16-10-7 pretendía ser (rg real)'),
    dict(nombre='Obesity_12-7_rg', dataset_id=544, model_size=12, hidden=[], nc=7,
         pca=12, rg=1e-4, descripcion='Repite Obesity_rg_12_7 con regularización real'),
    dict(nombre='Obesity_12-5-7_rg', dataset_id=544, model_size=12, hidden=[5], nc=7,
         pca=12, rg=1e-4, barrido=True,
         descripcion='Repite Obesity_rg_12-5-7 con regularización real'),

    # --- Diabetes (LENTOS: hasta 20000 muestras -> test de 4000) ---
    dict(nombre='Diabetes_PCA5', dataset_id=891, model_size=5, hidden=[], nc=2,
         pca=5, rg=0, lento=True, descripcion='Repite Diabetes: PCA a 5 -> red 5->2'),
    dict(nombre='Diabetes_12-5', dataset_id=891, model_size=12, hidden=[5], nc=2,
         pca=12, rg=0, lento=True,
         descripcion='Repite Diabetes_Todo_Dataset: 12->5->2 (cap 20000 muestras)'),
]


def configs_barrido(exp):
    """Configuraciones CKKS del barrido de una red ([] si no lleva barrido)."""
    if not exp.get('barrido'):
        return []
    return generar_configuraciones(num_capas_lineales(exp.get('hidden', [])))


def registrar(mensaje):
    """Escribe en pantalla y en el log de ejecución (con flush para poder
    seguir el progreso desde otra terminal con tail -f)."""
    linea = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mensaje}"
    print(linea, flush=True)
    with open(FICHERO_REGISTRO, 'a', encoding='utf-8') as f:
        f.write(linea + '\n')


def formato_duracion(segundos):
    h, resto = divmod(int(segundos), 3600)
    m, s = divmod(resto, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def base_completada(nombre, cifra):
    """
    True si la corrida base de ese experimento ya llegó al final en una ejecución
    anterior. Lo que hay que buscar en la BD depende de hasta dónde llega esa
    corrida: si no cifra (el cifrado lo aporta su rejilla) termina en la predicción
    plana, y si cifra termina en la homomórfica.
    """
    tipo = 'homomorfica' if cifra else 'plana'
    store = ResultadosStore()
    fila = store.conn.execute(
        """SELECT COUNT(*) FROM predicciones p
           JOIN experimentos e ON e.id = p.experimento_id
           WHERE e.directorio = ? AND p.tipo = ?""",
        (nombre, tipo)).fetchone()
    store.cerrar()
    return fila[0] > 0


def configs_ya_guardadas(nombre):
    """
    Configuraciones CKKS de ese experimento que ya están en la BD, como conjunto
    de (grado, escala_bits, poly_modulus_degree). Permite reanudar un barrido
    interrumpido sin repetir corridas que duran horas.
    """
    store = ResultadosStore()
    filas = store.conn.execute(
        """SELECT p.grado_taylor, p.global_scale_bits, p.poly_modulus_degree
           FROM predicciones p JOIN experimentos e ON e.id = p.experimento_id
           WHERE e.directorio = ? AND p.tipo = 'homomorfica'""",
        (nombre,)).fetchall()
    store.cerrar()
    hechas = set()
    for grado, escala, poly in filas:
        if escala is None or poly is None:
            continue
        hechas.add((str(grado), int(escala), int(poly)))
    return hechas


def clave_config(cfg):
    """La misma clave que devuelve configs_ya_guardadas, para una config nueva."""
    return (str(cfg['grado']), int(cfg['global_scale_bits']),
            int(cfg['poly_modulus_degree']))


def construir_comando(exp, path, batch_size, num_workers, fraccion_ram, semilla_maestra,
                      cifra=True):
    """
    Traduce la definición del experimento a los argumentos de main.py.

    Con `cifra=False` la corrida entrena y predice en claro pero no cifra: es lo
    que se hace en las redes con barrido, cuya configuración automática (grado 3,
    escala 2^31) ya es una de las de la rejilla.
    """
    hidden = exp.get('hidden', [])
    cmd = [
        PYTHON, os.path.join(RAIZ, 'main.py'), path,
        '--train',
        '--model_size', str(exp['model_size']),
        '--dataset', str(exp['dataset_id']),
        '--nc', str(exp.get('nc', 2)),
        '--pca', str(exp.get('pca', 0)),
        '--rg', str(exp.get('rg', 0)),
        '--lr', str(exp.get('lr', 0.001)),
        '--seeds', str(exp.get('n_seeds', 50)),
        '--bz', str(batch_size),
        '--nw', str(num_workers),
        '--ram', str(fraccion_ram),
        '--semilla', str(semilla_maestra),
    ]
    if not cifra:
        cmd.append('--sin-homomorfica')
    if len(hidden) >= 1:
        cmd += ['--h1', str(hidden[0])]
    if len(hidden) >= 2:
        cmd += ['--h2', str(hidden[1])]
    return cmd


def construir_comando_barrido(exp, path, cfg, batch_size, num_workers, fraccion_ram):
    """
    Comando main.py que ejecuta SOLO la predicción homomórfica con una
    configuración CKKS concreta, reutilizando el modelo ya entrenado del
    directorio (no vuelve a entrenar ni repite la predicción plana).
    """
    hidden = exp.get('hidden', [])
    cmd = [
        PYTHON, os.path.join(RAIZ, 'main.py'), path,
        '--solo-homomorfica',
        '--model_size', str(exp['model_size']),
        '--dataset', str(exp['dataset_id']),
        '--nc', str(exp.get('nc', 2)),
        '--pca', str(exp.get('pca', 0)),
        '--rg', str(exp.get('rg', 0)),
        '--degree', str(cfg['grado']),
        '--scale-bits', str(cfg['global_scale_bits']),
        '--poly-mod', str(cfg['poly_modulus_degree']),
        '--coeff-mod', str(cfg['coeff_mod_bit_sizes']).replace(' ', ''),
        '--bz', str(batch_size),
        '--nw', str(num_workers),
        '--ram', str(fraccion_ram),
    ]
    if len(hidden) >= 1:
        cmd += ['--h1', str(hidden[0])]
    if len(hidden) >= 2:
        cmd += ['--h2', str(hidden[1])]
    return cmd


def lanzar_subproceso(cmd):
    """
    Ejecuta main.py como proceso hijo, heredando stdout/stderr (así todo
    fluye al mismo salida_experimentos.log que usa el orquestador). Fija
    MALLOC_ARENA_MAX=2 para que glibc no reparta la memoria de los hilos en
    arenas separadas que luego no se puedan recuperar con malloc_trim.
    Devuelve el código de salida (0 = éxito).
    """
    env = dict(os.environ)
    env['MALLOC_ARENA_MAX'] = '2'
    print(f"$ {' '.join(cmd)}", flush=True)
    proceso = subprocess.run(cmd, env=env)
    return proceso.returncode


def razon_fallo(codigo):
    """Traduce el código de salida de un subproceso a algo legible en el log."""
    return (f"señal {-codigo} (posible OOM-kill del kernel)" if codigo < 0
            else f"código de salida {codigo}")


def ejecutar_con_reintento(construir, etiqueta, fraccion_ram):
    """
    Lanza en un subproceso el comando que devuelve `construir(fraccion_ram)`. Si
    falla (código de salida != 0, ya sea por una excepción propia o porque el
    kernel mató el proceso por falta de memoria), reintenta UNA vez con la mitad
    de RAM permitida antes de darlo por fallido.
    """
    codigo = lanzar_subproceso(construir(fraccion_ram))
    if codigo == 0:
        return 'OK'

    registrar(f"⚠️  {etiqueta} falló ({razon_fallo(codigo)}). Reintentando con "
              f"la mitad de RAM permitida ({fraccion_ram / 2:.0%})...")

    codigo2 = lanzar_subproceso(construir(fraccion_ram / 2))
    if codigo2 == 0:
        return 'OK (tras reintento con menos RAM)'

    return f"ERROR: falló también el reintento ({razon_fallo(codigo2)})"


def ejecutar_experimento(exp, batch_size, num_workers, semilla_maestra, fraccion_ram,
                         cifra=True):
    """Corrida base: entrenamiento -> predicción plana -> predicción homomórfica."""
    path = os.path.join(DIR_SALIDA, exp['nombre'])
    return ejecutar_con_reintento(
        lambda ram: construir_comando(exp, path, batch_size, num_workers, ram,
                                      semilla_maestra, cifra),
        exp['nombre'], fraccion_ram)


def ejecutar_config_barrido(exp, cfg, batch_size, num_workers, fraccion_ram):
    """Una configuración CKKS del barrido sobre el modelo ya entrenado."""
    path = os.path.join(DIR_SALIDA, exp['nombre'])
    return ejecutar_con_reintento(
        lambda ram: construir_comando_barrido(exp, path, cfg, batch_size, num_workers, ram),
        f"{exp['nombre']} · {cfg['etiqueta']}", fraccion_ram)


def main():
    parser = argparse.ArgumentParser(description="Ejecuta la batería de experimentos completa.")
    parser.add_argument("nombres", nargs='*',
                        help="Nombres de experimentos concretos (por defecto: todos)")
    parser.add_argument("--listar", action='store_true', help="Lista los experimentos y sale")
    parser.add_argument("--rapidos", action='store_true',
                        help="Excluye los experimentos lentos (Diabetes)")
    parser.add_argument("--bz", type=int, default=10,
                        help="Batch size de la predicción homomórfica paralela (default: 10)")
    parser.add_argument("--nw", type=int, default=os.cpu_count(),
                        help=f"Hilos de la predicción homomórfica (default: {os.cpu_count()})")
    parser.add_argument("--semilla", type=int, default=42,
                        help="Semilla maestra base para reproducibilidad (default: 42)")
    parser.add_argument("--ram", type=float, default=0.75,
                        help="Fracción de la RAM de WSL usable por cada experimento "
                             "(default: 0.75, tope absoluto 12 GB)")
    parser.add_argument("--continuar", action='store_true',
                        help="Salta los experimentos y las configuraciones CKKS que ya están "
                             "guardadas en la BD (para reanudar tras un corte)")
    parser.add_argument("--sin-barrido", action='store_true', dest='sin_barrido',
                        help="Solo la corrida base de cada red, sin la rejilla grado x escala")
    parser.add_argument("--solo-barrido", action='store_true', dest='solo_barrido',
                        help="Solo el barrido CKKS sobre los modelos ya entrenados "
                             "(no entrena ni repite la predicción plana)")
    args = parser.parse_args()

    if args.sin_barrido and args.solo_barrido:
        print("❌ --sin-barrido y --solo-barrido se excluyen entre sí.")
        sys.exit(1)

    if args.listar:
        print(f"{'Nombre':<25} {'Dataset':<8} {'Arquitectura':<18} {'PCA':<4} {'rg':<8} Descripción")
        print("-" * 110)
        total_configs = 0
        for exp in EXPERIMENTOS:
            arq = ' -> '.join(map(str, [exp['model_size']] + exp.get('hidden', []) + [exp['nc']]))
            lento = ' (LENTO)' if exp.get('lento') else ''
            print(f"{exp['nombre']:<25} {exp['dataset_id']:<8} {arq:<18} {exp.get('pca', 0):<4} "
                  f"{exp.get('rg', 0):<8} {exp['descripcion']}{lento}")
            configs = configs_barrido(exp)
            total_configs += len(configs)
            for cfg in configs:
                print(f"{'':<25}   ↳ {cfg['etiqueta']:<16} grado {cfg['grado']}, "
                      f"escala 2^{cfg['global_scale_bits']}, poly {cfg['poly_modulus_degree']}, "
                      f"coeff {cfg['coeff_mod_bit_sizes']}")
        print("-" * 110)
        con_barrido = sum(1 for e in EXPERIMENTOS if e.get('barrido'))
        sin_barrido = len(EXPERIMENTOS) - con_barrido
        print(f"{len(EXPERIMENTOS)} corridas base ({con_barrido} sin cifrar, el cifrado lo "
              f"aporta su rejilla; {sin_barrido} con su configuración automática) "
              f"+ {total_configs} configuraciones CKKS")
        print(f"= {len(EXPERIMENTOS)} predicciones planas y {sin_barrido + total_configs} "
              f"homomórficas, sin repetir ninguna configuración.")
        return

    seleccion = EXPERIMENTOS
    if args.nombres:
        conocidos = {e['nombre'] for e in EXPERIMENTOS}
        desconocidos = set(args.nombres) - conocidos
        if desconocidos:
            print(f"❌ Experimentos desconocidos: {sorted(desconocidos)}")
            print(f"   Disponibles: {sorted(conocidos)}")
            sys.exit(1)
        seleccion = [e for e in EXPERIMENTOS if e['nombre'] in args.nombres]
    if args.rapidos:
        seleccion = [e for e in seleccion if not e.get('lento')]

    os.makedirs(DIR_SALIDA, exist_ok=True)

    n_configs = 0 if args.sin_barrido else sum(len(configs_barrido(e)) for e in seleccion)
    n_base = 0 if args.solo_barrido else len(seleccion)
    registrar(f"===== INICIO DE LA BATERÍA: {n_base} corridas base + {n_configs} "
              f"configuraciones CKKS (bz={args.bz}, nw={args.nw}, "
              f"semilla base={args.semilla}, ram={args.ram:.0%}, "
              f"aislamiento por subproceso) =====")

    total_tareas = n_base + n_configs
    resumen = []
    tarea = 0
    interrumpido = False

    def ejecutar_tarea(etiqueta, descripcion, funcion):
        """Ejecuta una tarea midiendo el tiempo y anotándola en el resumen."""
        nonlocal tarea, interrumpido
        tarea += 1
        registrar(f"▶️  [{tarea}/{total_tareas}] {etiqueta}: {descripcion}")
        t0 = time.time()
        try:
            estado = funcion()
        except KeyboardInterrupt:
            registrar(f"⛔ Interrumpido por el usuario durante {etiqueta}")
            resumen.append((etiqueta, 'INTERRUMPIDO', time.time() - t0))
            interrumpido = True
            return False
        duracion = time.time() - t0
        resumen.append((etiqueta, estado, duracion))
        registrar(f"{'✅' if estado.startswith('OK') else '❌'} {etiqueta} terminado en "
                  f"{formato_duracion(duracion)} [{estado}]")
        return estado.startswith('OK')

    for i, exp in enumerate(seleccion):
        nombre = exp['nombre']
        modelo = os.path.join(DIR_SALIDA, nombre, 'models', 'best_model.pth')
        # La base solo cifra si esta red no va a recibir su rejilla: así ninguna
        # configuración se ejecuta dos veces.
        cifra_base = args.sin_barrido or not exp.get('barrido')

        # --- 1. Corrida base: entrenamiento + predicción plana (+ homomórfica) ---
        if not args.solo_barrido:
            if args.continuar and base_completada(nombre, cifra_base):
                registrar(f"⏭️  {nombre}: ya completado, se salta (--continuar)")
                resumen.append((nombre, 'SALTADO', 0))
                tarea += 1
            elif not ejecutar_tarea(
                    nombre,
                    exp['descripcion'] + ('' if cifra_base else ' [sin cifrar: lo hace su rejilla]'),
                    lambda exp=exp, i=i, cifra=cifra_base: ejecutar_experimento(
                        exp, args.bz, args.nw, semilla_maestra=args.semilla + i,
                        fraccion_ram=args.ram, cifra=cifra)):
                if interrumpido:
                    break
                # Sin modelo entrenado el barrido no puede correr: se salta entero
                registrar(f"⚠️  {nombre}: falló la corrida base, se omite su barrido CKKS")
                tarea += len(configs_barrido(exp))
                continue

        # --- 2. Barrido CKKS sobre el modelo ya entrenado ---
        configs = [] if args.sin_barrido else configs_barrido(exp)
        if configs and not os.path.isfile(modelo):
            registrar(f"⚠️  {nombre}: sin modelo entrenado en {modelo}, se omite su barrido")
            tarea += len(configs)
            continue

        hechas = configs_ya_guardadas(nombre) if (args.continuar and configs) else set()
        for cfg in configs:
            if clave_config(cfg) in hechas:
                registrar(f"⏭️  {nombre} · {cfg['etiqueta']}: ya en la BD, se salta (--continuar)")
                resumen.append((f"{nombre} · {cfg['etiqueta']}", 'SALTADO', 0))
                tarea += 1
                continue
            ejecutar_tarea(f"{nombre} · {cfg['etiqueta']}",
                           f"grado {cfg['grado']}, escala 2^{cfg['global_scale_bits']}, "
                           f"poly {cfg['poly_modulus_degree']}",
                           lambda exp=exp, cfg=cfg: ejecutar_config_barrido(
                               exp, cfg, args.bz, args.nw, args.ram))
            if interrumpido:
                break
        if interrumpido:
            break

    registrar("===== RESUMEN FINAL =====")
    for nombre, estado, duracion in resumen:
        registrar(f"   {nombre:<40} {formato_duracion(duracion):>10}  {estado}")
    total = sum(d for _, _, d in resumen)
    fallos = sum(1 for _, e, _ in resumen if not (e.startswith('OK') or e == 'SALTADO'))
    registrar(f"===== FIN: {len(resumen)} ejecutados, {fallos} con error, "
              f"tiempo total {formato_duracion(total)} =====")
    registrar(f"Resultados en resultados/resultados.db y resultados/resultados.xlsx")


if __name__ == "__main__":
    main()
