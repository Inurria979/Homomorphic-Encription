"""
BATERÍA DE EXPERIMENTOS
=======================
Repite los experimentos históricos (con los bugs ya corregidos: la regularización
se aplica de verdad, el early stopping restaura el mejor estado, y las semillas
hacen reproducibles tanto el split como la inicialización de pesos) y añade
algunos nuevos.

Cada experimento hace: entrenamiento multi-semilla -> predicción plana ->
predicción homomórfica, y guarda todo en resultados/resultados.db y
resultados/resultados.xlsx. Los modelos y gráficas van a experimentos_v2/<nombre>.

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
    python ejecutar_experimentos.py --continuar           # reanuda tras un corte
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

from ResultadosStore import ResultadosStore

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_SALIDA = os.path.join(RAIZ, 'experimentos_v2')
FICHERO_REGISTRO = os.path.join(DIR_SALIDA, 'registro_ejecucion.log')
PYTHON = sys.executable  # el intérprete del venv activo (.venv/bin/python)

# ---------------------------------------------------------------------------
# Definición de experimentos.
#   lento=True -> dataset grande (Diabetes, ~4000 muestras de test): la
#   predicción homomórfica puede tardar varias horas por experimento.
# ---------------------------------------------------------------------------
EXPERIMENTOS = [
    # --- Breast Cancer (rápidos, ~114 muestras de test) ---
    dict(nombre='BreastCancer_PCA5', dataset_id=17, model_size=5, hidden=[], nc=2,
         pca=5, rg=0, descripcion='Repite Breast_Cancer: PCA a 5 -> red 5->2'),
    dict(nombre='BreastCancer_PCA5_rg', dataset_id=17, model_size=5, hidden=[], nc=2,
         pca=5, rg=1e-4, descripcion='Repite directorio_test_rg, ahora con regularización real'),
    dict(nombre='BreastCancer_PCA10', dataset_id=17, model_size=10, hidden=[], nc=2,
         pca=10, rg=0, descripcion='NUEVO: PCA a 10 -> red 10->2'),
    dict(nombre='BreastCancer_Full30', dataset_id=17, model_size=30, hidden=[16, 8], nc=2,
         pca=0, rg=0, descripcion='Repite BC_entero: 30 características -> 30->16->8->2'),

    # --- Credit Approval (~138 muestras de test) ---
    dict(nombre='Credit_PCA5', dataset_id=27, model_size=5, hidden=[], nc=2,
         pca=5, rg=0, descripcion='Repite CreditApproval: PCA a 5 -> red 5->2'),

    # --- Obesidad (7 clases, ~423 muestras de test) ---
    dict(nombre='Obesity_16-7', dataset_id=544, model_size=16, hidden=[], nc=7,
         pca=0, rg=0, descripcion='Repite Obesity_SinCapasIntermedias: 16->7'),
    dict(nombre='Obesity_16-10-7', dataset_id=544, model_size=16, hidden=[10], nc=7,
         pca=0, rg=0, descripcion='Repite Obesity: 16->10->7'),
    dict(nombre='Obesity_16-10-7_rg', dataset_id=544, model_size=16, hidden=[10], nc=7,
         pca=0, rg=1e-4, descripcion='NUEVO: lo que Obesity_rg_16-10-7 pretendía ser (rg real)'),
    dict(nombre='Obesity_12-7_rg', dataset_id=544, model_size=12, hidden=[], nc=7,
         pca=12, rg=1e-4, descripcion='Repite Obesity_rg_12_7 con regularización real'),
    dict(nombre='Obesity_12-5-7_rg', dataset_id=544, model_size=12, hidden=[5], nc=7,
         pca=12, rg=1e-4, descripcion='Repite Obesity_rg_12-5-7 con regularización real'),

    # --- Diabetes (LENTOS: hasta 20000 muestras -> test de 4000) ---
    dict(nombre='Diabetes_PCA5', dataset_id=891, model_size=5, hidden=[], nc=2,
         pca=5, rg=0, lento=True, descripcion='Repite Diabetes: PCA a 5 -> red 5->2'),
    dict(nombre='Diabetes_12-5', dataset_id=891, model_size=12, hidden=[5], nc=2,
         pca=12, rg=0, lento=True, descripcion='Repite Diabetes_Todo_Dataset: 12->5->2 (cap 20000 muestras)'),
]


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


def experimento_completado(nombre):
    """True si ese experimento ya tiene una predicción homomórfica en la BD
    (es decir, llegó hasta el final en una ejecución anterior)."""
    store = ResultadosStore()
    fila = store.conn.execute(
        """SELECT COUNT(*) FROM predicciones p
           JOIN experimentos e ON e.id = p.experimento_id
           WHERE e.directorio = ? AND p.tipo = 'homomorfica'""",
        (nombre,)).fetchone()
    store.cerrar()
    return fila[0] > 0


def construir_comando(exp, path, batch_size, num_workers, fraccion_ram, semilla_maestra):
    """Traduce la definición del experimento a los argumentos de main.py."""
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
        '--seeds', str(exp.get('n_seeds', 5)),
        '--bz', str(batch_size),
        '--nw', str(num_workers),
        '--ram', str(fraccion_ram),
        '--semilla', str(semilla_maestra),
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


def ejecutar_experimento(exp, batch_size, num_workers, semilla_maestra, fraccion_ram):
    """
    Lanza el experimento en un subproceso. Si falla (código de salida != 0,
    ya sea por una excepción propia o porque el kernel mató el proceso por
    falta de memoria), reintenta UNA vez con la mitad de RAM permitida antes
    de darlo por fallido.
    """
    path = os.path.join(DIR_SALIDA, exp['nombre'])
    cmd = construir_comando(exp, path, batch_size, num_workers, fraccion_ram, semilla_maestra)

    codigo = lanzar_subproceso(cmd)
    if codigo == 0:
        return 'OK'

    razon = (f"señal {-codigo} (posible OOM-kill del kernel)" if codigo < 0
             else f"código de salida {codigo}")
    registrar(f"⚠️  {exp['nombre']} falló ({razon}). Reintentando con "
              f"la mitad de RAM permitida ({fraccion_ram / 2:.0%})...")

    cmd_reintento = construir_comando(exp, path, batch_size, num_workers,
                                      fraccion_ram / 2, semilla_maestra)
    codigo2 = lanzar_subproceso(cmd_reintento)
    if codigo2 == 0:
        return 'OK (tras reintento con menos RAM)'

    razon2 = (f"señal {-codigo2} (posible OOM-kill del kernel)" if codigo2 < 0
             else f"código de salida {codigo2}")
    return f"ERROR: falló también el reintento ({razon2})"


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
                        help="Salta los experimentos que ya tienen predicción homomórfica "
                             "guardada en la BD (para reanudar tras un corte)")
    args = parser.parse_args()

    if args.listar:
        print(f"{'Nombre':<25} {'Dataset':<8} {'Arquitectura':<18} {'PCA':<4} {'rg':<8} Descripción")
        print("-" * 110)
        for exp in EXPERIMENTOS:
            arq = ' -> '.join(map(str, [exp['model_size']] + exp.get('hidden', []) + [exp['nc']]))
            lento = ' (LENTO)' if exp.get('lento') else ''
            print(f"{exp['nombre']:<25} {exp['dataset_id']:<8} {arq:<18} {exp.get('pca', 0):<4} "
                  f"{exp.get('rg', 0):<8} {exp['descripcion']}{lento}")
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

    registrar(f"===== INICIO DE LA BATERÍA: {len(seleccion)} experimentos "
              f"(bz={args.bz}, nw={args.nw}, semilla base={args.semilla}, "
              f"ram={args.ram:.0%}, aislamiento por subproceso) =====")

    resumen = []
    for i, exp in enumerate(seleccion):
        if args.continuar and experimento_completado(exp['nombre']):
            registrar(f"⏭️  [{i + 1}/{len(seleccion)}] {exp['nombre']}: ya completado, se salta (--continuar)")
            resumen.append((exp['nombre'], 'SALTADO', 0))
            continue
        registrar(f"▶️  [{i + 1}/{len(seleccion)}] {exp['nombre']}: {exp['descripcion']}")
        t0 = time.time()
        try:
            estado = ejecutar_experimento(exp, args.bz, args.nw,
                                          semilla_maestra=args.semilla + i,
                                          fraccion_ram=args.ram)
        except KeyboardInterrupt:
            registrar(f"⛔ Interrumpido por el usuario durante {exp['nombre']}")
            resumen.append((exp['nombre'], 'INTERRUMPIDO', time.time() - t0))
            break
        duracion = time.time() - t0
        resumen.append((exp['nombre'], estado, duracion))
        registrar(f"{'✅' if estado.startswith('OK') else '❌'} {exp['nombre']} terminado en "
                  f"{formato_duracion(duracion)} [{estado}]")

    registrar("===== RESUMEN FINAL =====")
    for nombre, estado, duracion in resumen:
        registrar(f"   {nombre:<25} {formato_duracion(duracion):>10}  {estado}")
    total = sum(d for _, _, d in resumen)
    fallos = sum(1 for _, e, _ in resumen if not (e.startswith('OK') or e == 'SALTADO'))
    registrar(f"===== FIN: {len(resumen)} ejecutados, {fallos} con error, "
              f"tiempo total {formato_duracion(total)} =====")
    registrar(f"Resultados en resultados/resultados.db y resultados/resultados.xlsx")


if __name__ == "__main__":
    main()
