"""
BARRIDO DE PARÁMETROS CKKS
==========================
Hasta ahora solo se ajustaba la parte de ML; el cifrado usaba parámetros fijos
(polinomio de grado 3, escala 31, cadena de 9 primos, poly 16384) para todas las
redes. Este script barre la parte de CIFRADO: para cada red entrena UNA vez y
luego evalúa la predicción homomórfica con varias configuraciones CKKS
(grado 3 y 5, distintas escalas y cadenas de módulos adaptadas a la profundidad,
ver configuracion_ckks.py).

Cada red produce: 1 entrenamiento + 1 predicción plana + N predicciones
homomórficas (una por configuración CKKS). Todas quedan en la BD con sus
propios parámetros, así que la web/Excel pueden comparar el coste del cifrado
en función de esos parámetros.

Igual que ejecutar_experimentos.py, cada corrida se lanza como un SUBPROCESO
independiente y bloqueante (nunca en paralelo) con MALLOC_ARENA_MAX=2, para que
el SO recupere la RAM entre ejecuciones y no se desborde WSL.

Uso (desde la raíz del repo, con el venv):

    .venv/bin/python barrido_ckks.py --listar          # ver el plan
    .venv/bin/python barrido_ckks.py --rapido          # prueba corta (breast cancer)
    .venv/bin/python barrido_ckks.py                   # barrido completo de breast cancer
    .venv/bin/python barrido_ckks.py CKKS_BC_Full30    # solo una(s) red(es)
    .venv/bin/python barrido_ckks.py --continuar       # saltar lo ya hecho
"""

import argparse
import os
import subprocess
import sys
import time

from configuracion_ckks import (generar_configuraciones, num_capas_lineales,
                                GRADOS_POR_DEFECTO, ESCALAS_POR_DEFECTO,
                                POLY_MODULUS_MAX)

RAIZ = os.path.dirname(os.path.abspath(__file__))
# El intérprete del entorno activo (el mismo que lanzó este script), igual que
# en ejecutar_experimentos.py. Antes estaba cableado a .venv/bin/python, lo que
# rompía todos los subprocesos si el entorno tenía otro nombre o otra ruta.
VENV_PYTHON = sys.executable
MAIN = os.path.join(RAIZ, "main.py")
BASE_DIR = "experimentos_v2"

# Redes de Breast Cancer sobre las que se barre el cifrado. Se incluyen redes
# CON capas ocultas (6->4->2, 30->16->8->2) porque solo ahí influye el grado del
# polinomio (sin activaciones, el grado es irrelevante).
REDES = [
    dict(nombre="CKKS_BC_PCA5",   dataset_id=17, model_size=5,  hidden=[],     nc=2, pca=5,  rg=0,
         desc="5->2 (sin capas ocultas): solo importa la escala"),
    dict(nombre="CKKS_BC_PCA10",  dataset_id=17, model_size=10, hidden=[],     nc=2, pca=10, rg=0,
         desc="10->2 (sin capas ocultas): solo importa la escala"),
    dict(nombre="CKKS_BC_6-4-2",  dataset_id=17, model_size=6,  hidden=[4],    nc=2, pca=6,  rg=0,
         desc="6->4->2 (1 capa oculta): grado 3 vs 5, rápido"),
    dict(nombre="CKKS_BC_Full30", dataset_id=17, model_size=30, hidden=[16, 8], nc=2, pca=0, rg=0,
         desc="30->16->8->2 (2 capas ocultas): barrido completo grado x escala"),
]

# Conjunto reducido y escalas cortas para la prueba rápida de validación.
RAPIDO_NOMBRES = ["CKKS_BC_PCA5", "CKKS_BC_6-4-2"]
RAPIDO_ESCALAS = (25, 40)


def configs_de_red(red, escalas):
    """Configuraciones CKKS a probar para una red, según su profundidad."""
    L = num_capas_lineales(red["hidden"])
    return generar_configuraciones(L, escalas=escalas)


def cmd_entrenar(red, semilla):
    """Comando main.py que entrena + predice plana, SIN cifrado."""
    directorio = f"{BASE_DIR}/{red['nombre']}"
    cmd = [VENV_PYTHON, MAIN, directorio, "--train", "--sin-homomorfica",
           "--dataset", str(red["dataset_id"]), "--model_size", str(red["model_size"]),
           "--nc", str(red["nc"]), "--pca", str(red["pca"]),
           "--rg", str(red["rg"]), "--semilla", str(semilla)]
    if len(red["hidden"]) >= 1:
        cmd += ["--h1", str(red["hidden"][0])]
    if len(red["hidden"]) >= 2:
        cmd += ["--h2", str(red["hidden"][1])]
    return cmd


def cmd_config(red, cfg, bz, nw, ram):
    """Comando main.py que ejecuta SOLO la homomórfica con una config CKKS."""
    directorio = f"{BASE_DIR}/{red['nombre']}"
    cmd = [VENV_PYTHON, MAIN, directorio, "--solo-homomorfica",
           "--dataset", str(red["dataset_id"]), "--model_size", str(red["model_size"]),
           "--nc", str(red["nc"]), "--pca", str(red["pca"]), "--rg", str(red["rg"]),
           "--degree", str(cfg["grado"]),
           "--scale-bits", str(cfg["global_scale_bits"]),
           "--poly-mod", str(cfg["poly_modulus_degree"]),
           "--coeff-mod", str(cfg["coeff_mod_bit_sizes"]).replace(" ", ""),
           "--bz", str(bz), "--nw", str(nw), "--ram", str(ram)]
    if len(red["hidden"]) >= 1:
        cmd += ["--h1", str(red["hidden"][0])]
    if len(red["hidden"]) >= 2:
        cmd += ["--h2", str(red["hidden"][1])]
    return cmd


def lanzar(cmd, descripcion):
    """Lanza un subproceso bloqueante y aislado (RAM controlada)."""
    env = dict(os.environ)
    env["MALLOC_ARENA_MAX"] = "2"   # que glibc no acapare arenas por hilo
    print(f"\n{'='*70}\n▶ {descripcion}\n  {' '.join(cmd)}\n{'='*70}", flush=True)
    ini = time.time()
    r = subprocess.run(cmd, env=env)
    dur = time.time() - ini
    ok = r.returncode == 0
    print(f"{'✅' if ok else '❌'} {descripcion}  ({dur:.0f}s, código {r.returncode})", flush=True)
    return ok


def plan(redes, escalas):
    """Devuelve la lista de (red, cfg_o_None) a ejecutar; None = entrenamiento."""
    tareas = []
    for red in redes:
        tareas.append((red, None))  # entrenamiento
        for cfg in configs_de_red(red, escalas):
            tareas.append((red, cfg))
    return tareas


def main():
    ap = argparse.ArgumentParser(description="Barrido de parámetros CKKS sobre breast cancer.")
    ap.add_argument("nombres", nargs="*", help="Redes concretas (por defecto todas)")
    ap.add_argument("--rapido", action="store_true", help="Prueba corta: pocas redes y escalas")
    ap.add_argument("--listar", action="store_true", help="Muestra el plan y sale")
    ap.add_argument("--continuar", action="store_true",
                    help="Salta las corridas cuyo entrenamiento ya existe (reanudar)")
    ap.add_argument("--semilla", type=int, default=42, help="Semilla maestra (default 42)")
    ap.add_argument("--bz", type=int, default=10, help="Batch size de la homomórfica")
    ap.add_argument("--nw", type=int, default=os.cpu_count(), help="Hilos de la homomórfica")
    ap.add_argument("--ram", type=float, default=0.75, help="Fracción de RAM WSL usable")
    args = ap.parse_args()

    redes = REDES
    escalas = None  # None => escalas por defecto de configuracion_ckks
    if args.rapido:
        redes = [r for r in REDES if r["nombre"] in RAPIDO_NOMBRES]
        escalas = RAPIDO_ESCALAS
    if args.nombres:
        redes = [r for r in REDES if r["nombre"] in args.nombres]
        if not redes:
            print("Ninguna red coincide. Disponibles:", [r["nombre"] for r in REDES])
            return

    # --- Mostrar el plan ---
    total_configs = n_descartadas = 0
    print("PLAN DEL BARRIDO CKKS")
    print("=" * 70)
    grados_pedidos = GRADOS_POR_DEFECTO
    escalas_pedidas = escalas or ESCALAS_POR_DEFECTO
    for red in redes:
        cfgs = configs_de_red(red, escalas)
        total_configs += len(cfgs)
        L = num_capas_lineales(red["hidden"])
        # Combinaciones que se piden vs las viables: la diferencia son las que no
        # caben en RAM (poly_modulus_degree > POLY_MODULUS_MAX) o no son seguras.
        pedidas = (len(grados_pedidos) if L > 1 else 1) * len(escalas_pedidas)
        descartadas = pedidas - len(cfgs)
        n_descartadas += descartadas
        print(f"\n{red['nombre']}  ({red['desc']}; {L} capas lineales)")
        print(f"  entrenamiento (plana) + {len(cfgs)} configuraciones homomórficas:")
        for c in cfgs:
            print(f"    - {c['etiqueta']:16s} coeff={c['coeff_mod_bit_sizes']} "
                  f"poly={c['poly_modulus_degree']} ~{c['ram_contexto_gb']}GB ctx")
        if descartadas:
            print(f"    ⚠️  {descartadas} configuración(es) descartada(s): requieren "
                  f"poly_modulus_degree > {POLY_MODULUS_MAX} (no caben en la RAM disponible)")
    print(f"\nTOTAL: {len(redes)} entrenamientos + {total_configs} predicciones homomórficas "
          f"= {len(redes) + total_configs} subprocesos.")
    if n_descartadas:
        print(f"       ({n_descartadas} configuraciones descartadas por RAM; ver "
              f"POLY_MODULUS_MAX en configuracion_ckks.py)")

    if args.listar:
        return

    # --- Ejecutar ---
    tareas = plan(redes, escalas)
    n_ok = n_fallo = 0
    t0 = time.time()
    for red, cfg in tareas:
        directorio = f"{BASE_DIR}/{red['nombre']}"
        if cfg is None:
            # Entrenamiento de la red.
            if args.continuar and os.path.isfile(f"{directorio}/models/best_model.pth"):
                print(f"↷ {red['nombre']}: modelo ya entrenado, se salta el entrenamiento.")
                continue
            ok = lanzar(cmd_entrenar(red, args.semilla), f"ENTRENAR {red['nombre']}")
        else:
            # Una configuración CKKS (solo homomórfica).
            if not os.path.isfile(f"{directorio}/models/best_model.pth"):
                print(f"⚠️  {red['nombre']}: sin modelo entrenado; se omite {cfg['etiqueta']}.")
                n_fallo += 1
                continue
            ok = lanzar(cmd_config(red, cfg, args.bz, args.nw, args.ram),
                        f"{red['nombre']} · {cfg['etiqueta']}")
        n_ok += ok
        n_fallo += (not ok)

    print(f"\n{'='*70}\nBARRIDO TERMINADO en {(time.time()-t0)/60:.1f} min · "
          f"{n_ok} correctas, {n_fallo} con fallo/omitidas.\n{'='*70}")
    return 0 if n_fallo == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
