"""
GENERADOR DE CONFIGURACIONES CKKS
=================================
Hasta ahora el cifrado usaba parámetros FIJOS para todas las redes (polinomio de
grado 3, escala 31 bits, cadena de 9 primos, poly_modulus_degree 16384). Este
módulo genera configuraciones CKKS **adaptadas a la profundidad de la red**:

  - Redes llanas (sin capas ocultas) y grado 3 -> cadena de módulos corta y
    poly_modulus_degree pequeño  => más rápido y menos RAM.
  - Redes profundas y/o grado 5  -> cadena más larga y poly_modulus_degree
    mayor (lo exige la profundidad multiplicativa y la seguridad).

Reglas estándar de CKKS/TenSEAL (SEAL) que se aplican:

1. La **profundidad multiplicativa** disponible = nº de primos intermedios de
   `coeff_mod_bit_sizes`. Cada capa lineal (`.mm`) gasta 1 nivel; cada activación
   polinómica (`.polyval` de grado d) gasta ceil(log2 d) niveles (circuito mínimo).
2. La **escala global** se fija ~ al tamaño de los primos intermedios para que el
   reescalado sea estable. Los primos especiales (primero y último) son los
   mayores (60 bits).
3. El **total de bits** del módulo de coeficientes está acotado por seguridad
   (128 bits): se elige el MENOR `poly_modulus_degree` que admita ese total.

Las constantes están arriba y son fáciles de ajustar si la documentación de
TenSEAL sugiere otros valores.
"""

import math

# --- Constantes ajustables -------------------------------------------------

# Primos intermedios de más allá de los estrictamente necesarios (colchón de
# seguridad frente al gasto real de niveles). La config fija que ya funcionaba
# usaba 2 de margen; 1 es más ajustado (se valida empíricamente).
MARGEN_NIVELES = 1

# Bits de los primos "especiales" (primero y último de la cadena): los mayores.
PRIMO_ESPECIAL = 60

# Límite del total de bits del módulo de coeficientes para 128 bits de seguridad
# (tabla estándar de SEAL / HomomorphicEncryption.org).
MAX_BITS_SEGURIDAD = {
    4096: 109,
    8192: 218,
    16384: 438,
    32768: 881,
}
POLY_MODULUS_CANDIDATOS = [8192, 16384, 32768]

# LÍMITE POR HARDWARE (no por seguridad). `poly_modulus_degree` mayor => cada
# texto cifrado y, sobre todo, las CLAVES DE GALOIS ocupan el doble de RAM, y esa
# memoria es FIJA (no se reduce bajando el batch). Medido en esta máquina:
#     N=8192  (4 primos):  0.08 GB de contexto+claves
#     N=16384 (10 primos): 1.24 GB
#     N=16384 (12 primos): 1.74 GB
#     N=32768 (10 primos): 2.58 GB   <- 2.1x que N=16384 con los MISMOS primos
# Con N=32768 la predicción paralela de la red 30->16->8->2 pedía ~10-13 GB y
# superaba el límite de RAM de WSL (11.7 GB visibles, límite de uso 8.76 GB):
# las 3 configuraciones de escala 40 abortaron por `PresupuestoRAMExcedido` tras
# 6 h de espera cada una (ver docs/incidente_ram_ckks_N32768.md y docs/logs/).
# Con este tope, esas configuraciones se descartan ANTES de lanzarlas.
#
# ACTUALIZACIÓN: la parte que hacía inviable N=32768 eran las CLAVES DE GALOIS,
# que ya no se generan (la inferencia usa CKKSTensor con mm y polyval, que no
# rotan y por tanto no las necesitan). Medido al quitarlas: el pico de RAM de la
# red 30->16->8->2 baja de 4.09 a 2.68 GB y el contexto se crea en 1 s en vez de
# 13 s. El tope se mantiene en 16384 como valor por defecto prudente; se puede
# levantar por parámetro (poly_max) cuando la red lo justifique.
POLY_MODULUS_MAX = 16384

# Escala mínima (bits) por poly_modulus_degree. SEAL exige que cada primo de la
# cadena sea distinto y ≡ 1 (mod 2N); con primos MUY pequeños no hay suficientes
# candidatos y falla con "failed to find enough qualifying primes". A mayor N,
# más estricta la congruencia y menos primos pequeños disponibles. Estos suelos
# son conservadores (validados empíricamente: escala 21 falla en N=16384).
ESCALA_MINIMA_POR_POLY = {4096: 18, 8192: 20, 16384: 25, 32768: 26}

# Barrido por defecto: grados del polinomio y escalas (bits) a probar.
# El rango de escalas está informado por las referencias: TenSEAL usa 21 bits en
# su CNN de MNIST; el paper CKKS usa 30-56. Una escala menor da módulos más
# pequeños (más rápido) a costa de precisión; es justo lo que interesa medir.
# El grado 7 ya es válido: al ajustar el polinomio al rango real (ver
# PrediccionHomomorfica), a mayor grado menor error, así que interesa medirlo.
GRADOS_POR_DEFECTO = (3, 5, 7)
# 25 bits es el suelo seguro para N>=16384 (escala 21 falla por falta de primos);
# 25/31/40 cubren pequeña/media/grande manteniendo margen de precisión.
ESCALAS_POR_DEFECTO = (25, 31, 40)

# --- Configuración ÓPTIMA (una sola, sin barrer) ---------------------------
# Para el resto de datasets (Diabetes tarda horas): NO se barre, se usa una única
# config derivada analíticamente. El grado óptimo se confirma con el barrido de
# cáncer (validación empírica -> default analítico). La escala es empíricamente
# irrelevante para la precisión (barrido: 25 vs 40 -> idéntica), así que se elige
# la menor cómoda; la cadena y el poly salen de la profundidad de la red.
GRADO_OPTIMO = 3
ESCALA_OPTIMA = 31


# --- Cálculo de profundidad ------------------------------------------------

def num_capas_lineales(hidden_layers):
    """Nº de capas lineales de la red = capas ocultas + 1 (la de salida)."""
    return len(hidden_layers) + 1


def profundidad_activacion(grado):
    """
    Niveles multiplicativos que consume evaluar el polinomio de activación de
    grado `grado`. Para `polyval` de grado d el circuito mínimo tiene profundidad
    ceil(log2 d) (paper CKKS, Cheon et al. 2017, Lemma 6/7; TenSEAL usa el circuito
    de profundidad mínima). 'sqt' (x^2) gasta 1 nivel.
        ceil(log2 3)=2, ceil(log2 5)=3, ceil(log2 7)=3.
    """
    if grado == "sqt":
        return 1
    return math.ceil(math.log2(grado))


def profundidad_multiplicativa(num_lineales, grado):
    """
    Niveles multiplicativos que consume el forward completo:
      - una multiplicación por cada capa lineal (num_lineales)
      - una activación polinómica entre capas (num_lineales - 1), cada una
        gastando profundidad_activacion(grado).
    Con una sola capa lineal NO hay activaciones, así que el grado es irrelevante.
    """
    activaciones = max(0, num_lineales - 1)
    return num_lineales + activaciones * profundidad_activacion(grado)


def menor_poly_modulus(total_bits, poly_max=None):
    """
    Menor poly_modulus_degree cuyo límite de seguridad admite `total_bits`, sin
    superar `poly_max` (tope por RAM disponible, ver POLY_MODULUS_MAX).
    Devuelve None si no hay ninguno viable: la configuración se descarta.
    """
    tope = POLY_MODULUS_MAX if poly_max is None else poly_max
    for n in POLY_MODULUS_CANDIDATOS:
        if n > tope:
            break                    # más allá del tope no cabe en RAM
        if total_bits <= MAX_BITS_SEGURIDAD[n]:
            return n
    return None  # no cabe con seguridad de 128 bits dentro del tope de RAM


def ram_estimada_gb(poly_modulus_degree, n_primos_totales):
    """
    Estimación grosera de la RAM del contexto CKKS + claves (Galois/relin), que es
    la parte FIJA (no baja con el batch). Calibrada con medidas en esta máquina:
    N=16384/10 primos -> 1.24 GB; N=32768/10 -> 2.58 GB; N=16384/12 -> 1.74 GB.
    Escala ~ linealmente con N y con el nº de primos. Sirve para avisar antes de
    lanzar una configuración que no va a caber (el pico real de la predicción
    paralela es varias veces esto).
    """
    return 1.24 * (poly_modulus_degree / 16384) * (n_primos_totales / 10)


# --- Construcción de una configuración -------------------------------------

def construir_config(num_lineales, grado, escala_bits,
                     margen=MARGEN_NIVELES, primo_especial=PRIMO_ESPECIAL):
    """
    Devuelve un dict con la configuración CKKS para una red de `num_lineales`
    capas lineales, o None si no es viable (no cabe con seguridad de 128 bits, o
    requiere un poly_modulus_degree por encima de POLY_MODULUS_MAX, es decir, no
    cabe en la RAM de esta máquina).
    """
    niveles = profundidad_multiplicativa(num_lineales, grado)
    n_primos = niveles + margen
    coeff = [primo_especial] + [escala_bits] * n_primos + [primo_especial]
    total = sum(coeff)
    poly = menor_poly_modulus(total)
    if poly is None:
        return None
    # Descartar escalas demasiado pequeñas para ese N: SEAL no hallaría suficientes
    # primos distintos de ese tamaño (ver ESCALA_MINIMA_POR_POLY).
    if escala_bits < ESCALA_MINIMA_POR_POLY.get(poly, 0):
        return None
    return {
        "grado": grado,
        "escala_bits": escala_bits,
        "global_scale_bits": escala_bits,
        "niveles_necesarios": niveles,
        "n_primos_medios": n_primos,
        "coeff_mod_bit_sizes": coeff,
        "total_bits": total,
        "poly_modulus_degree": poly,
        "ram_contexto_gb": round(ram_estimada_gb(poly, len(coeff)), 2),
        "etiqueta": f"g{grado}_s{escala_bits}_N{poly}",
    }


def config_optima(num_lineales, grado=GRADO_OPTIMO, escala_bits=ESCALA_OPTIMA,
                  margen=MARGEN_NIVELES):
    """
    Devuelve la ÚNICA configuración CKKS óptima (analítica) para una red de
    `num_lineales` capas lineales. Es la que se usa para los datasets que NO se
    barren (Diabetes, etc.): no hay que probar, la profundidad de la red fija la
    cadena y el poly, y la escala se elige por debajo (irrelevante para precisión).

    Si la red no tiene activaciones (una sola capa lineal), el grado es irrelevante.
    """
    if num_lineales <= 1:
        grado = GRADO_OPTIMO
    cfg = construir_config(num_lineales, grado, escala_bits, margen)
    if cfg is None:
        raise ValueError(
            f"No hay configuración CKKS viable para {num_lineales} capas lineales, "
            f"grado {grado}, escala {escala_bits} bits: no cabe con seguridad de 128 "
            f"bits dentro de poly_modulus_degree <= {POLY_MODULUS_MAX} "
            f"(tope por RAM disponible). Baja la escala o el grado.")
    return cfg


def generar_configuraciones(num_lineales, grados=GRADOS_POR_DEFECTO,
                            escalas=ESCALAS_POR_DEFECTO, margen=MARGEN_NIVELES):
    """
    Genera todas las configuraciones CKKS a probar para una arquitectura.

    Si la red no tiene activaciones (una sola capa lineal), el grado no influye
    en el cálculo, así que se colapsa a un único grado para no repetir corridas
    idénticas.
    """
    if grados is None:
        grados = GRADOS_POR_DEFECTO
    if escalas is None:
        escalas = ESCALAS_POR_DEFECTO
    if num_lineales <= 1:
        grados = (grados[0],)

    configs, vistos = [], set()
    for g in grados:
        for s in escalas:
            c = construir_config(num_lineales, g, s, margen)
            if c is None:
                continue
            clave = (c["grado"], c["global_scale_bits"], c["poly_modulus_degree"],
                     tuple(c["coeff_mod_bit_sizes"]))
            if clave in vistos:
                continue
            vistos.add(clave)
            configs.append(c)
    return configs


# --- Prueba rápida por línea de comandos -----------------------------------

if __name__ == "__main__":
    ejemplos = [
        ("5 -> 2", []),
        ("6 -> 4 -> 2", [4]),
        ("30 -> 16 -> 8 -> 2", [16, 8]),
        ("16 -> 10 -> 7", [10]),
    ]
    for nombre, hidden in ejemplos:
        L = num_capas_lineales(hidden)
        print(f"\n### {nombre}   (capas lineales = {L})")
        for c in generar_configuraciones(L):
            print(f"  {c['etiqueta']:16s} grado={c['grado']} escala={c['escala_bits']}b "
                  f"niveles={c['niveles_necesarios']} primos_medios={c['n_primos_medios']} "
                  f"coeff={c['coeff_mod_bit_sizes']} total={c['total_bits']}b "
                  f"poly={c['poly_modulus_degree']}")
