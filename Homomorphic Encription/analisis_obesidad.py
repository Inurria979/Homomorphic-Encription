"""
ANÁLISIS DE LOS FALLOS EN OBESIDAD
==================================
La predicción homomórfica se desploma en las redes de Obesidad CON capas ocultas
(16->10->7 pasa de 93.85% a 44.92%), mientras que las redes sin capas ocultas dan
coste cero y Breast Cancer con DOS capas ocultas también. Este script contrasta la
explicación: no es el error de aproximación del polinomio, sino la geometría del
problema de clasificación.

Hace dos análisis sobre las matrices de confusión ya generadas:

1. DISTANCIAS. Reordena las clases por severidad y mide a cuántos niveles de
   distancia se producen los fallos. Si el ruido del polinomio desdibuja fronteras
   estrechas, los fallos deben concentrarse entre clases contiguas.

2. JERARQUÍA. Colapsa las 7 clases al agrupamiento infrapeso / normopeso /
   sobrepeso-obesidad para estimar qué exactitud alcanzaría la primera etapa de
   una pipeline jerárquica sobre datos cifrados.

No modifica nada: solo lee los informes de texto de experimentos_v2/<exp>/pred
y .../pred_hom, que genera Prediccion.save_results_txt.

Uso:
    venv/bin/python analisis_obesidad.py
"""

import os
import re

RAIZ = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(RAIZ, 'experimentos_v2')

# OJO: LabelEncoder codifica las clases por orden ALFABÉTICO, que no coincide con
# el orden de severidad. Leer la matriz de confusión sin remapear hace que
# confusiones entre clases vecinas parezcan saltos grandes.
NOMBRE = {
    0: 'Insufficient', 1: 'Normal', 2: 'Obesity_I', 3: 'Obesity_II',
    4: 'Obesity_III', 5: 'Overweight_I', 6: 'Overweight_II',
}

# Orden real de severidad, de menor a mayor peso corporal.
SEVERIDAD = [0, 1, 5, 6, 2, 3, 4]
RANGO = {etiqueta: posicion for posicion, etiqueta in enumerate(SEVERIDAD)}

# Agrupamiento de la pipeline jerárquica propuesta como trabajo futuro.
GRUPO = {0: 'infrapeso', 1: 'normopeso',
         5: 'sobrepeso', 6: 'sobrepeso', 2: 'sobrepeso', 3: 'sobrepeso', 4: 'sobrepeso'}

EXPERIMENTOS = [
    # Con capas ocultas: es donde aparece el coste
    'Obesity_16-10-7', 'Obesity_16-10-7_rg', 'Obesity_12-5-7_rg',
    # Sin capas ocultas: control, plana y cifrada deben ser idénticas
    'Obesity_16-7', 'Obesity_12-7_rg',
]


def leer_matriz(ruta):
    """
    Extrae la matriz de confusión del informe de texto de una predicción.
    Devuelve una lista de filas (real) por columnas (predicho), o None.
    """
    if not os.path.isfile(ruta):
        return None
    filas = []
    with open(ruta, encoding='utf-8') as f:
        for linea in f.read().splitlines():
            m = re.match(r'^Cl(\d+)\s+(.*)$', linea.strip())
            if not m:
                continue
            campos = m.group(2).split()
            # La cabecera de la tabla también empieza por "Cl0"; nos quedamos
            # solo con las filas cuyos campos son todos numéricos.
            if campos and all(c.isdigit() for c in campos):
                filas.append([int(c) for c in campos])
    return filas or None


def analizar_distancias(nombre, matriz):
    """Reparte los fallos según su distancia en la escala de severidad."""
    n = len(matriz)
    total = sum(sum(fila) for fila in matriz)
    aciertos = sum(matriz[i][i] for i in range(n))
    errores = total - aciertos

    print(f"\n{'=' * 72}\n{nombre}")
    print(f"  {total} muestras · {aciertos} aciertos ({100 * aciertos / total:.2f}%) "
          f"· {errores} fallos")
    if not errores:
        print("  Sin fallos que analizar.")
        return

    por_distancia = {}
    detalle = []
    for real in range(n):
        for pred in range(n):
            if real == pred or matriz[real][pred] == 0:
                continue
            cuenta = matriz[real][pred]
            d = abs(RANGO[real] - RANGO[pred])
            por_distancia[d] = por_distancia.get(d, 0) + cuenta
            detalle.append((cuenta, real, pred, d))

    print("\n  Fallos por distancia en la escala de severidad:")
    for d in sorted(por_distancia):
        c = por_distancia[d]
        print(f"    distancia {d}: {c:4d}  ({100 * c / errores:5.1f}%)  "
              f"{'█' * round(40 * c / errores)}")

    contiguos = por_distancia.get(1, 0)
    lejanos = sum(c for d, c in por_distancia.items() if d >= 3)
    print(f"\n  → {contiguos}/{errores} = {100 * contiguos / errores:.1f}% "
          f"entre clases CONTIGUAS")
    print(f"  → {lejanos}/{errores} = {100 * lejanos / errores:.1f}% "
          f"saltan 3 niveles o más")

    print("\n  Confusiones principales:")
    for cuenta, real, pred, d in sorted(detalle, reverse=True)[:6]:
        print(f"    {cuenta:4d}  {NOMBRE[real]:14s} → {NOMBRE[pred]:14s}  (distancia {d})")


def analizar_jerarquia():
    """
    Exactitud que se obtendría agrupando las 7 clases en infrapeso / normopeso /
    sobrepeso. Un fallo entre dos clases del mismo grupo deja de contar como tal.
    """
    print(f"\n{'=' * 72}")
    print("PIPELINE JERÁRQUICA: exactitud de la primera etapa (3 grupos)")
    print(f"{'=' * 72}")
    print(f"{'experimento':22s} {'tipo':8s} {'7 clases':>9s} {'3 grupos':>9s} {'ganancia':>9s}")
    print('-' * 62)
    for exp in EXPERIMENTOS:
        for etiqueta, fichero in (('plana', 'pred'), ('cifrada', 'pred_hom')):
            matriz = leer_matriz(os.path.join(BASE, exp, fichero))
            if not matriz:
                continue
            n = len(matriz)
            total = sum(sum(fila) for fila in matriz)
            fino = sum(matriz[i][i] for i in range(n))
            grueso = sum(matriz[r][p] for r in range(n) for p in range(n)
                         if GRUPO[r] == GRUPO[p])
            a, b = 100 * fino / total, 100 * grueso / total
            print(f'{exp:22s} {etiqueta:8s} {a:8.2f}% {b:8.2f}% {b - a:+8.2f}')


def main():
    if not os.path.isdir(BASE):
        print(f"No existe {BASE}. Ejecuta antes los experimentos de Obesidad.")
        return
    for exp in EXPERIMENTOS:
        for etiqueta, fichero in (('PLANA', 'pred'), ('HOMOMÓRFICA', 'pred_hom')):
            matriz = leer_matriz(os.path.join(BASE, exp, fichero))
            if matriz:
                analizar_distancias(f'{exp}  ·  {etiqueta}', matriz)
    analizar_jerarquia()


if __name__ == '__main__':
    main()
