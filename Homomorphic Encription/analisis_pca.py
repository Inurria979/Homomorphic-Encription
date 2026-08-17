"""
CRITERIO DE REDUCCIÓN DE DIMENSIONALIDAD (PCA)
==============================================
Calcula cuántas componentes principales retiene cada dataset con el criterio
adoptado: la MENOR cantidad que explique al menos el 90% de la varianza. Se
elige ese umbral porque reproduce el valor de la literatura para Breast Cancer
Wisconsin (7 componentes), así que la regla queda validada donde hay referencia
y se aplica igual a los demás.

El espectro se calcula sobre train+val, apartando el test con la misma semilla
fija que usa DataProcessor.split_data. Así k no mira el test y, como ese corte no
depende de la semilla del Monte Carlo (solo se remezcla la frontera interna entre
train y val), sale el mismo valor para las 50 semillas de un experimento.

Los datos se cargan con DataProcessor, de modo que pasan por el mismo
preprocesado que los experimentos (tope de muestras, codificación de
categóricas, imputación y estandarización).

`componentes_optimas` es además lo que usa main.py cuando se le pasa
`--model_size 0`, para que el número de la memoria y el del pipeline salgan del
mismo sitio.

Uso: python analisis_pca.py
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from DataProcessor import DataProcessor

DATASETS = (17, 27, 544, 891)
UMBRAL = 0.90


def varianza_acumulada(procesador):
    """Varianza acumulada por número de componentes, sobre train+val."""
    X_train_val, _, _, _ = train_test_split(
        procesador.X, procesador.y, test_size=0.2, random_state=42,
        stratify=procesador.y)
    pca = PCA().fit(StandardScaler().fit_transform(X_train_val))
    return np.cumsum(pca.explained_variance_ratio_)


def componentes_optimas(dataset_id, umbral=UMBRAL):
    """Menor k que explica al menos `umbral` de la varianza del dataset."""
    return int(np.searchsorted(varianza_acumulada(DataProcessor(dataset_id)),
                               umbral) + 1)


if __name__ == '__main__':
    print(f"{'Dataset':<32}{'Variables':>10}{'Componentes':>13}"
          f"{'Varianza':>10}{'1ª comp.':>10}")
    print('-' * 75)

    for dataset_id in DATASETS:
        procesador = DataProcessor(dataset_id)
        acumulada = varianza_acumulada(procesador)
        k = int(np.searchsorted(acumulada, UMBRAL) + 1)
        print(f"{procesador.dataset_nombre:<32}{len(acumulada):>10}{k:>13}"
              f"{acumulada[k - 1] * 100:>9.1f}%{acumulada[0] * 100:>9.1f}%")
