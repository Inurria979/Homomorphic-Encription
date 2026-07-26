# 🔐 Inferencia con Cifrado Homomórfico (FHE) sobre Redes Neuronales

Este proyecto entrena redes neuronales para clasificación y las evalúa de dos
formas sobre el mismo conjunto de test:

1. **Predicción plana** — inferencia normal, sin cifrar (línea base).
2. **Predicción homomórfica** — la red completa se ejecuta **sobre datos
   cifrados** con [TenSEAL](https://github.com/OpenMined/TenSEAL) (esquema CKKS).
   El servidor que infiere **nunca ve los datos en claro**: cifra la entrada,
   propaga por las capas y solo se descifra el resultado final.

El objetivo es medir el **coste** (tiempo, RAM, precisión) de garantizar la
privacidad mediante cifrado homomórfico, comparando ambas predicciones con un
conjunto amplio de métricas de ML y de sistema que se guardan en una base de
datos SQLite y un Excel con gráficas.

---

## 📑 Índice

- [Idea clave: aproximar la ReLU por un polinomio](#-idea-clave-aproximar-la-relu-por-un-polinomio)
- [Instalación](#-instalación)
- [El ejemplo del cáncer de mama (paso a paso)](#-el-ejemplo-del-cáncer-de-mama-paso-a-paso)
- [Ejecutar un experimento individual](#-ejecutar-un-experimento-individual-mainpy)
- [La batería de experimentos](#-la-batería-de-experimentos)
- [Barrido de parámetros CKKS](#-barrido-de-parámetros-ckks)
- [Control de memoria (evita tumbar WSL)](#-control-de-memoria-evita-tumbar-wsl)
- [Resultados: base de datos + Excel](#-resultados-base-de-datos--excel)
- [Datasets soportados](#-datasets-soportados)
- [Estructura del proyecto](#-estructura-del-proyecto)

---

## 🔬 Idea clave: aproximar la ReLU por un polinomio

El cifrado CKKS solo sabe **sumar y multiplicar** sobre datos cifrados. Una capa
lineal (`Wx + b`) encaja perfecto, pero la activación **ReLU = max(0, x)** no es
polinómica y no se puede calcular sobre el cifrado.

La solución: aproximar la ReLU por un **polinomio**, que sí es sumas y
multiplicaciones. El modelo se **entrena** con la ReLU normal de PyTorch (rápido y
preciso) y en la **predicción cifrada** se sustituye por su aproximación
polinómica.

El polinomio **se ajusta por mínimos cuadrados al rango real** de las
pre-activaciones (medido en una pasada en claro), **no** a un intervalo fijo
[-1, 1]. Esto es importante: aunque los pesos rondan [-1, 1], la pre-activación
`z = Σ wᵢ·xᵢ + b` suma sobre muchas entradas (*fan-in*) y con features
estandarizadas alcanza **±8…±15**. Un polinomio ajustado a [-1, 1] se evaluaría
muy fuera de su zona (extrapolación). Ajustándolo al rango real, **a mayor grado,
mejor aproximación** (grados 3, 5, 7…), hasta donde permita la profundidad
multiplicativa disponible del cifrado.

Los parámetros del propio esquema CKKS (cadena de módulos, `poly_modulus_degree`,
escala) se **derivan de la profundidad** de la red, y se pueden **barrer** en el
dataset rápido (ver [Barrido de parámetros CKKS](#-barrido-de-parámetros-ckks)).

> Justificación completa, con órdenes de magnitud medidos y referencias a los
> papers de CKKS y TenSEAL, en
> [`docs/afinado_cifrado_ckks.md`](docs/afinado_cifrado_ckks.md).

---

## ⚙️ Instalación

El proyecto usa un entorno virtual en `.venv/` (PyTorch CPU, TenSEAL, scikit-learn,
psutil, openpyxl...). Todo lo necesario está en `requirements.txt`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

A partir de aquí, **todos los comandos usan `.venv/bin/python`** (el Python del
sistema no tiene las dependencias).

---

## 🧬 El ejemplo del cáncer de mama (paso a paso)

El dataset **Breast Cancer Wisconsin (Diagnostic)** (UCI id 17) tiene 569
biopsias con 30 características numéricas cada una, y la etiqueta es el
diagnóstico: **benigno** o **maligno**. Es el ejemplo canónico del proyecto.

### 1. Lanzar el experimento

```bash
.venv/bin/python main.py Breast_Cancer --train --dataset 17 --model_size 5 --pca 5
```

Qué significa cada parte:

| Argumento | Efecto |
|-----------|--------|
| `Breast_Cancer` | carpeta donde se guardan modelo, gráficas y resultados |
| `--train` | entrena desde cero (sin el flag, solo predice con un modelo ya guardado) |
| `--dataset 17` | Breast Cancer Wisconsin |
| `--pca 5` | reduce las 30 características a **5 componentes principales** |
| `--model_size 5` | la red tiene entrada de tamaño 5 (debe coincidir con `--pca`) |

Con `--model_size 5` la red es directamente `5 → 2` (sin capas ocultas): 5
entradas (los componentes PCA) y 2 salidas (benigno / maligno).

### 2. Qué ocurre por dentro

```
   [1] Carga y limpieza del dataset (UCI id 17)
        └─ split estratificado: 70% train / 10% val / 20% test
   [2] Entrenamiento Monte Carlo (5 semillas)
        └─ por cada semilla se remezcla train/val (el test queda intacto),
           se entrena con early stopping y se mide la accuracy de validación
        └─ se elige la mejor semilla y se reentrena con train+val
   [3] Predicción PLANA sobre el test  → fichero  Breast_Cancer/pred
   [4] Predicción HOMOMÓRFICA sobre el test cifrado → Breast_Cancer/pred_hom
   [5] Todo (métricas ML + sistema) se guarda en la BD y el Excel
```

### 3. Resultado típico

Con esta configuración el modelo alcanza en el conjunto de test:

| Métrica | Predicción plana | Predicción homomórfica |
|---------|-----------------:|-----------------------:|
| Accuracy | 98.25 % | 98.25 % |
| F1-macro | 98.10 % | 98.10 % |
| MCC | 0.963 | 0.963 |
| ROC-AUC | 99.57 % | — (¹) |
| Tiempo de inferencia | ~0.005 s | ~17-20 s |
| Pico de RAM del proceso | ~0.4 GB | ~3 GB |

**La predicción cifrada da exactamente el mismo resultado que la plana** — el
cifrado no degrada la precisión aquí — a cambio de ser miles de veces más lenta.
Ese es justo el coste de la privacidad que el proyecto mide.

(¹) ROC-AUC y log-loss necesitan las probabilidades por clase; la predicción
homomórfica solo descifra la clase ganadora, así que quedan a `None`.

### 4. Ficheros que deja en `Breast_Cancer/`

```
models/best_model.pth          modelo entrenado
model_config.json              arquitectura + mejor semilla
Nfeatures_5_experiment_info.json  resumen del entrenamiento
datasets/test_data.pt          conjunto de test reservado
scaler.pkl                     normalizador ajustado
results/                       gráficas de entrenamiento (png)
pred      / pred.png           resultados y matriz de confusión (plana)
pred_hom  / pred_hom.png       resultados y matriz de confusión (homomórfica)
```

Y en la raíz, `resultados/resultados.db` y `resultados/resultados.xlsx` con todo
lo anterior en forma consultable.

---

## 🚀 Ejecutar un experimento individual (`main.py`)

```bash
.venv/bin/python main.py <carpeta> [opciones]
```

| Opción | Por defecto | Descripción |
|--------|:-----------:|-------------|
| `--train` | (desactivado) | Entrena desde cero. Sin el flag, solo predice con el modelo ya guardado en `<carpeta>`. |
| `--dataset` | 17 | ID UCI: 17=Breast Cancer, 891=Diabetes, 544=Obesidad, 27=Credit. |
| `--model_size` | 5 | Tamaño de la capa de entrada (= nº de features tras PCA). |
| `--pca` | 0 | Componentes PCA (0 = sin PCA, usa todas las features). Debe igualar a `--model_size`. |
| `--h1`, `--h2` | 0 | Tamaños de capas ocultas (0 = no añadir esa capa). |
| `--nc` | 2 | Número de clases (2 binario, 7 obesidad...). |
| `--lr` | 0.001 | Learning rate. |
| `--seeds` | 5 | Nº de semillas del Monte Carlo. |
| `--rg` | 0 | Regularización L2 (weight decay). |
| `--bz`, `--nw` | 0 (auto) | Batch size / nº de hilos de la predicción homomórfica paralela. |
| `--ram` | 0.75 | Fracción de la RAM de WSL usable (ver [control de memoria](#-control-de-memoria-evita-tumbar-wsl)). |
| `--semilla` | (ninguna) | Semilla maestra para reproducibilidad. |

**Ejemplos:**

```bash
# Breast Cancer con las 30 características originales -> red 30 -> 16 -> 8 -> 2
.venv/bin/python main.py BC_full --train --dataset 17 --model_size 30

# Obesidad (7 clases) con PCA a 12 -> red 12 -> 5 -> 7, con regularización
.venv/bin/python main.py Obesidad --train --dataset 544 --model_size 12 --pca 12 --h1 5 --nc 7 --rg 1e-4

# Volver a predecir sin reentrenar (usa el modelo ya guardado)
.venv/bin/python main.py Breast_Cancer
```

---

## 🧪 La batería de experimentos

`ejecutar_experimentos.py` lanza en secuencia una batería de 12 experimentos
predefinidos (Breast Cancer, Credit, Obesidad y Diabetes con distintas
arquitecturas y con/sin regularización). Entrena y evalúa cada uno, y lo guarda
todo en la BD y el Excel.

```bash
.venv/bin/python ejecutar_experimentos.py --listar      # ver la lista
.venv/bin/python ejecutar_experimentos.py               # ejecutar todos
.venv/bin/python ejecutar_experimentos.py --rapidos     # todos menos Diabetes (que tarda horas)
.venv/bin/python ejecutar_experimentos.py BreastCancer_PCA5 Obesity_16-10-7   # solo algunos
```

Para dejar el PC trabajando sin supervisión:

```bash
nohup .venv/bin/python -u ejecutar_experimentos.py > salida_experimentos.log 2>&1 &
tail -f experimentos_v2/registro_ejecucion.log        # seguir el progreso
```

Características importantes:

- **Aislamiento por subproceso**: cada experimento se ejecuta como un
  `python main.py ...` independiente. Así el sistema operativo recupera el 100%
  de la RAM al terminar cada uno (clave para no acumular memoria y tumbar WSL en
  tandas largas).
- **`--continuar`**: reanuda una batería interrumpida, saltando los experimentos
  que ya tienen predicción homomórfica en la BD.
- **Reintento automático**: si un experimento falla (o se queda sin RAM), se
  reintenta una vez con la mitad de `--ram` antes de darlo por fallido y seguir.
- **Reproducible**: semilla maestra fija (`--semilla`, default 42).
- Los experimentos de **Diabetes** tardan **horas** (el test cifrado tiene miles
  de muestras); están marcados como lentos y van al final.

---

## 🔧 Barrido de parámetros CKKS

Además de afinar la parte de ML, se puede afinar el **cifrado**. `barrido_ckks.py`
barre —**solo en Breast Cancer**, que es rápido— la parte de cifrado: para cada
red entrena **una vez** y evalúa la predicción homomórfica con varias
configuraciones CKKS. Se barre el **grado del polinomio** ∈ {3, 5, 7} × la
**escala** ∈ {25, 31, 40} bits; la cadena de módulos y el `poly_modulus_degree`
se **derivan de la profundidad** de la red (redes llanas → `poly` pequeño y cadena
corta; profundas → mayores), en `configuracion_ckks.py`.

```bash
.venv/bin/python barrido_ckks.py --listar      # ver el plan y nº de subprocesos
.venv/bin/python barrido_ckks.py               # ejecutarlo (subprocesos aislados)
.venv/bin/python barrido_ckks.py --rapido      # subconjunto corto de validación
.venv/bin/python barrido_ckks.py CKKS_BC_Full30   # solo una(s) red(es)
.venv/bin/python barrido_ckks.py --continuar    # reanudar (salta lo ya entrenado)
```

Cada red produce **1 predicción plana + N homomórficas**, todas en la BD con sus
propios parámetros CKKS (grado, escala, cadena, `poly`) y con el **polinomio
ajustado** que se usó (rango de activación medido + coeficientes). Así se puede
comparar el coste (tiempo, RAM, precisión) en función de esos parámetros.

**El resto de datasets NO se barren** (Diabetes tardaría días): usan una única
**configuración óptima** derivada analíticamente de la profundidad de la red
(`configuracion_ckks.config_optima`), que es la que aplica por defecto la batería
normal (`ejecutar_experimentos.py`).

Fundamento y datos medidos: [`docs/afinado_cifrado_ckks.md`](docs/afinado_cifrado_ckks.md).

> ⚠️ **Límite de RAM:** `poly_modulus_degree` 32768 **no cabe** en esta máquina
> (11.7 GB en WSL): doblar N dobla la memoria del contexto y las claves, que es
> fija y no baja con el batch. Por eso `configuracion_ckks.POLY_MODULUS_MAX` lo
> topa en 16384 y las configuraciones que lo requieran se descartan al instante
> (en vez de abortar tras 6 h). Análisis completo:
> [`docs/incidente_ram_ckks_N32768.md`](docs/incidente_ram_ckks_N32768.md).

---

## 🧠 Control de memoria (evita tumbar WSL)

La predicción homomórfica paralela mantiene en RAM tensores CKKS de cientos de
MB por hilo y puede desbordar la memoria de WSL. Hay dos capas de protección:

1. **Límite de RAM** (`MonitorMemoria.py`): se usa como máximo `--ram` (75% por
   defecto) de la RAM total de WSL, con tope absoluto de 12 GB y dejando siempre
   ≥1 GB libre al sistema.
2. **Calibración automática**: antes de paralelizar se ejecuta una muestra sonda
   para medir cuánta RAM consume, y con esa medida se ajustan **el tamaño de lote
   y el número de hilos** para no superar el límite. El proceso va por oleadas,
   recalculando el margen antes de cada una.

Si pese a esperar (hasta 6 horas) la RAM no baja, el experimento se **aborta** de
forma controlada en vez de arriesgar la máquina (se recupera con `--continuar`).

El pico de RAM, los hilos efectivos y demás quedan registrados en la BD.

---

## 💾 Resultados: base de datos + Excel

Cada ejecución guarda automáticamente en:

- **`resultados/resultados.db`** — base de datos SQLite (la fuente de verdad).
- **`resultados/resultados.xlsx`** — Excel que se **regenera entero desde la BD**
  tras cada escritura (así nunca se desincronizan).

### Tablas

| Tabla | Contenido |
|-------|-----------|
| `experimentos` | Un registro por entrenamiento: dataset, arquitectura, hiperparámetros, mejor semilla, tamaños de dataset (train/val/test), distribución de clases y ratio de balance. |
| `entrenamientos_semilla` | Accuracy/loss de validación de cada semilla probada. |
| `predicciones` | Cada evaluación (`plana` u `homomorfica`) con **todas** las métricas (abajo). |
| `metricas_clase` | Desglose por clase de cada predicción (precisión, recall, F1, especificidad, soporte). |

### Métricas de ML (por predicción)

Accuracy, balanced accuracy, error rate, precision/recall/F1 en macro-micro-weighted,
sensibilidad, especificidad, PPV, NPV, **ROC-AUC**, **MCC** (Matthews), **Cohen's
kappa**, log-loss, aciertos/fallos, TP/TN/FP/FN y la matriz de confusión.

### Métricas de sistema (por predicción)

Núcleos lógicos/físicos, hilos de PyTorch, pico de hilos, tiempo de CPU
(usuario/sistema/total), **eficiencia de CPU**, tiempo de reloj, RAM
inicial/media/pico del proceso, RAM mínima disponible, pico de swap, y los
tiempos por fase (contexto, cifrado, predicción, descifrado) + parámetros CKKS
(grado, escala, cadena, `poly`) y el **polinomio ajustado** de la ReLU (rango de
activación medido `rango_activacion` + coeficientes `poly_coeffs`).

### Gráficas y matrices en el propio Excel

- **Graficas**: barras comparando plana vs homomórfica (accuracy, F1, throughput)
  y el factor de ralentización, RAM y eficiencia de CPU por experimento.
- **MatricesConfusion**: la matriz de cada predicción como rejilla con escala de
  color y la diagonal (aciertos) resaltada.
- **Comparativa**: plana vs homomórfica lado a lado con deltas.

Las tasas se guardan en **porcentaje (0-100)**; MCC/kappa en su rango natural
(-1..1); log-loss en bruto.

### Visualización web interactiva

Además del Excel, hay una web interactiva con formato de **artículo de
investigación** (React + ECharts en el front, FastAPI en el back) para explorar
los resultados: fidelidad de precisión plana vs homomórfica, coste temporal,
ralentización, RAM, matrices de confusión y métricas por clase, con filtros y
selectores en vivo.

```bash
./web/lanzar.sh          # compila el frontend y abre http://localhost:8000
```

Lee la BD en solo-lectura; no toca el pipeline. Detalles en
[`web/README.md`](web/README.md).

### Importar resultados antiguos

Para volcar a la BD/Excel carpetas de resultados ya generadas (ficheros `pred`,
`pred_hom`, `model_config.json`):

```bash
.venv/bin/python importar_resultados.py            # escanea todo el proyecto
.venv/bin/python importar_resultados.py Diabetes   # solo carpetas concretas
```

Es idempotente (las ya registradas se saltan).

---

## 📊 Datasets soportados

| ID | Dataset | Clases | Uso típico |
|----|---------|:------:|------------|
| 17 | Breast Cancer Wisconsin | 2 | ejemplo principal |
| 891 | CDC Diabetes Health Indicators | 2 | dataset grande (test de ~4000, cifrado LENTO) |
| 544 | Obesity Levels (hábitos) | 7 | ejemplo multiclase |
| 27 | Credit Approval | 2 | features mixtas categóricas/numéricas |

Los datasets con más de 20 000 muestras se recortan (muestreo estratificado)
para que la predicción cifrada no dure días.

---

## 📁 Estructura del proyecto

| Fichero | Responsabilidad |
|---------|-----------------|
| `main.py` | Punto de entrada de un experimento (entrenar + predecir plana + homomórfica). |
| `ejecutar_experimentos.py` | Orquesta la batería completa (subprocesos aislados). |
| `DataProcessor.py` | Descarga/limpia datasets de UCI, split y DataLoaders. |
| `Trainer.py` | Red `ConfigurableNN` + entrenamiento con early stopping. |
| `scripts.py` | Monte Carlo cross-validation + guardado de modelo y gráficas. |
| `Prediccion.py` | Base: carga modelo/test, calcula métricas, guarda txt/png. |
| `PrediccionPlana.py` | Inferencia sin cifrar. |
| `PrediccionHomomorfica.py` | Inferencia cifrada (todo el test de una vez); ajusta el polinomio de la ReLU al rango real. |
| `PrediccionHomomorficaParalela.py` | Inferencia cifrada por lotes en paralelo, con control de RAM. |
| `configuracion_ckks.py` | Deriva la configuración CKKS de la profundidad de la red (cadena, `poly`, escala) + `config_optima`. |
| `barrido_ckks.py` | Barre parámetros CKKS (grado × escala) en Breast Cancer, aislado por subprocesos. |
| `docs/afinado_cifrado_ckks.md` | Justificación del afinado del cifrado (pesos, pre-activaciones, polinomio, CKKS). |
| `docs/incidente_ram_ckks_N32768.md` | Análisis del agotamiento de RAM con `poly_modulus_degree` 32768 y su corrección. |
| `Metricas.py` | Cálculo exhaustivo de métricas de ML. |
| `MonitorSistema.py` | Telemetría de sistema (CPU, RAM, hilos) con psutil. |
| `MonitorMemoria.py` | Límite de RAM y calibración para la predicción paralela. |
| `ResultadosStore.py` | Persistencia en SQLite (esquema + migración). |
| `ExcelReport.py` | Genera el Excel (hojas, gráficas, matrices) desde la BD. |
| `importar_resultados.py` | Importa resultados antiguos en formato texto a la BD. |
| `web/` | Panel web interactivo (Streamlit + Plotly) para explorar los resultados. |

---

## ⚠️ Nota

Este proyecto es una **prueba de concepto** académica sobre el coste de la
inferencia con cifrado homomórfico. No es una implementación lista para
producción clínica.
