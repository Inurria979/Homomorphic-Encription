"""
MÉTRICAS DE MACHINE LEARNING
============================
Cálculo exhaustivo de métricas de clasificación a partir de las etiquetas
reales, las predichas y (opcionalmente) las probabilidades. Funciona tanto
para problemas binarios como multiclase.

Todas las tasas se devuelven como FRACCIONES (0-1); quien las guarde decide
si las pasa a porcentaje. MCC y kappa van en su rango natural (-1..1).

Diseñado para ser la única fuente de verdad de métricas: lo usan tanto la
predicción plana como la homomórfica (esta última sin probabilidades, así
que las métricas que las requieren -ROC-AUC, log-loss- quedan a None).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, precision_recall_fscore_support,
    roc_auc_score, log_loss, precision_score,
)


def _ratio(numerador, denominador):
    return float(numerador) / float(denominador) if denominador else 0.0


def _mcc_seguro(y_true, y_pred):
    """
    Coeficiente de Matthews, robusto ante casos degenerados (p. ej. una sola
    clase presente, donde el MCC no está definido): devuelve 0.0 en vez de fallar.
    """
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        return 0.0


def _especificidad_por_clase(cm):
    """
    Especificidad one-vs-rest para cada clase a partir de la matriz de
    confusión: para la clase i, todo lo que no es i es "negativo".
      TN_i = total - fila_i - columna_i + cm[i,i]
      FP_i = columna_i - cm[i,i]
    """
    cm = np.asarray(cm, dtype=float)
    total = cm.sum()
    especificidades = []
    for i in range(cm.shape[0]):
        fila = cm[i, :].sum()
        col = cm[:, i].sum()
        tp = cm[i, i]
        fp = col - tp
        tn = total - fila - col + tp
        especificidades.append(_ratio(tn, tn + fp))
    return especificidades


def calcular_metricas(y_true, y_pred, y_proba=None, num_clases=None):
    """
    Devuelve un dict con un conjunto amplio de métricas de clasificación.

    y_true, y_pred : arrays de etiquetas enteras.
    y_proba        : (opcional) matriz (n_muestras x n_clases) de
                     probabilidades. Habilita ROC-AUC y log-loss.
    num_clases     : (opcional) fuerza el nº de clases/etiquetas; si no se
                     pasa se infiere de los datos.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if num_clases is not None:
        etiquetas = list(range(num_clases))
    else:
        etiquetas = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    n_clases = len(etiquetas)
    es_binario = n_clases <= 2

    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
    n = len(y_true)
    n_correctas = int((y_true == y_pred).sum())

    # --- Promedios macro / micro / weighted ---
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=etiquetas, average='macro', zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=etiquetas, average='micro', zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=etiquetas, average='weighted', zero_division=0)

    # --- Por clase ---
    prec_c, rec_c, f1_c, soporte_c = precision_recall_fscore_support(
        y_true, y_pred, labels=etiquetas, average=None, zero_division=0)
    espec_c = _especificidad_por_clase(cm)
    por_clase = []
    for idx, etq in enumerate(etiquetas):
        por_clase.append({
            'clase': int(etq),
            'precision': float(prec_c[idx]),
            'recall': float(rec_c[idx]),
            'f1': float(f1_c[idx]),
            'especificidad': float(espec_c[idx]),
            'soporte': int(soporte_c[idx]),
        })

    metricas = {
        'n_samples': n,
        'num_classes': n_clases,
        'n_correct': n_correctas,
        'n_incorrect': n - n_correctas,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'error_rate': _ratio(n - n_correctas, n),
        'precision_macro': float(prec_macro),
        'precision_micro': float(prec_micro),
        'precision_weighted': float(prec_w),
        'recall_macro': float(rec_macro),
        'recall_micro': float(rec_micro),
        'recall_weighted': float(rec_w),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'f1_weighted': float(f1_w),
        'mcc': _mcc_seguro(y_true, y_pred),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred, labels=etiquetas)),
        'conf_matrix': cm,
        'per_class': por_clase,
        'roc_auc': None,
        'log_loss': None,
    }

    # --- Métricas binarias específicas (TP/TN/FP/FN, sens/espec/PPV/NPV) ---
    if es_binario and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metricas.update({
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'sensitivity': _ratio(tp, tp + fn),   # recall de la clase positiva
            'specificity': _ratio(tn, tn + fp),   # recall de la clase negativa
            'ppv': _ratio(tp, tp + fp),            # precisión positiva
            'npv': _ratio(tn, tn + fn),            # precisión negativa
        })
    else:
        # Multiclase: sens = recall macro, espec = especificidad macro,
        # ppv = precisión macro, npv = precisión de la clase 0 (referencia)
        try:
            npv_ref = float(precision_score(y_true, y_pred, labels=etiquetas,
                                            average=None, zero_division=0)[0])
        except Exception:
            npv_ref = 0.0
        metricas.update({
            'tp': None, 'tn': None, 'fp': None, 'fn': None,
            'sensitivity': float(rec_macro),
            'specificity': float(np.mean(espec_c)),
            'ppv': float(prec_macro),
            'npv': npv_ref,
        })

    # --- Métricas que requieren probabilidades ---
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            if es_binario and y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metricas['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metricas['roc_auc'] = float(roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro',
                    labels=etiquetas))
        except Exception:
            metricas['roc_auc'] = None
        try:
            metricas['log_loss'] = float(log_loss(y_true, y_proba, labels=etiquetas))
        except Exception:
            metricas['log_loss'] = None

    # 'sensitivity' y 'specificity' ya quedan fijadas en ambas ramas (binaria y
    # multiclase); son las claves que espera Prediccion.save_results_txt.
    return metricas


def distribucion_clases(y):
    """Devuelve {clase: nº de muestras} para un array de etiquetas."""
    y = np.asarray(y).ravel()
    valores, cuentas = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(valores, cuentas)}


def ratio_balance(y):
    """
    Ratio de balance de clases = (clase minoritaria / clase mayoritaria).
    1.0 = perfectamente balanceado; cercano a 0 = muy desbalanceado.
    """
    dist = distribucion_clases(y)
    if not dist:
        return None
    cuentas = list(dist.values())
    return _ratio(min(cuentas), max(cuentas))
