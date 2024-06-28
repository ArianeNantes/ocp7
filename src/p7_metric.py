import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def pd_pred_prob_to_binary(y_prob, threshold=0.5):
    return pd.Series([int(proba > threshold) for proba in y_prob])


def cu_pred_prob_to_binary(y_prob, threshold=0.5):
    return y_prob.apply(lambda proba: int(proba > threshold))


# Il n'existe pas de recal_score pour cuda équivallent à celui de sklearn
# La fonc peut être utilisée aussi pour pandas
def cupd_recall_score(tp, fn):
    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)


# Pénalise les Faux Negatifs dans le F1 Score
def penalize_f1(fp, fn, tp, weight_fn=10, weight_fp=1):
    # f1 standard = 2 * tp / (2 * tp + fp + fn)

    weighted_f1 = (
        2 * tp / (2 * tp + (weight_fn * fn + weight_fp * fp) / (weight_fn + weight_fp))
    )
    return weighted_f1


# Pénalise les Faux Negatifs dans le F1 Score
# Inutile de recalculer la matrice de confusion à l'intérieur surtout que cela dépend si cuda ou pandas
def penalize_f1_old(y_true, y_pred, weight_fn=10, weight_fp=1):

    _, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # fn = Nombre de faux négatifs (oubli de prédire un défaut)
    # fp = Nombre de faux positifs (défaut prédit à tort)
    # f1 standard = 2 * tp / (2 * tp + fp + fn)

    weighted_f1 = (
        2 * tp / (2 * tp + (weight_fn * fn + weight_fp * fp) / (weight_fn + weight_fp))
    )
    return weighted_f1


def penalize_business_gain(
    tn, fp, fn, tp, gain_tp=0, gain_tn=50, loss_fp=10, loss_fn=100
):
    gain = gain_tp * tp + gain_tn * tn - loss_fp * fp - loss_fn * fn

    n_default = tp + fp
    n_ok = tn + fn
    max_gain = gain_tp * n_default + gain_tn * n_ok
    min_gain = -loss_fp * n_default - loss_fn * n_ok
    normalized_gain = (gain - min_gain) / (max_gain - min_gain)

    return normalized_gain


# Inutile de recalculer la matrice de confusion à l'intérieur surtout que cela dépend si cuda ou pandas
def penalize_business_gain_old(
    y_true, y_pred, gain_tp=0, gain_tn=50, loss_fp=10, loss_fn=100
):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    gain = gain_tp * tp + gain_tn * tn - loss_fp * fp - loss_fn * fn

    n_default = y_true.sum()
    n_ok = len(y_true) - n_default
    max_gain = gain_tp * n_default + gain_tn * n_ok
    min_gain = -loss_fp * n_default - loss_fn * n_ok
    normalized_gain = (gain - min_gain) / (max_gain - min_gain)

    return normalized_gain


# Créer une métrique personnalisée à partir de la fonction de perte
# custom_f1 = make_scorer(weight_f1, greater_is_better=True)
