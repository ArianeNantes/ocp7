import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
)
from scipy import special

from src.p7_constantes import EPSILON


def clip_sigmoid(logits, epsilon=EPSILON):
    # fonction sigmoïd en plus stable (y_scores = 1 / (1 + np.exp(-raw_scores)))
    # Les branches neg et pos sont gérées et ne génèrent pas d'overflow
    prob = special.expit(logits)
    prob[prob >= 1] = 1 - epsilon
    prob[prob <= 0] = epsilon
    return prob


# Imite la fonction de perte binary logloss. Permet de tester les fonctions personnalisées
def logloss_objective(pred_raw_scores, train_data):
    y_true = train_data.get_label()

    prob = clip_sigmoid(pred_raw_scores)

    grad = prob - y_true
    hess = prob * (1 - prob)
    return grad, hess


def logloss_metric(pred_raw_scores, train_data):
    y_true = train_data.get_label()
    prob = clip_sigmoid(pred_raw_scores)

    ll = np.empty_like(prob)
    pos = y_true >= 1 - EPSILON
    ll[pos] = np.log(prob[pos])
    ll[~pos] = np.log(1 - prob[~pos])

    is_higher_better = False
    return "logloss", -ll.mean(), is_higher_better


# Pour diminuer les Faux négatifs, choisir alpha > 1 (pour diminuer les Faux positifs, alpha < 1)
# Plus alpha est grand, plus on augmente le recall et on diminue la précision
# Plus alpha est petit, plus on augmente la précision et on diminue le recall
def weighted_logloss_from_preds(logits, train_data, alpha=1.0):
    y_true = train_data.get_label()
    prob = clip_sigmoid(logits)

    grad = (y_true * prob * (alpha - 1)) + prob - (alpha * y_true)
    hess = (y_true * (alpha - 1) + 1) * prob * (1 - prob)

    return grad, hess


def make_objective_weighted_logloss(alpha=1.0):
    def objective_weighted_logloss_wrapper(logits, train_data):
        return weighted_logloss_from_preds(
            logits=logits,
            train_data=train_data,
            alpha=alpha,
        )

    return objective_weighted_logloss_wrapper


# Non utilisée car on mesure l'auc ou le business gain pour les callbacks.
# Si on voulait l'utiliser dans feval, il faudrait faire un wrapper pour respecter
# la signature imposer (sans alpha). Voir exemple business_gain
"""def weighted_logloss_metric(logits, dtrain, alpha=1.0):

    y = dtrain.get_label()
    probs = clip_sigmoid(logits)
    elements = -alpha * y * np.log(probs) - (1 - y) * np.log(1 - probs)
    is_higher_better = False
    return f"WCE_alpha{alpha}", (np.sum(elements) / len(y)), is_higher_better"""


# Permet d'intégrer l'auc score dans feval pour lgbm afin de pruner sur l'auc avec optuna
# Nécessaire si la loss est une fonction personnalisée car le modèle produit alors des logits au lieu des probas
# La signature est imposée
def feval_auc_logits(pred_logits, lgbDataset):
    pred_probs = special.expit(pred_logits)
    y_true = lgbDataset.get_label()
    is_higher_better = True
    return "auc", roc_auc_score(y_true, pred_probs), is_higher_better


def feval_auc_probs(pred_probs, lgbDataset):
    y_true = lgbDataset.get_label()
    is_higher_better = True
    return "auc", roc_auc_score(y_true, pred_probs), is_higher_better


def business_gain_from_preds(
    preds, lgbDataset, threshold_prob=0.5, preds_are_logits=True
):
    if preds_are_logits:
        y_prob = special.expit(preds)
    else:
        y_prob = preds
    y_true = lgbDataset.get_label()

    y_pred = pd_pred_prob_to_binary(y_prob=y_prob, threshold=threshold_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    business_gain = penalize_business_gain(tn=tn, fp=fp, fn=fn, tp=tp)
    is_higher_better = True
    return "business_gain", business_gain, is_higher_better


# Pour pruner sur le business gain dans optuna, il faut le calculer dans feval.
# les signatures des feval sont imposées et ne prennent que 2 arguments.
# Pour intégrer le seuil de proba en argument supplémentaire, nous utilisons un wrapper qui respecte la signature imposée
def make_feval_business_gain(threshold_prob=0.5, preds_are_logits=True):
    def feval_business_gain_wrapper(preds, lgbDataset):
        return business_gain_from_preds(
            preds,
            lgbDataset,
            threshold_prob=threshold_prob,
            preds_are_logits=preds_are_logits,
        )

    return feval_business_gain_wrapper


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


# [TODO] Remplacer partout l'appel de cette fonc par business_gain_score puis supprimer la fonction
def penalize_business_gain(tn, fp, fn, tp, gain_tp=1, gain_tn=1, loss_fp=1, loss_fn=10):
    gain = gain_tp * tp + gain_tn * tn - loss_fp * fp - loss_fn * fn

    n_default = tp + fn
    n_ok = tn + fp
    max_gain = gain_tp * n_default + gain_tn * n_ok
    min_gain = -loss_fp * n_default - loss_fn * n_ok
    normalized_gain = (gain - min_gain) / (max_gain - min_gain)

    return normalized_gain


# Calcule le score métier. Le score maximum est 1
def business_gain_score(tn, fp, fn, tp, gain_tp=5, gain_tn=1, loss_fp=1, loss_fn=10):
    gain = gain_tp * tp + gain_tn * tn - loss_fp * fp - loss_fn * fn

    # Nombre de défauts réels
    n_default = tp + fn
    # Nombre de remboursements OK réels
    n_ok = tn + fp

    # Maximum de gain métier en valeur, si on a tout prédit correctement
    max_gain = gain_tp * n_default + gain_tn * n_ok
    # Minimum de gain métier en valeur, si toutes les prédictions sont fausses
    min_gain = -loss_fp * n_default - loss_fn * n_ok
    # On ramène en pourcentage
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
