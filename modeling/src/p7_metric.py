import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer


def business_gain_score(tn, fp, fn, tp, gain_tp=1, gain_tn=1, loss_fp=-1, loss_fn=-10):

    param_bg = {
        "loss_fn": -50_000,
        "loss_fp": -5_000,
        "gain_tp": 5_000,
        "gain_tn": 5_000,
    }

    gain = gain_tp * tp + gain_tn * tn + loss_fp * fp + loss_fn * fn
    # print("gain", gain)

    # Nombre de défauts réels (=nombre de positifs)
    n_default = tp + fn
    # print("Nombre de défauts réels", n_default)

    # Nombre de remboursements OK réels (=nombre de négatifs)
    n_ok = tn + fp
    # print("Nombre de OK réels", n_ok)

    # Maximum de gain métier en valeur, si on a tout prédit correctement
    max_gain = gain_tp * n_default + gain_tn * n_ok
    # print("MAx_gain", max_gain)

    # Minimum de gain métier en valeur, si toutes les prédictions sont fausses
    min_gain = loss_fn * n_default + loss_fp * n_ok
    # print("min gain", min_gain)

    # On ramène entre 0 et 1 (comme un min_max_scaler)
    normalized_gain = (gain - min_gain) / (max_gain - min_gain)
    return normalized_gain


# Créer une métrique personnalisée à partir de la fonction de perte
# custom_f1 = make_scorer(weight_f1, greater_is_better=True)


def business_gain_from_pred(
    y_true,
    y_pred,
    param_bg={
        "loss_fn": -50_000,
        "loss_fp": -5_000,
        "gain_tp": 5_000,
        "gain_tn": 5_000,
    },
):
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    return business_gain_score(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        gain_tp=param_bg["gain_tp"],
        gain_tn=param_bg["gain_tn"],
        loss_fp=param_bg["loss_fp"],
        loss_fn=param_bg["loss_fn"],
    )


# On transforme en scorer pour pouvoir l'utiliser dans les courbes d'apprentissage sklearn
business_gain_scorer = make_scorer(business_gain_from_pred)
