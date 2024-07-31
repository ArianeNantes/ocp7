import array
from decimal import Rounded
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import CallbackRegistry
import numpy as np
import pandas as pd
import lightgbm as lgb
import re
import gc
import os
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier, record_evaluation
import cudf
import cuml
from cuml.pipeline import Pipeline
from cuml.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
)
from scipy import special
from IPython.display import display
import warnings
from copy import deepcopy

from src.p7_file import make_dir
from src.p7_regex import sel_var
from src.p7_util import format_time_min, format_time
from src.p7_metric import (
    cu_pred_prob_to_binary,
    pd_pred_prob_to_binary,
    penalize_f1,
    penalize_business_gain,
    cupd_recall_score,
)
from src.p7_preprocess import Imputer, VarianceSelector, CuRobustScaler, CuMinMaxScaler
from src.p7_preprocess import get_binary_features, balance_smote, balance_nearmiss

warnings.simplefilter(action="ignore", category=FutureWarning)

from src.p7_constantes import (
    NUM_THREADS,
    DATA_BASE,
    DATA_INTERIM,
    MODEL_DIR,
    # GENERATE_SUBMISSION_FILES,
    # STRATIFIED_KFOLD,
    VAL_SEED,
    # NUM_FOLDS,
    # EARLY_STOPPING,
)
from src.p7_simple_kernel import CONFIG_SIMPLE
from src.p7_util import timer


# see : https://github.com/jrzaurin/LightGBM-with-Focal-Loss/blob/d510432cdccd680c7fd3f3cbe780db69ccc297a4/utils/metrics.py
def lgb_f1_score(preds, lgbDataset):
    """
    Implementation of the f1 score to be used as evaluation score for lightgbm

    Parameters:
    -----------
    y_true: numpy.ndarray
            array with the true labels
    lgbDataset: lightgbm.Dataset
    """
    binary_preds = [int(p >= 0.5) for p in preds]
    y_true = lgbDataset.get_label()
    return "f1", f1_score(y_true, binary_preds), True


def lgb_recall_score(preds, lgbDataset):

    binary_preds = [int(p >= 0.5) for p in preds]
    y_true = lgbDataset.get_label()
    return "recall", recall_score(y_true, binary_preds), True


def lgb_weighted_recall_score(preds, lgbDataset):

    binary_preds = [int(p >= 0.5) for p in preds]
    y_true = lgbDataset.get_label()
    return (
        "weighted_recall",
        recall_score(y_true, binary_preds, average="weighted"),
        True,
    )


def lgb_precision_score(preds, lgbDataset):

    binary_preds = [int(p >= 0.5) for p in preds]
    y_true = lgbDataset.get_label()
    return "precision", precision_score(y_true, binary_preds, zero_division=0), True


def lgb_accuracy_score(preds, lgbDataset):

    binary_preds = [int(p >= 0.5) for p in preds]
    y_true = lgbDataset.get_label()
    return "accuracy", accuracy_score(y_true, binary_preds), True


def lgb_balanced_accuracy_score(preds, lgbDataset):

    binary_preds = [int(p >= 0.5) for p in preds]
    y_true = lgbDataset.get_label()
    return "balanced accuracy", balanced_accuracy_score(y_true, binary_preds), True


def lgb_cross_evaluate(
    train,
    params,
    early_stopping=True,
    verbose=True,
):

    predictors = [
        f
        for f in train.columns
        if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]

    X_train, y_train = (
        train[predictors],
        train["TARGET"],
    )

    categorical_features = list(X_train.loc[:, train.dtypes == "object"].columns.values)

    # Conversion en dataset lgbm
    train_set = lgb.Dataset(
        X_train,
        y_train,
        feature_name=predictors,
        categorical_feature=categorical_features,
        free_raw_data=True,
    )

    metrics_to_record = {}

    # Le early_stopping n'est pas pris en charge si dart
    if early_stopping:
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            # lgb.record_evaluation(metrics_to_record),
        ]
    else:
        callbacks = None

    metrics = [
        "binary_logloss",
        "auc",
    ]
    feval = [
        lgb_accuracy_score,
        lgb_balanced_accuracy_score,
        lgb_recall_score,
        lgb_weighted_recall_score,
        lgb_f1_score,
        lgb_precision_score,
    ]
    cv_results = lgb.cv(
        params,
        train_set,
        # num_boost_round=1_000,
        nfold=10,
        seed=VAL_SEED,
        shuffle=True,
        stratified=True,
        callbacks=callbacks,
        metrics=metrics,
        feature_name=predictors,
        categorical_feature=categorical_features,
        feval=feval,
    )
    """
    La stratification est réalisée par la target.
    On peut aussi stratifier avec sklearn puis utiliser les folds:
    # Diviser les données en k folds stratifiés
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Effectuer la validation croisée
    cv_results = lgb.cv(params, lgb_train, num_boost_round=1000, folds=skf)
    """

    if verbose:
        for k, v in cv_results.items():
            print(f"{k} : {v}")

    return cv_results


def pre_process_1_fold(
    X_train,
    y_train,
    X_val,
    pipe_preprocess=None,
    balance="none",
    k_neighbors=0,
    sampling_strategy=0.7,
):
    # Eventuel pipeline de pré-traitement
    if pipe_preprocess:
        pipe = deepcopy(pipe_preprocess)
        X_train_processed = pipe.fit_transform(X_train)
        X_val_processed = pipe.transform(X_val)
    else:
        X_train_processed = X_train
        X_val_processed = X_val

    # Eventuel rééquilibrage avec SMOTE ou NearMiss
    if balance == "none":
        X_train_balanced = X_train_processed
        y_train_balanced = y_train
    elif balance == "smote":
        X_train_balanced, y_train_balanced = balance_smote(
            X_train_processed,
            y_train,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=VAL_SEED,
            verbose=False,
        )
    elif balance == "nearmiss":
        X_train_balanced, y_train_balanced, features_null_var = balance_nearmiss(
            X_train_processed,
            y_train,
            k_neighbors=k_neighbors,
            minority_to_majority_ratio=sampling_strategy,
            verbose=False,
        )
    else:
        print(
            f"{balance} est une valeur incorrecte pour balance. Les valeurs possibles sont 'none', 'smote' ou 'nearmiss'"
        )
    return X_train_balanced, y_train_balanced, X_val_processed


# X_train etc. sont déjà pre-processes et resampled. Les features ne sont que des prédicteurs (pas de SK_ID_CURR)
def lgb_validate_1_fold(X_train, y_train, X_test, y_test, model_params, other_params):
    # On transforme en dataset lgb
    lgb_train = lgb.Dataset(
        X_train,
        y_train,
        params={"verbosity": -1},
        free_raw_data=False,
    )
    lgb_val = lgb.Dataset(
        X_test,
        y_test,
        reference=lgb_train,
        params={"verbosity": -1},
        free_raw_data=False,
    )
    t0_fit_time = time.time()
    model = lgb.train(
        params=model_params,
        train_set=lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=other_params["n_estimators"],
        feval=other_params["feval"],
        callbacks=other_params["callbacks"],
    )
    fit_time = time.time() - t0_fit_time

    # Si on utilise une fonc built-il pour la perte, le modèle renvoie déjà des probas,
    # Si on utilise les fonc perso, on a des logits et on doit appliquer sigmoid pour obtenir les probas,
    if other_params["loss"] == "binary":
        y_score_val = model.predict(X_test)
    else:
        y_score_val = special.expit(model.predict(X_test))

    # Prédiction de la classe en fonction du seuil de probabilité
    y_pred_val = pd_pred_prob_to_binary(
        y_score_val, threshold=other_params["threshold_prob"]
    )

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_val).ravel()

    # Mesures sur le jeu de validation
    dic_val_scores = {}
    dic_val_scores["auc"] = roc_auc_score(y_test, y_score_val)
    dic_val_scores["accuracy"] = accuracy_score(y_test, y_pred_val)
    dic_val_scores["recall"] = recall_score(y_test, y_pred_val)
    dic_val_scores["penalized_f1"] = penalize_f1(fp=fp, fn=fn, tp=tp)
    dic_val_scores["business_gain"] = penalize_business_gain(tn=tn, fp=fp, fn=fn, tp=tp)

    dic_val_scores["fit_time"] = fit_time
    dic_val_scores["tn"] = tn
    dic_val_scores["fn"] = fn
    dic_val_scores["tp"] = tp
    dic_val_scores["fp"] = fp
    return dic_val_scores


def train_and_test_lgbm(train, test, params, plot=True, verbose=True):

    predictors = [
        f
        for f in train.columns
        if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]

    X_train, y_train = (
        train[predictors],
        train["TARGET"],
    )
    X_test, y_test = (
        test[predictors],
        test["TARGET"],
    )

    categorical_features = list(X_train.loc[:, train.dtypes == "object"].columns.values)

    # Conversion en dataset lgbm
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=True)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=True)

    metrics = {}
    if verbose:
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            # lgb.log_evaluation(period=50),
            lgb.record_evaluation(metrics),
        ]
        print("Train shape :", train.shape)
        print("Eval shape :", test.shape)
        print("Nombre de prédicteurs :", len(predictors))
        print("Nombre de features catégorielles :", len(categorical_features))
        print("\nEntraînement...")
    else:
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.record_evaluation(metrics),
        ]
    t0_fit = time.time()
    lgb_clf = lgb.train(
        params,
        lgb_train,
        num_boost_round=400,  # Correspond à n_estimators max
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "eval"],
        feature_name=predictors,
        categorical_feature=categorical_features,
        callbacks=callbacks,
    )

    fit_duration = time.time() - t0_fit
    if verbose:
        print("Durée du fit (hh:mm:ss):", format_time(fit_duration))
        print("Prédictions et scoring...")

    # On calcule les probabilités d'appartenir à la classe d'indice 1, c'est à dire à la classe Default.
    # On remplace cette fonctionnalité de sklearn : y_pred = clf.predict_proba(test_x, num_iteration=clf.best_iteration_)[:, 1]
    # On calcule les scores bruts pour le jeu de test
    raw_scores = lgb_clf.predict(
        X_test, raw_score=True, num_iteration=lgb_clf.best_iteration
    )
    # Pour calculer les probas on applique la fonction sigmoïd aux scores bruts
    y_test_scores = 1 / (1 + np.exp(-raw_scores))

    # On calcule le auc_roc_score
    auc_score = roc_auc_score(y_test, y_score=y_test_scores)

    best_iteration = lgb_clf.best_iteration
    # print(f"best_iteration {best_iteration}")
    if plot:
        ax = lgb.plot_metric(metrics, metric="binary_logloss")
        plt.show()
        ax = lgb.plot_metric(metrics, metric="auc")
        plt.show()

    if plot:
        plot_roc_curve(y_test, y_score=y_test_scores)
        plot_precision_recall_curve(y_test, y_score=y_test_scores)

    # On convertit les probas en True False en fonction du seuil
    # [Question] On convertit en fonction d'un seuil perso ou on applique method .predict() qui va avoir un seuil à 0.5 ?
    # finalement on convertit en fonction d'un seuil
    threshold = 0.5
    y_pred = np.where(y_test_scores >= threshold, 1, 0)
    del raw_scores
    del y_test_scores
    gc.collect()

    # On calcule les scores qui réclament y_pred (et non y_score)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    weighted_recall = recall_score(y_test, y_pred, average="weighted")

    if plot:
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred)
        plot_recall(y_true=y_test, y_pred=y_pred)

    dic_score = {
        "auc": auc_score,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "weighted_recall": weighted_recall,
    }
    if verbose:
        print(classification_report(y_test, y_pred))
        for k, v in dic_score.items():
            print(f"{k} : {v:.2%}")
    return dic_score, fit_duration


def test_lgbm_sk(
    train, test, params, hyperparams, ceil_proba=0.5, config=CONFIG_SIMPLE
):

    # Passe les variables catégorielles en dtype category sinon LGM ne pourra pas les traiter
    cat_features = list(train.loc[:, train.dtypes == "object"].columns.values)
    for feature in cat_features:
        train[feature] = pd.Series(train[feature], dtype="category")
        test[feature] = pd.Series(test[feature], dtype="category")

    print("Train shape :", train.shape)

    predictors = [
        f
        for f in train.columns
        if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]

    train_x, train_y = (
        train[predictors],
        train["TARGET"],
    )
    test_x, test_y = (
        test[predictors],
        test["TARGET"],
    )

    clf = LGBMClassifier(**{**params, **hyperparams})
    metrics = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=config["early_stopping"]),
        lgb.log_evaluation(period=50),
        lgb.record_evaluation(metrics),
    ]

    print("Fit")
    t0_fit = time.time()
    clf.fit(
        train_x,
        train_y,
        eval_set=[(train_x, train_y), (test_x, test_y)],
        eval_metric={"auc"},
        # verbose=200,
        callbacks=callbacks,
    )
    fit_duration = time.time() - t0_fit
    print("Durée du fit (hh:mm:ss):", format_time_min(fit_duration))

    # On calcule les probabilités d'appartenir à la classe d'indice 1, c'est à dire à la classe Default.
    y_pred = clf.predict_proba(test_x, num_iteration=clf.best_iteration_)[:, 1]
    print("type y_pred", type(y_pred))

    print(f"best_iteration {clf.best_iteration_}")

    ax = lgb.plot_metric(metrics, metric="binary_logloss")
    plt.show()

    auc_score = roc_auc_score(test_y, y_score=y_pred)
    print(f"roc_auc_score : {auc_score:.4f}")
    plot_roc_curve(test_y, y_score=y_pred)
    plot_precision_recall_curve(test_y, y_score=y_pred)

    # On convertit les probas en True False en fonction du seuil
    return


def evaluate_lgbm(train_df, params, hyperparams, config=CONFIG_SIMPLE):
    folds = StratifiedKFold(
        n_splits=config["num_folds"],
        shuffle=True,
        random_state=VAL_SEED,
    )

    # Passe les variables catégorielles en dtype category sinon LGM ne pourra pas les traiter
    cat_features = list(train_df.loc[:, train_df.dtypes == "object"].columns.values)
    for feature in cat_features:
        train_df[feature] = pd.Series(train_df[feature], dtype="category")

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])

    predictors = [
        f
        for f in train_df.columns
        if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]

    for n_fold, (train_idx, valid_idx) in enumerate(
        folds.split(train_df[predictors], train_df["TARGET"])
    ):
        train_x, train_y = (
            train_df[predictors].iloc[train_idx],
            train_df["TARGET"].iloc[train_idx],
        )
        valid_x, valid_y = (
            train_df[predictors].iloc[valid_idx],
            train_df["TARGET"].iloc[valid_idx],
        )

        clf = LGBMClassifier(**{**params, **hyperparams})
        metrics = {}
        callbacks = [
            lgb.early_stopping(stopping_rounds=config["early_stopping"]),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(metrics),
        ]
        clf.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric={"auc"},
            # verbose=200,
            callbacks=callbacks,
        )

        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_
        )[:, 1]

        print(f"\nn_fold {n_fold + 1}, best_iteration {clf.best_iteration_}")

        auc_score = roc_auc_score(valid_y, oof_preds[valid_idx])
        print("Fold %2d AUC : %.6f" % (n_fold + 1, auc_score))
        print("callbacks")
        print(callbacks[2])

        lgb.plot_metric(metrics, metric="binary_logloss")

    """# On extrait les valeurs de la matrice de confusion
    conf_matrix = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = conf_matrix.ravel()

    dic_metrics = {"eval_acc": eval_acc, "roc_auc_score": auc_score}

    print(f"Auc Score: {auc_score:.3%}")
    print(f"Eval Accuracy: {eval_acc:.3%}")
    # Plot de la courbe ROC
    plot_roc_curve(
        sk_model,
        X_val,
        y_val,
        # name = 'ROC Curve (sklearn)',
    )
    plt.savefig(os.path.join(MODEL_DIR, "model_test_mlflow_roc_curve.png"))

    # Plot Matrice de confusion
    plot_confusion_matrix(y_val, preds)
    plt.savefig(os.path.join(MODEL_DIR, "model_test_mlflow_conf_matrix.png"))

    # Plot Matrice de confusion normalisée True
    plot_recall(y_val, preds)
    plt.savefig(os.path.join(MODEL_DIR, "model_test_mlflow_recall.png"))

    # Plot Matrice de confusion normalisée Pred
    plot_precision(y_val, preds)

    report = classification_report(y_val, preds)
    print(report)"""

    # Traçage mlflow
    """mlflow.log_artifact(os.path.join(MODEL_DIR, "model_test_mlflow_roc_curve.png"))
    mlflow.log_artifact(os.path.join(MODEL_DIR, "model_test_mlflow_conf_matrix.png"))
    mlflow.log_artifact(os.path.join(MODEL_DIR, "model_test_mlflow_recall.png"))"""
    # return dic_metrics
    return


# Vieux, remplacé par cuml_cross_evaluate
def logreg_cross_evaluate(
    train_and_val,
    pipe,
    hyperparams,
    threshold_prob=0.5,
    n_folds=5,
    random_state=VAL_SEED,
):
    """
    Tous les param de logreg desklearn:
    penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
    random_state=None, solver='lbfgs', max_iter=100, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None
    Solver : on va utiliser saga ou sag plus performant sur les larges datasets. saga permet plus de penalty : 'elasticnet', 'l1', 'l2', None, tandis que sag permet penalty : 'l2' et None
    """

    predictors = [f for f in train_and_val.columns if f not in ["TARGET", "SK_ID_CURR"]]

    X = train_and_val[["SK_ID_CURR", predictors[0]]].to_numpy()
    y = train_and_val["TARGET"].to_numpy()
    folds = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    # Create arrays and dataframes to store results
    # oof_preds = np.zeros(train_df.shape[0])

    metrics_mean = {}
    metrics_std = {}

    dic_train_scores = {
        "auc": [],
        "accuracy": [],
        "recall": [],
        "penalized_f1": [],
        "business_gain": [],
        "fit_time": [],
    }
    dic_val_scores = deepcopy(dic_train_scores)
    t0 = time.time()
    conf_mat = np.zeros((2, 2))

    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        t0_fit_time = time.time()

        X_train = train_and_val.loc[train_idx, ["SK_ID_CURR"] + predictors]
        y_train = train_and_val.loc[train_idx, "TARGET"].to_numpy()
        X_val = train_and_val.loc[valid_idx, ["SK_ID_CURR"] + predictors]
        y_val = train_and_val.loc[valid_idx, "TARGET"].to_numpy()
        print(
            f"Fold  {i_fold + 1}/{n_folds}, durée écoulée : {format_time(time.time() - t0)}"
        )
        # print(f"\tPrétraitement")
        X_train_processed = pipe.fit_transform(X_train)
        X_val_processed = pipe.transform(X_val)
        # print(f"\tFit model")
        clf = LogisticRegression(**hyperparams)
        clf.fit(X_train_processed, y_train)

        fit_time = time.time() - t0_fit_time

        # Probabilité d'appartenir à la classe default pour le jeu de validation
        y_score_val = clf.predict_proba(X_val_processed)[1].to_numpy()
        # Prédiction de la classe en fonction du seuil de probabilité
        y_pred_val = pd_pred_prob_to_binary(y_score_val, threshold=threshold_prob)

        # Mesures sur le jeu de validation
        dic_val_scores["auc"].append(roc_auc_score(y_val, y_score_val))
        dic_val_scores["accuracy"].append(accuracy_score(y_val, y_pred_val))
        dic_val_scores["recall"].append(recall_score(y_val, y_pred_val))
        dic_val_scores["penalized_f1"].append(penalize_f1(y_val, y_pred_val))
        dic_val_scores["business_gain"].append(
            penalize_business_gain(y_val, y_pred_val)
        )
        dic_val_scores["fit_time"].append(fit_time)

        # MAtrice de confusion moyenne sur le jeu de validation
        conf_mat += confusion_matrix(y_val, y_pred_val)

        # Probabilité d'appartenir à la classe default pour le jeu de train
        y_score_train = clf.predict_proba(X_train_processed)[1].to_numpy()
        # Prédiction de la classe en fonction du seuil de probabilité
        y_pred_train = pd_pred_prob_to_binary(y_score_train, threshold=threshold_prob)

        # Mesures sur le jeu de train
        dic_train_scores["auc"].append(roc_auc_score(y_train, y_score_train))
        dic_train_scores["accuracy"].append(accuracy_score(y_train, y_pred_train))
        dic_train_scores["recall"].append(recall_score(y_train, y_pred_train))
        dic_train_scores["penalized_f1"].append(penalize_f1(y_train, y_pred_train))
        dic_train_scores["business_gain"].append(
            penalize_business_gain(y_train, y_pred_train)
        )
        dic_train_scores["fit_time"].append(fit_time)

    train_res_folds = pd.DataFrame(
        dic_train_scores, index=np.arange(start=1, stop=n_folds + 1)
    )
    train_res_folds.index.name = "fold"
    train_res_means = train_res_folds.mean()
    train_res_std = train_res_folds.std()

    val_res_folds = pd.DataFrame(
        dic_val_scores, index=np.arange(start=1, stop=n_folds + 1)
    )
    val_res_folds.index.name = "fold"
    val_res_means = val_res_folds.mean()
    val_res_std = val_res_folds.std()

    results = pd.DataFrame(
        {
            "train_mean": train_res_means,
            "train_std": train_res_std,
            "val_mean": val_res_means,
            "val_std": val_res_std,
        }
    )
    print("\nScores globaux :")
    display(results)
    print()
    conf_mat = np.round(conf_mat / n_folds, decimals=0)
    plot_confusion_matrix_mean(conf_mat)

    """# On extrait les valeurs de la matrice de confusion
    conf_matrix = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = conf_matrix.ravel()

    dic_metrics = {"eval_acc": eval_acc, "roc_auc_score": auc_score}

    print(f"Auc Score: {auc_score:.3%}")
    print(f"Eval Accuracy: {eval_acc:.3%}")
    # Plot de la courbe ROC
    plot_roc_curve(
        sk_model,
        X_val,
        y_val,
        # name = 'ROC Curve (sklearn)',
    )
    plt.savefig(os.path.join(MODEL_DIR, "model_test_mlflow_roc_curve.png"))

    # Plot Matrice de confusion
    plot_confusion_matrix(y_val, preds)
    plt.savefig(os.path.join(MODEL_DIR, "model_test_mlflow_conf_matrix.png"))

    # Plot Matrice de confusion normalisée True
    plot_recall(y_val, preds)
    plt.savefig(os.path.join(MODEL_DIR, "model_test_mlflow_recall.png"))

    # Plot Matrice de confusion normalisée Pred
    plot_precision(y_val, preds)

    report = classification_report(y_val, preds)
    print(report)"""

    # Traçage mlflow
    """mlflow.log_artifact(os.path.join(MODEL_DIR, "model_test_mlflow_roc_curve.png"))
    mlflow.log_artifact(os.path.join(MODEL_DIR, "model_test_mlflow_conf_matrix.png"))
    mlflow.log_artifact(os.path.join(MODEL_DIR, "model_test_mlflow_recall.png"))"""
    # return dic_metrics
    return


# [TODO] Enlever SK_ID_CURR du modèle à fitter ?
def cuml_cross_validate(
    X_train_and_val,
    y_train_and_val,
    pipe_preprocess,
    cuml_model,
    threshold_prob=0.5,
    n_folds=5,
    random_state=42,
):

    predictors = [
        f for f in X_train_and_val.columns if f not in ["TARGET", "SK_ID_CURR"]
    ]
    # A faire avant le pipe
    # binary_features = get_binary_features(train_and_val)

    X_tmp = X_train_and_val[["SK_ID_CURR", predictors[0]]].to_numpy()
    y_tmp = y_train_and_val.to_pandas()

    folds = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    dic_val_scores = {
        "auc": [],
        "accuracy": [],
        "recall": [],
        "penalized_f1": [],
        "business_gain": [],
        "fit_time": [],
        "tn": [],
        "tp": [],
        "fn": [],
        "fp": [],
    }

    for train_idx_array, valid_idx_array in folds.split(X_tmp, y_tmp):
        train_idx = train_idx_array.tolist()
        valid_idx = valid_idx_array.tolist()
        t0_fit_time = time.time()
        X_train = X_train_and_val[predictors].iloc[train_idx]
        X_val = X_train_and_val[predictors].iloc[valid_idx]

        if y_train_and_val is None:
            y_train = X_train_and_val.loc[train_idx, "TARGET"]
            y_val = X_train_and_val.loc[valid_idx, "TARGET"]
        else:
            y_train = y_train_and_val.iloc[train_idx]
            y_val = y_train_and_val.iloc[valid_idx]

        # print(f"\tPrétraitement")
        pipe = deepcopy(pipe_preprocess)
        X_train_processed = pipe.fit_transform(X_train)
        X_val_processed = pipe.transform(X_val)

        # Afin de repartir d'un modèle non fitté, on crée une nouvelle copie du modèle passé en paramètre
        clf = deepcopy(cuml_model)
        clf.fit(X_train_processed, y_train)

        fit_time = time.time() - t0_fit_time

        # Probabilité d'appartenir à la classe default pour le jeu de validation
        y_score_val = clf.predict_proba(X_val_processed)[1]
        # Prédiction de la classe en fonction du seuil de probabilité
        y_pred_val = cu_pred_prob_to_binary(y_score_val, threshold=threshold_prob)

        # Order C applatit la matrice en ligne comme ravel() de numpy, mais
        # Attention, ne renvoie pas des scalaires mais des cupy ndarrays
        mat = cuml.metrics.confusion_matrix(y_val, y_pred_val)
        tn_array, fp_array, fn_array, tp_array = mat.ravel(order="C")
        # On extrait les scalaires des cupy ndarrays()
        tn = tn_array.item()
        fp = fp_array.item()
        fn = fn_array.item()
        tp = tp_array.item()
        dic_val_scores["tn"].append(tn)
        dic_val_scores["fn"].append(fn)
        dic_val_scores["tp"].append(tp)
        dic_val_scores["fp"].append(fp)

        # Mesures sur le jeu de validation
        dic_val_scores["auc"].append(cuml.metrics.roc_auc_score(y_val, y_score_val))
        dic_val_scores["accuracy"].append(
            cuml.metrics.accuracy_score(y_val, y_pred_val)
        )
        dic_val_scores["recall"].append(cupd_recall_score(tp=tp, fn=fn))
        dic_val_scores["penalized_f1"].append(penalize_f1(tp=tp, fp=fp, fn=fn))
        dic_val_scores["business_gain"].append(
            penalize_business_gain(tp=tp, fn=fn, tn=tn, fp=fp)
        )

        dic_val_scores["fit_time"].append(fit_time)

    del pipe
    del clf
    gc.collect()
    # print(dic_val_scores)
    return dic_val_scores


def cuml_cross_evaluate(
    X_train_and_val,
    pipe_preprocess,
    cuml_model,
    balance="none",
    k_neighbors=5,
    y_train_and_val=None,
    train_scores=False,
    threshold_prob=0.5,
    n_folds=5,
    random_state=42,
    verbose=False,
):

    predictors = [
        f
        for f in X_train_and_val.columns
        if f not in ["TARGET", "SK_ID_CURR", "Unnamed: 0"]
    ]
    # A faire avant le pipe
    # binary_features = get_binary_features(train_and_val)

    X_tmp = X_train_and_val[["SK_ID_CURR", predictors[0]]].to_numpy()
    if y_train_and_val is not None:
        y_tmp = y_train_and_val.to_pandas()
    else:
        y_tmp = X_train_and_val["TARGET"].to_pandas()

    folds = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    dic_val_scores = {
        "auc": [],
        "accuracy": [],
        "recall": [],
        "penalized_f1": [],
        "business_gain": [],
        "fit_time": [],
        "tn": [],
        "tp": [],
        "fn": [],
        "fp": [],
    }

    if train_scores:
        dic_train_scores = deepcopy(dic_val_scores)

    for train_idx_array, valid_idx_array in folds.split(X_tmp, y_tmp):
        train_idx = train_idx_array.tolist()
        valid_idx = valid_idx_array.tolist()
        t0_fit_time = time.time()
        X_train = X_train_and_val[predictors].iloc[train_idx]
        X_val = X_train_and_val[predictors].iloc[valid_idx]

        if y_train_and_val is None:
            y_train = X_train_and_val.loc[train_idx, "TARGET"]
            y_val = X_train_and_val.loc[valid_idx, "TARGET"]
        else:
            y_train = y_train_and_val.iloc[train_idx]
            y_val = y_train_and_val.iloc[valid_idx]

        # print(f"\tPrétraitement")
        pipe = deepcopy(pipe_preprocess)
        X_train_processed = pipe.fit_transform(X_train)
        X_val_processed = pipe.transform(X_val)

        # Eventuel rééquilibrage avec SMOTE ou NearMiss
        if balance == "none":
            X_train_balanced = X_train_processed
            y_train_balanced = y_train
        elif balance == "smote":
            X_train_balanced, y_train_balanced = balance_smote(
                X_train_processed,
                y_train,
                k_neighbors=k_neighbors,
                random_state=random_state,
                verbose=verbose,
            )
        elif balance == "nearmiss":
            X_train_balanced, y_train_balanced, features_null_var = balance_nearmiss(
                X_train_processed, y_train, k_neighbors=k_neighbors, verbose=verbose
            )
        else:
            print(
                f"{balance} est une valeur incorrecte pour balance. Les valeurs possibles sont 'none', 'smote' ou 'nearmiss'"
            )

        # Afin de repartir d'un modèle non fitté, on crée une nouvelle copie du modèle passé en paramètre
        clf = deepcopy(cuml_model)
        clf.fit(X_train_balanced, y_train_balanced)

        fit_time = time.time() - t0_fit_time

        ############### Prédictions et mesures sur le Jeu de TRAIN
        if train_scores:

            # Probabilité d'appartenir à la classe default pour le jeu de train
            y_score_train = clf.predict_proba(X_train_balanced)[1]
            # Prédiction de la classe en fonction du seuil de probabilité
            y_pred_train = cu_pred_prob_to_binary(
                y_score_train, threshold=threshold_prob
            )

            # Order C applatit la matrice en ligne comme ravel() de numpy, mais
            # Attention, ne renvoie pas des scalaires mais des cupy ndarrays
            mat = cuml.metrics.confusion_matrix(y_train_balanced, y_pred_train)
            tn_array, fp_array, fn_array, tp_array = mat.ravel(order="C")
            # On extrait les scalaires des cupy ndarrays()
            tn = tn_array.item()
            fp = fp_array.item()
            fn = fn_array.item()
            tp = tp_array.item()
            dic_train_scores["tn"].append(tn)
            dic_train_scores["fn"].append(fn)
            dic_train_scores["tp"].append(tp)
            dic_train_scores["fp"].append(fp)

            # Mesures sur le jeu de train
            dic_train_scores["auc"].append(
                cuml.metrics.roc_auc_score(y_train_balanced, y_score_train)
            )
            dic_train_scores["accuracy"].append(
                cuml.metrics.accuracy_score(y_train_balanced, y_pred_train)
            )
            dic_train_scores["recall"].append(cupd_recall_score(tp=tp, fn=fn))
            dic_train_scores["penalized_f1"].append(penalize_f1(tp=tp, fp=fp, fn=fn))
            dic_train_scores["business_gain"].append(
                penalize_business_gain(tp=tp, fn=fn, tn=tn, fp=fp)
            )

            dic_train_scores["fit_time"].append(fit_time)

        ############### Prédictions et mesures sur le Jeu de VALIDATION

        # Probabilité d'appartenir à la classe default pour le jeu de validation
        y_score_val = clf.predict_proba(X_val_processed)[1]
        # Prédiction de la classe en fonction du seuil de probabilité
        y_pred_val = cu_pred_prob_to_binary(y_score_val, threshold=threshold_prob)

        # Order C applatit la matrice en ligne comme ravel() de numpy, mais
        # Attention, ne renvoie pas des scalaires mais des cupy ndarrays
        mat = cuml.metrics.confusion_matrix(y_val, y_pred_val)
        tn_array, fp_array, fn_array, tp_array = mat.ravel(order="C")
        # On extrait les scalaires des cupy ndarrays()
        tn = tn_array.item()
        fp = fp_array.item()
        fn = fn_array.item()
        tp = tp_array.item()
        dic_val_scores["tn"].append(tn)
        dic_val_scores["fn"].append(fn)
        dic_val_scores["tp"].append(tp)
        dic_val_scores["fp"].append(fp)

        # Mesures sur le jeu de validation
        dic_val_scores["auc"].append(cuml.metrics.roc_auc_score(y_val, y_score_val))
        dic_val_scores["accuracy"].append(
            cuml.metrics.accuracy_score(y_val, y_pred_val)
        )
        dic_val_scores["recall"].append(cupd_recall_score(tp=tp, fn=fn))
        dic_val_scores["penalized_f1"].append(penalize_f1(tp=tp, fp=fp, fn=fn))
        dic_val_scores["business_gain"].append(
            penalize_business_gain(tp=tp, fn=fn, tn=tn, fp=fp)
        )

        dic_val_scores["fit_time"].append(fit_time)

    del pipe
    del clf
    gc.collect()
    # print(dic_val_scores)
    if train_scores:
        return dic_train_scores, dic_val_scores
    else:
        return dic_val_scores


def plot_roc_curve(y_true, y_score):
    # Taux de faux positifs et taux de faux négatifs
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve (sklearn)")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid(True)
    return


def plot_precision_recall_curve(y_true, y_score):
    # Calculer la précision et le rappel
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # Tracer la courbe de précision-rappel
    plt.figure()
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Courbe de Précision-Rappel")
    return


def plot_confusion_matrix(y_true, y_pred, figsize=(5, 4)):
    conf_mat = confusion_matrix(y_true, y_pred)
    conf_mat_df = pd.DataFrame(
        conf_mat,
        index=["Ok", "Défaut"],
        columns=["Ok", "Défaut"],
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        conf_mat_df,
        annot=True,
        fmt=".0f",
        vmin=0,
        linewidths=0.5,
        cmap="Blues",
        # cbar_kws={"format": self.format_percent},
        ax=ax,
    )
    fig.suptitle("Matrice de confusion")
    plt.yticks(rotation=0)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vraie catégorie")
    return fig


def plot_confusion_matrix_mean(tn, fp, fn, tp, figsize=(5, 4)):
    values = [tn, fp, fn, tp]
    rounded_values = np.round(values)
    conf_mat = rounded_values.reshape(2, 2)
    conf_mat_df = pd.DataFrame(
        conf_mat,
        index=["Ok", "Défaut"],
        columns=["Ok", "Défaut"],
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        conf_mat_df,
        annot=True,
        fmt=".0f",
        vmin=0,
        linewidths=0.5,
        cmap="Blues",
        # cbar_kws={"format": self.format_percent},
        ax=ax,
    )
    fig.suptitle("Matrice de confusion moyenne")
    plt.yticks(rotation=0)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vraie catégorie")
    return ax


def plot_recall_mean(
    tn,
    fp,
    fn,
    tp,
    title="Recall moyen (% par ligne)",
    subtitle="",
    figsize=(5, 5),
    n_samples=True,
    verbose=False,
):
    """Plotte le Recall (=la matrice de confusion en pourcentage par lignes) à partir des valeurs présentes dans la matrice de confusion binaire.
    Utilisée pour une cross_validation (matrice de confusion moyenne à travers les folds) ou si CUDA

    Args:
        tn (float): Nombre de vrais négatifs = Ok prédits en Ok
        fp (float): Nombre de faux positifs = vrai Ok prédits en Défaut de remboursement (pas trop grave)
        fn (float): Nombre de faux négatifs  = vrai Default prédits en Ok (Grave, oubli de prédire le default de remboursement)
        tp (float): Nombre de vrais positifs = défaut prédits en défaut
        title (str, optional): Titre de la figure. Defaults to "Recall moyen (% par ligne)".
        subtitle (str, optional): Sous_titre du graphique (ex : modèle concerné). Defaults to "".
        n_samples (bool, optional): Affichage oui/non du nombre d'observations concernées en dessous de chaque recall. Defaults to True.

    Returns:
        matplotlib.figure.Figure: figure à afficher dans le notebook ou à logguer dans mlflow
    """

    if subtitle:
        figsize = (5, 4.3)
    values = [tn, fp, fn, tp]
    rounded_values = np.round(values)
    conf_mat = rounded_values.reshape(2, 2)

    row_sums = conf_mat.sum(axis=1, keepdims=True)
    normalized_mat = conf_mat / row_sums

    # Si n_samples est vrai, on affiche le nombre d'observations en valeurs en dessous du recall en %
    if n_samples:
        # On crée la matrice des annotations avec les valeurs arrondies entre parenthèses
        annotations = np.array(
            [
                [
                    f"{normalized_mat[i, j]:.0%}\n(n={int(rounded_values[i * 2 + j])})"
                    for j in range(2)
                ]
                for i in range(2)
            ]
        )
        fmt = ""
    # Si n_samples faux, on affiche uniquement le recall en %
    else:
        annotations = True
        fmt = ".0%"

    conf_mat_df = pd.DataFrame(
        normalized_mat,
        index=["Ok", "Défaut"],
        columns=["Ok", "Défaut"],
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        conf_mat_df,
        annot=annotations,
        fmt=fmt,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cmap="Blues",
        cbar_kws={"format": format_percent},
        ax=ax,
    )
    fig.suptitle(title)

    if subtitle:
        # title = title + "\n" + subtitle
        ax.set_title(
            subtitle,
            ha="left",
            x=0,
            # Les 3 solutions suivantes pour la fontsize sont équivallentes si on n'a pas modifé le comportement par défaut
            # fontsize=plt.rcParams["font.size"],
            # fontsize="medium",
            fontsize=ax.xaxis.label.get_fontsize(),
        )
    else:
        fig.suptitle(title)
    plt.tight_layout()
    plt.yticks(rotation=0)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vraie catégorie")

    # Si on ne veut pas afficher la figure dans le notebook lors de l'appel de la fonc
    if not verbose:
        plt.close(fig)
    return fig


def plot_recall_mean_old(tn, fp, fn, tp, figsize=(5, 4)):
    values = [tn, fp, fn, tp]
    rounded_values = np.round(values)
    conf_mat = rounded_values.reshape(2, 2)

    row_sums = conf_mat.sum(axis=1, keepdims=True)
    normalized_mat = conf_mat / row_sums

    conf_mat_df = pd.DataFrame(
        normalized_mat,
        index=["Ok", "Défaut"],
        columns=["Ok", "Défaut"],
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        conf_mat_df,
        annot=True,
        fmt=".0%",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cmap="Blues",
        cbar_kws={"format": format_percent},
        ax=ax,
    )
    fig.suptitle("Recall moyen (% par ligne)")
    plt.yticks(rotation=0)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vraie catégorie")
    return


def plot_confusion_matrix_mean_old(conf_mat, figsize=(5, 4)):
    conf_mat_df = pd.DataFrame(
        conf_mat,
        index=["Ok", "Défaut"],
        columns=["Ok", "Défaut"],
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        conf_mat_df,
        annot=True,
        fmt=".0f",
        vmin=0,
        linewidths=0.5,
        cmap="Blues",
        # cbar_kws={"format": self.format_percent},
        ax=ax,
    )
    fig.suptitle("Matrice de confusion moyenne")
    plt.yticks(rotation=0)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vraie catégorie")
    return ax


def format_percent(x, _):
    return f"{x * 100:.0f}%"


def plot_precision(y_true, y_pred, figsize=(5, 4)):
    conf_mat = confusion_matrix(y_true, y_pred, normalize="pred")
    conf_mat_df = pd.DataFrame(
        conf_mat,
        index=["Ok", "Défaut"],
        columns=["Ok", "Défaut"],
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        conf_mat_df,
        annot=True,
        fmt=".0%",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cmap="Blues",
        cbar_kws={"format": format_percent},
        ax=ax,
    )
    fig.suptitle("Matrice de confusion en % par colonne (Précision)")
    plt.yticks(rotation=0)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vraie catégorie")
    return fig


def plot_recall(y_true, y_pred, figsize=(5, 4)):
    conf_mat = confusion_matrix(y_true, y_pred, normalize="true")
    conf_mat_df = pd.DataFrame(
        conf_mat,
        index=["Ok", "Défaut"],
        columns=["Ok", "Défaut"],
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        conf_mat_df,
        annot=True,
        fmt=".0%",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cmap="Blues",
        cbar_kws={"format": format_percent},
        ax=ax,
    )
    fig.suptitle("Matrice de confusion en % par ligne (Recall)")
    plt.yticks(rotation=0)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vraie catégorie")
    return fig


def set_vertical_margins(ax, top=0.15, bottom=0.05):
    """Ajoute des marges haute et basse inégales à un graphique

    Args:
        ax (Axes): Sous-plot matplotlib
        top (float, optional): Marge haute. Defaults to 0.15.
        bottom (float, optional): Marge basse. Defaults to 0.05.
    """
    ax.set_ymargin(0)
    ax.autoscale_view()
    lim = ax.get_ylim()
    delta = np.diff(lim)
    top = lim[0] - delta * top
    bottom = lim[1] + delta * bottom
    ax.set_ylim(top, bottom)


def plot_evaluation_scores(
    train_scores,
    val_scores,
    title="Scores de validation croisée\n",
    subtitle="",
    figsize=(8, 5),
    verbose=True,
):

    not_to_plot = ["fp", "fn", "tp", "tn", "fit_time"]
    metric_names = [k for k in train_scores.keys() if k not in not_to_plot]

    # Les dictionnaires en paramètres contiennent des listes de scores pour pour tous les folds,
    # On construit le dictionnaire contenant les scores moyens pour le train et la validation
    mean_scores = {
        "Train": [np.mean(v) for k, v in train_scores.items() if k not in not_to_plot],
        "Validation": [
            np.mean(v) for k, v in val_scores.items() if k not in not_to_plot
        ],
    }
    # On calcule les écart_types pour les afficher sous forme de barre d'erreur
    errors = {
        "Train": [np.std(v) for k, v in train_scores.items() if k not in not_to_plot],
        "Validation": [
            np.std(v) for k, v in val_scores.items() if k not in not_to_plot
        ],
    }

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)

    if subtitle:
        ax.set_title(
            subtitle,
            ha="left",
            x=0,
            fontsize=ax.xaxis.label.get_fontsize(),
        )

    # Graphique en barres horizontales
    y_pos = np.arange(len(metric_names))  # Cordonnée verticale des groupes de barres
    bar_height = 0.4  # Epaisseur des barres
    multiplier = 0

    for jdd, mean_score in mean_scores.items():
        offset = bar_height * multiplier
        rects = ax.barh(
            y=y_pos + offset,
            width=mean_score,
            height=-bar_height,
            label=jdd,
            align="edge",
            xerr=errors[jdd],
        )
        ax.bar_label(rects, fmt="%0.3f", padding=5)
        multiplier += 1

    ax.set_xlabel("Score")
    ax.set_ylabel("Métrique")
    ax.set_xlim(0, 1)
    ax.set_yticks(y_pos, metric_names)
    ax.set_xlim(0, 1.05)
    # On place une marge en haut et on affiche la légende au dessus des barres
    set_vertical_margins(ax=ax, top=0.15, bottom=0.05)
    ax.invert_yaxis()
    plt.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    if not verbose:
        plt.close()
    return fig


# En définitive, avec les boxplots on ne voit pas assez bien car les folds sont similaires. Préférer les barres
def plot_evaluation_scores_box(
    train_scores,
    val_scores,
    baseline_scores=None,
    vline=None,
    title="Scores de validation croisée\n",
    subtitle="",
    figsize=None,
):

    # Transforme les dic en df et les regroupe en un seul avec une colonne indiquant la provenance
    df_train_scores = pd.DataFrame(train_scores)
    df_train_scores["jdd"] = "Train"
    df_val_scores = pd.DataFrame(val_scores)
    df_val_scores["jdd"] = "Validation"
    df_scores = pd.concat([df_train_scores, df_val_scores])

    if baseline_scores:
        df_baseline_scores = pd.DataFrame(baseline_scores)
        df_baseline_scores["jdd"] = "Baseline validation"
        df_scores = pd.concat([df_scores, df_baseline_scores])

    # On calcule les ratios des faux positifs et faux négatifs (par rapport à la totalité) pour les afficher entre 0 et 1
    df_scores["fn_ratio"] = df_scores["fn"] / (
        df_scores["tp"] + df_scores["tn"] + df_scores["fp"] + df_scores["fn"]
    )
    df_scores["fp_ratio"] = df_scores["fp"] / (
        df_scores["tp"] + df_scores["tn"] + df_scores["fp"] + df_scores["fn"]
    )
    # On drop les scores qui ne sont pas à plotter (ils ne sont pas entre 0 et 1)
    scores_to_drop = ["fp", "fn", "tp", "tn", "fit_time"]
    df_scores = df_scores.drop(scores_to_drop, axis=1)

    # On transforme le DataFrame en un format long où chaque ligne représente une observation unique de score.
    df_melt = pd.melt(df_scores, id_vars=["jdd"], var_name="score", value_name="value")

    if not figsize:
        if baseline_scores:
            figsize = (8, 8)
        else:
            figsize = (8, 6)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)

    sns.boxplot(
        data=df_melt, x="value", y="score", orient="h", hue="jdd", legend="auto"
    )
    if subtitle:
        ax.set_title(
            subtitle,
            ha="left",
            x=0,
            fontsize=ax.xaxis.label.get_fontsize(),
        )
    if vline:
        ax.axvline(x=vline, color="k", linestyle="--", label="baseline")
    ax.set_xlabel("Score")
    ax.set_ylabel("Métrique")
    ax.set_xlim(0, 1)

    ax.legend()
    fig.tight_layout()
    return fig


class CuDummyClassifier:
    def __init__(self, random_state=VAL_SEED):
        self.random_state = random_state
        self.y_scores_default = None
        self.random_state = random_state

    def fit(self, X, y=None):
        pass

    def predict_proba(self, X_val, y_val=None):
        # On fixe la graine de hasard si random_state n'est pas None
        if self.random_state:
            np.random.seed(self.random_state)

        # On génère aléatoirement des probas pour la classe default
        y_scores_default = np.random.rand(X_val.shape[0])
        y_scores = cudf.DataFrame({1: y_scores_default})
        # y_scores["0"] = 1 - y_scores["1"]
        return y_scores

    """def predict(self, threshold_prob=0.5):
        # Proba d'appartenir à la classe défaut
        y_pred = self.predict_proba()[1]
        # Classe de proba 0 ou 1
        y_pred = (y_pred < threshold_prob).astype(int)
        return y_pred"""
