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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from src.p7_file import make_dir
from src.p7_regex import sel_var
from src.p7_util import format_time_min, format_time

warnings.simplefilter(action="ignore", category=FutureWarning)

from src.p7_constantes import (
    NUM_THREADS,
    DATA_BASE,
    DATA_INTERIM,
    MODEL_DIR,
    # GENERATE_SUBMISSION_FILES,
    # STRATIFIED_KFOLD,
    # RANDOM_SEED,
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
        seed=42,
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
        random_state=config["random_seed"],
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
