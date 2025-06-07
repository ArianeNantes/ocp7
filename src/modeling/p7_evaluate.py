import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import gc
import cupy as cp
import time
import cuml
from cuml.linear_model import LogisticRegression
from cuml.model_selection import StratifiedKFold as CuStratifiedKFold
from sklearn.model_selection import StratifiedKFold as SkStratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as sk_auc

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y as sk_valid_check_X_y
from sklearn.utils.validation import check_array as sk_valid_check_array

from sklearn.metrics import (
    roc_auc_score,
    # confusion_matrix,
    precision_recall_curve,
)

import warnings
from copy import deepcopy

from src.modeling.p7_util import format_time
from src.modeling.p7_metric import business_gain_score


warnings.simplefilter(action="ignore", category=FutureWarning)

from src.modeling.p7_constantes import VAL_SEED


# Classe parente pour l'évaluation simple ou en validation croisée
# [TODO] si le temps, faire la classe parente et faire dériver EvaluatorCV et Evaluator de cette classe parente
"""
class Eval:
    def __init__(self):
        return self

    @staticmethod
    def _init_scores(self):
        return {
            "auc": [],
            "accuracy": [],
            "recall": [],
            "f1_score": [],
            "business_gain": [],
            "fit_time": [],
            "tn": [],
            "tp": [],
            "fn": [],
            "fp": [],
        }

    # Calcule les scores sur un le jeu de train ou de validation
    def _score_set(
        self, fitted_pipe, X_subset, y_true_subset, dic_scores, store_oof=False
    ):
        dic_scores["fit_time"] = self.fit_time
        if self.device in ["GPU", "CUDA"]:
            y_prob = fitted_pipe.predict_proba(X_subset)[1]
            y_prob = y_prob.to_numpy()
        else:
            y_prob = fitted_pipe.predict_proba(X_subset)[:, 1]

        y_true = y_true_subset.to_numpy()
        if store_oof:
            self.oof_probs.append(y_prob)
            self.oof_trues.append(y_true)

        y_pred = (y_prob > self.threshold_prob).astype(np.int32)

        tn, fp, fn, tp = self._compute_confusion(y_true, y_pred)

        # Ajout direct
        dic_scores["tn"].append(tn)
        dic_scores["fp"].append(fp)
        dic_scores["fn"].append(fn)
        dic_scores["tp"].append(tp)

        # À partir de TP, FP, FN, TN → calculs rapides
        dic_scores["accuracy"].append((tp + tn) / (tp + tn + fp + fn))
        dic_scores["recall"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        dic_scores["f1_score"].append(
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        dic_scores["business_gain"].append(
            business_gain_score(
                tn=tn,
                fp=fp,
                fn=fn,
                tp=tp,
                loss_fn=self.param_bg["loss_fn"],
                loss_fp=self.param_bg["loss_fp"],
                gain_tp=self.param_bg["gain_tp"],
                gain_tn=self.param_bg["gain_tn"],
            )
        )
        # dic_scores["fit_time"].append(fit_time)

        # ROC AUC nécessite encore y_prob (et pas binaire)
        dic_scores["auc"].append(roc_auc_score(y_true, y_prob))

        del X_subset, y_prob, y_pred
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        return
    """


class EvaluatorCV:
    def __init__(
        self,
        pipe,
        cv=5,
        score_train_set=True,
        param_bg={
            "loss_fn": -10,
            "loss_fp": -1,
            "gain_tp": 1,
            "gain_tn": 1,
        },
        threshold_prob=0.5,
        device="GPU",
        random_state=42,
        verbose=True,
    ):
        self.pipe = deepcopy(pipe)
        self.cv = cv
        self.score_train_set = score_train_set
        self.param_bg = param_bg
        self.threshold_prob = threshold_prob
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.train_scores = None
        self.val_scores = None
        self.mean_val_scores = None
        # On retient pour tous les folds, sur le jeu de validation, les scores de probabilité d'appartenir à une classe sur et
        # Les vraies targets. Cela pourra être utile après coup. Par exemple, tracer une roc_auc curve.
        self.oof_probs = []
        self.oof_trues = []
        # self.start_time_ = None
        # self.end_time_ = None

        # Si l'argument est de type pipeline, on récupère le modèle dans le dernier step,
        # sinon on considère que c'est le modèle lui-même
        if "pipeline" in str(type(self.pipe)):
            self.model = self.pipe.steps[-1][1]
        else:
            self.model = self.pipe
        self.n_rows = None
        self.n_features = None
        # Balancing
        self.balance_str = "Unbalanced"
        model_params = self.model.get_params()
        if "class_weight" in model_params.keys():
            class_weight_str = str(model_params["class_weight"])
            if class_weight_str != "None":
                self.balance_str = f"class_weight={class_weight_str}"
        try:
            named_steps = self.pipe.named_steps
            named_keys = [k.upper() for k in named_steps.keys()]
            if "SMOTE" in named_keys:
                self.balance_str = "SMOTE"
        except:
            pass
        self.subtitle = ""

    def evaluate(
        self,
        X,
        y,
    ):
        self.start_time_ = time.time()
        self.val_scores = self._init_scores()
        self.fit_time = 0
        # Utilisés pour les tableaux récapitulatifs et sous-titres
        self.n_features = X.shape[1]
        self.n_rows = X.shape[0]
        self._init_balance_str()

        if self.score_train_set:
            self.train_scores = deepcopy(self.val_scores)

        if self.device in ["GPU", "CUDA"]:
            folds = CuStratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
        else:
            folds = SkStratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )

        for fold_num, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            # print(f"Fold {fold_num + 1}/{self.cv}")

            cp.random.seed(self.random_state)
            np.random.seed(self.random_state)

            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]

            # La copie du modèle de départ est obligatoire pour tenter de fixer les graines GPU
            pipe_copy = deepcopy(self.pipe)
            t0 = time.time()
            pipe_copy.fit(X_train, y_train)
            self.fit_time = time.time() - t0

            # Calcul et mise à jour des scores sur le jeu de train, si cela est demandé dans le constructeur
            if self.score_train_set:
                self._score_set(
                    fitted_pipe=pipe_copy,
                    X_subset=X_train,
                    y_true_subset=y_train,
                    dic_scores=self.train_scores,
                    store_oof=False,
                )
                self.mean_train_scores = {
                    k: np.mean(v) for (k, v) in self.train_scores.items()
                }

            # Calcul et mise à jour des scores sur le jeu de validation
            self._score_set(
                fitted_pipe=pipe_copy,
                X_subset=X_val,
                y_true_subset=y_val,
                dic_scores=self.val_scores,
                store_oof=True,
            )
        self.end_time_ = time.time()
        self.mean_val_scores = {k: np.mean(v) for (k, v) in self.val_scores.items()}
        if self.verbose:
            duration = self.end_time_ - self.start_time_
            print(
                f"Durée de l'évaluation en validation croisée sur {self.device} ({self.cv} folds): {format_time((duration))}"
            )
        # return self

    def _init_scores(self):
        return {
            "auc": [],
            "accuracy": [],
            "recall": [],
            "f1_score": [],
            "business_gain": [],
            "fit_time": [],
            "tn": [],
            "tp": [],
            "fn": [],
            "fp": [],
        }

    # Calcule les scores sur un le jeu de train ou de validation
    def _score_set(
        self, fitted_pipe, X_subset, y_true_subset, dic_scores, store_oof=False
    ):
        dic_scores["fit_time"] = self.fit_time
        if self.device in ["GPU", "CUDA"]:
            y_prob = fitted_pipe.predict_proba(X_subset)[1]
            y_prob = y_prob.to_numpy()
        else:
            y_prob = fitted_pipe.predict_proba(X_subset)[:, 1]

        y_true = y_true_subset.to_numpy()
        if store_oof:
            self.oof_probs.append(y_prob)
            self.oof_trues.append(y_true)

        y_pred = (y_prob > self.threshold_prob).astype(np.int32)

        tn, fp, fn, tp = self._compute_confusion(y_true, y_pred)

        # Ajout direct
        dic_scores["tn"].append(tn)
        dic_scores["fp"].append(fp)
        dic_scores["fn"].append(fn)
        dic_scores["tp"].append(tp)

        # À partir de TP, FP, FN, TN → calculs rapides
        dic_scores["accuracy"].append((tp + tn) / (tp + tn + fp + fn))
        dic_scores["recall"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        dic_scores["f1_score"].append(
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        dic_scores["business_gain"].append(
            business_gain_score(
                tn=tn,
                fp=fp,
                fn=fn,
                tp=tp,
                loss_fn=self.param_bg["loss_fn"],
                loss_fp=self.param_bg["loss_fp"],
                gain_tp=self.param_bg["gain_tp"],
                gain_tn=self.param_bg["gain_tn"],
            )
        )
        # dic_scores["fit_time"].append(fit_time)

        # ROC AUC nécessite encore y_prob (et pas binaire)
        dic_scores["auc"].append(roc_auc_score(y_true, y_prob))

        del X_subset, y_prob, y_pred
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

        return

    @staticmethod
    def _compute_confusion(y_true, y_pred):
        # Calcul rapide
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tn, fp, fn, tp

    def get_mean_train_scores(self, prefix=""):  # ex: prefix="train_"
        if self.train_scores:
            return {f"{prefix}{k}": np.mean(v) for (k, v) in self.train_scores.items()}

        else:
            param_score_train_set_str = ""
            if not self.score_train_set:
                param_score_train_set_str = (
                    f" - Le paramètre 'score_train_set' est {self.score_train_set}"
                )
            print(
                f"Les scores sur le jeu de train n'ont pas été calculés{param_score_train_set_str}"
            )
            return

    def get_all_train_scores(self):
        if self.train_scores:
            return self.train_scores
        else:
            param_score_train_set_str = ""
            if not self.score_train_set:
                param_score_train_set_str = (
                    f" - Le paramètre 'score_train_set' est {self.score_train_set}"
                )
            print(
                f"Les scores sur le jeu de train n'ont pas été calculés{param_score_train_set_str}"
            )
            return

    def get_mean_val_scores(self, prefix=""):
        if self.val_scores:
            return {f"{prefix}{k}": np.mean(v) for (k, v) in self.val_scores.items()}
        else:
            print(
                "Les scores de validation n'ont pas été calculés. Appeler la fonc .evaluate"
            )

    def get_all_val_scores(self):
        if self.val_scores:
            return self.val_scores
        else:
            print(
                "Les scores de validation n'ont pas été calculés. Appeler la fonc .evaluate"
            )

    def get_business_gain(self, verbose=True):
        if self.train_scores and self.val_scores:
            if self.verbose:
                print(
                    f"Business gain Train : {self.mean_train_scores['business_gain']:.4f},\tBusiness gain Validation : {self.mean_val_scores['business_gain']:.4f}"
                )
            return (
                self.mean_train_scores["business_gain"],
                self.mean_val_scores["business_gain"],
            )
        elif self.val_scores:
            print(
                f"Business gain Validation : {self.mean_val_scores['business_gain']:.4f}"
            )
            return self.mean_val_scores["business_gain"]
        else:
            print("Les Business_gain n'ont pas été calculés")
            return

    def get_model_params(self):
        return self.model.get_params()

    def get_bg_params(self):
        return self.param_bg

    def get_metric_params(self):
        params = self.param_bg
        params["threshold_prob"] = self.threshold_prob
        return params

    def get_all_params(self):
        dic_threshold = {"trheshold_prob": self.threshold_prob}
        return {
            **self.get_model_params(),
            **self.get_bg_params(),
            **dic_threshold,
        }

    def _init_balance_str(self):
        # Balancing
        self.balance_str = "Unbalanced"
        model_params = self.model.get_params()
        if "class_weight" in model_params.keys():
            class_weight_str = str(model_params["class_weight"])
            if class_weight_str != "None":
                self.balance_str = f"class_weight={class_weight_str}"
        try:
            named_steps = self.pipe.named_steps
            named_keys = [k.upper() for k in named_steps.keys()]
            if "SMOTE" in named_keys:
                self.balance_str = "SMOTE"
        except:
            pass
        return

    def _init_subtitle(self, subtitle_type="balancing"):
        subtitle = self.model.__class__.__name__
        if subtitle_type == "balancing":
            if not self.balance_str:
                self._init_balance_str()
            subtitle = subtitle + f" - {self.balance_str}"
            subtitle = subtitle + f"\nShape X : {self.n_rows}, {self.n_features}"
            self.subtitle = subtitle
        return

    def plot_evaluation_scores(
        self, title=None, subtitle=None, subtitle_type="balancing"
    ):
        if not title:
            title = "Scores moyens - Entraînement VS Validation"
        if not subtitle:
            self._init_subtitle(subtitle_type)
            subtitle = self.subtitle

        fig = plot_evaluation_scores(
            train_scores=self.train_scores,
            val_scores=self.val_scores,
            title=title,
            subtitle=subtitle,
            verbose=True,
        )
        return fig

    def plot_conf_mat(self, title=None, subtitle=None, subtitle_type="balancing"):
        if not title:
            title = f"Mat. de Confusion moyenne (% par ligne)"
        if not subtitle:
            self._init_subtitle(subtitle_type)
            subtitle = self.subtitle
        fig = plot_recall_mean(
            tn=self.mean_val_scores["tn"],
            fp=self.mean_val_scores["fp"],
            fn=self.mean_val_scores["fn"],
            tp=self.mean_val_scores["tp"],
            title=title,
            subtitle=subtitle,
        )
        return fig

    def plot_roc_curve(self, title=None, subtitle=None, subtitle_type="balancing"):
        if not subtitle:
            self._init_subtitle(subtitle_type)
            subtitle = self.subtitle
        if not title:
            title = "Courbe ROC moyenne"

        figsize = (6, 5)
        # On vérifie que les liste de y_probs et les listes de y_trues ont la bonne longueur
        assert (
            len(self.oof_trues) == len(self.oof_probs)
            and len(self.oof_trues) == self.cv
        )

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title)

        if subtitle:
            ax.set_title(
                subtitle,
                ha="left",
                x=0,
                fontsize=ax.xaxis.label.get_fontsize(),
            )

        for i in range(self.cv):
            fpr, tpr, _ = roc_curve(self.oof_trues[i], self.oof_probs[i])
            roc_auc = sk_auc(fpr, tpr)
            aucs.append(roc_auc)

            # Interpolation du TPR pour un FPR commun
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # commence bien en (0,0)
            tprs.append(interp_tpr)
            # plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {i+1} ROC AUC = {roc_auc:.2f}")

        # Calcul de la moyenne et de l'écart-type
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = sk_auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            # label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            label=f"ROC Moyenne (mean AUC = {mean_auc: 0.4f})",
            lw=2,
            alpha=0.8,
        )

        # Bande de ±1 std autour de la moyenne
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tpr_lower,
            tpr_upper,
            color="blue",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.plot([0, 1], [0, 1], linestyle="--", color="r", lw=1)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("Taux de Faux Positifs")
        plt.ylabel("Traux de Vrais Positifs")
        plt.legend(loc="lower right")
        plt.grid(False)
        plt.tight_layout()
        # plt.show()
        return fig


class Evaluator:
    def __init__(
        self,
        pipe,
        score_train_set=True,
        threshold_prob=0.5,
        param_bg={
            "loss_fn": -10,
            "loss_fp": -1,
            "gain_tp": 1,
            "gain_tn": 1,
        },
        random_state=VAL_SEED,
        verbose=True,
    ):
        self.pipe = deepcopy(pipe)
        self.score_train_set = score_train_set
        self.threshold_prob = threshold_prob
        self.param_bg = param_bg
        self.random_state = random_state
        self.verbose = verbose
        self.train_scores = {}
        self.test_scores = {}
        self.y_test_probs = None
        self.y_test_preds = None
        self.n_features = None
        self.n_rows = None
        # Si l'argument est de type pipeline, on récupère le modèle dans le dernier step,
        # sinon on considère que c'est le modèle lui-même
        if "pipeline" in str(type(self.pipe)):
            self.model = self.pipe.steps[-1][1]
        else:
            self.model = self.pipe
        self.n_rows = None
        self.n_features = None
        # Balancing
        self.balance_str = "Unbalanced"
        model_params = self.model.get_params()
        if "class_weight" in model_params.keys():
            class_weight_str = str(model_params["class_weight"])
            if class_weight_str != "None":
                self.balance_str = f"class_weight={class_weight_str}"
        try:
            named_steps = self.pipe.named_steps
            named_keys = [k.upper() for k in named_steps.keys()]
            if "SMOTE" in named_keys:
                self.balance_str = "SMOTE"
        except:
            pass
        self.subtitle = ""

    def evaluate(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        self.start_time_ = time.time()
        self.val_scores = self._init_scores()
        self.fit_time = 0
        # Utilisés pour les tableaux récapitulatifs et sous-titres
        self.n_features = X_train.shape[1]
        self.n_rows = X_train.shape[0]
        self._init_balance_str()

        if self.score_train_set:
            self.train_scores = deepcopy(self.val_scores)

        t0 = time.time()
        self.pipe.fit(X_train, y_train)
        self.fit_time = time.time() - t0

        # On met à jour le modèle après fit
        if "pipeline" in str(type(self.pipe)):
            self.model = self.pipe.steps[-1][1]
        else:
            self.model = self.pipe

        # Calcul et mise à jour des scores sur le jeu de train, si cela est demandé dans le constructeur
        if self.score_train_set:
            _ = self._score_set(
                fitted_pipe=self.pipe,
                X_subset=X_train,
                y_true_subset=y_train,
                dic_scores=self.train_scores,
                store_oof=False,
            )

        # Calcul et mise à jour des scores sur le jeu de test
        self.y_test_probs, self.y_test_preds = self._score_set(
            fitted_pipe=self.pipe,
            X_subset=X_test,
            y_true_subset=y_test,
            dic_scores=self.test_scores,
            store_oof=True,
        )
        self.y_test_true = y_test
        self.end_time_ = time.time()

        if self.verbose:
            duration = self.end_time_ - self.start_time_
            print(f"Durée de l'évaluation : {format_time((duration))}")
        # return self

    def _init_scores(self):
        return {
            "auc": 0,
            "accuracy": 0,
            "recall": 0,
            "f1_score": 0,
            "business_gain": 0,
            "fit_time": 0,
            "tn": 0,
            "tp": 0,
            "fn": 0,
            "fp": 0,
        }

    # Calcule les scores sur un le jeu de train ou de validation
    def _score_set(
        self, fitted_pipe, X_subset, y_true_subset, dic_scores, store_oof=False
    ):
        dic_scores["fit_time"] = self.fit_time
        y_prob = fitted_pipe.predict_proba(X_subset)[:, 1]
        y_true = y_true_subset.to_numpy()
        """if store_oof:
            self.oof_probs.append(y_prob)
            self.oof_trues.append(y_true)"""
        y_pred = (y_prob > self.threshold_prob).astype(np.int32)

        tn, fp, fn, tp = self._compute_confusion(y_true, y_pred)

        # Ajout direct
        dic_scores["tn"] = tn
        dic_scores["fp"] = fp
        dic_scores["fn"] = fn
        dic_scores["tp"] = tp

        # À partir de TP, FP, FN, TN → calculs rapides
        dic_scores["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        dic_scores["recall"] = recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        dic_scores["f1_score"] = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        dic_scores["business_gain"] = business_gain_score(
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            loss_fn=self.param_bg["loss_fn"],
            loss_fp=self.param_bg["loss_fp"],
            gain_tp=self.param_bg["gain_tp"],
            gain_tn=self.param_bg["gain_tn"],
        )

        # ROC AUC nécessite encore y_prob (et pas binaire)
        dic_scores["auc"] = roc_auc_score(y_true, y_prob)

        del X_subset
        gc.collect()
        return y_prob, y_pred

    @staticmethod
    def _compute_confusion(y_true, y_pred):
        # Calcul rapide
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tn, fp, fn, tp

    def get_train_scores(self, prefix=""):
        if self.train_scores:
            return {f"{prefix}{k}": v for (k, v) in self.train_scores.items()}
        else:
            param_score_train_set_str = ""
            if not self.score_train_set:
                param_score_train_set_str = (
                    f" - Le paramètre 'score_train_set' est {self.score_train_set}"
                )
            print(
                f"Les scores sur le jeu de train n'ont pas été calculés{param_score_train_set_str}"
            )
            return

    def get_test_scores(self, prefix=""):
        if self.test_scores:
            return {f"{prefix}{k}": v for (k, v) in self.test_scores.items()}
        else:
            print(
                "Les scores de validation n'ont pas été calculés. Appeler la fonc .evaluate"
            )

    def get_business_gain(self, verbose=True):
        if self.train_scores and self.test_scores:
            if self.verbose:
                print(
                    f"Business gain Train : {self.get_train_scores()['business_gain']:.4f},\tBusiness gain Test : {self.get_test_scores()['business_gain']:.4f}"
                )
            return (
                self.get_train_scores()["business_gain"],
                self.get_test_scores()["business_gain"],
            )
        elif self.val_scores:
            print(f"Business gain Test : {self.get_test_scores()['business_gain']:.4f}")
            return self.get_test_scores()["business_gain"]
        else:
            print("Les Business_gain n'ont pas été calculés")
            return

    def get_model_params(self):
        return self.model.get_params()

    def get_bg_params(self):
        return self.param_bg

    def get_all_params(self):
        dic_threshold = {"trheshold_prob": self.threshold_prob}
        return {
            **self.get_model_params(),
            **self.get_bg_params(),
            **dic_threshold,
        }

    def get_metric_params(self):
        params = self.param_bg
        params["threshold_prob"] = self.threshold_prob
        return params

    def _init_balance_str(self):
        # Balancing
        self.balance_str = "Unbalanced"
        model_params = self.model.get_params()
        if "class_weight" in model_params.keys():
            class_weight_str = str(model_params["class_weight"])
            if class_weight_str != "None":
                self.balance_str = f"class_weight={class_weight_str}"
        try:
            named_steps = self.pipe.named_steps
            named_keys = [k.upper() for k in named_steps.keys()]
            if "SMOTE" in named_keys:
                self.balance_str = "SMOTE"
        except:
            pass
        return

    def _init_subtitle(self, subtitle_type="balancing"):
        subtitle = self.model.__class__.__name__
        if subtitle_type == "balancing":
            if not self.balance_str:
                self._init_balance_str()
            subtitle += f" - {self._init_balance_str()}"
            subtitle += f"\nShape X : {self.n_rows}, {self.n_features}"
            self.subtitle = subtitle
        return

    def plot_evaluation_scores(
        self, title=None, subtitle=None, subtitle_type="balancing"
    ):
        figsize = (8, 5)
        if not title:
            title = "Scores - Entraînement VS Test"
        if not subtitle:
            self._init_subtitle(subtitle_type)
            subtitle = self.subtitle

        # Scores des dictionnaires de scores qu'on ne veut pas faire figurer dans le graphique
        not_to_plot = ["fp", "fn", "tp", "tn", "fit_time"]
        metric_names = [k for k in self.test_scores.keys() if k not in not_to_plot]
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
        y_pos = np.arange(
            len(metric_names)
        )  # Cordonnée verticale des groupes de barres
        bar_height = 0.4  # Epaisseur des barres
        multiplier = 0

        scores = {
            "Train": [self.train_scores[k] for k in metric_names],
            "Validation": [self.test_scores[k] for k in metric_names],
        }
        for jdd, score in scores.items():
            offset = bar_height * multiplier
            rects = ax.barh(
                y=y_pos + offset,
                width=score,
                height=-bar_height,
                label=jdd,
                align="edge",
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

        return fig

    def plot_conf_mat(self, title=None, subtitle=None, subtitle_type="balancing"):
        if not title:
            title = f"Matrice de Confusion (% par ligne)"
        if not subtitle:
            self._init_subtitle(subtitle_type)
            subtitle = self.subtitle
        fig = plot_recall_mean(
            tn=self.test_scores["tn"],
            fp=self.test_scores["fp"],
            fn=self.test_scores["fn"],
            tp=self.test_scores["tp"],
            title=title,
            subtitle=subtitle,
        )
        return fig

    def plot_roc_curve(self, title=None, subtitle=None, subtitle_type="balancing"):
        if not title:
            title = "Courbe ROC"
        if not subtitle:
            self._init_subtitle(subtitle_type)
            subtitle = self.subtitle
        figsize = (6, 5)

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title)

        if subtitle:
            ax.set_title(
                subtitle,
                ha="left",
                x=0,
                fontsize=ax.xaxis.label.get_fontsize(),
            )

        fpr, tpr, thresholds = roc_curve(self.y_test_true, self.y_test_probs)

        best_threshold = tpr - fpr
        idx_best = np.argmax(best_threshold)
        best_threshold = thresholds[idx_best]
        print("best_threshold :", best_threshold)

        test_auc_roc = roc_auc_score(self.y_test_true, self.y_test_probs)

        ax.plot(
            fpr,
            tpr,
            color="b",
            # label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            label=f"ROC Test\n(AUC = {test_auc_roc: 0.3f})\n(Best threshold = {best_threshold:.1%})",
            lw=2,
            alpha=0.8,
        )

        ax.plot([0, 1], [0, 1], linestyle="--", color="r", lw=1)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("Taux de Faux Positifs")
        plt.ylabel("Traux de Vrais Positifs")
        plt.legend(loc="lower right")
        plt.grid(False)
        plt.tight_layout()
        # plt.show()
        return fig


# wrapper pour pouvoir utiliser des fonctionnalités avancées de sklearn avec un modèle de type cuml
class CumlWrapper:
    def __init__(self, cuml_model):
        self.cuml_model = cuml_model

    def fit(self, X, y):
        X, y = sk_valid_check_X_y(X, y)
        self.cuml_model.fit(X, y)
        return self

    def predict(self, X):
        X = sk_valid_check_array(X)
        return self.cuml_model.predict(
            X
        ).get()  # `.get()` pour récupérer un numpy array

    def predict_proba(self, X):
        X = sk_valid_check_array(X)
        return self.cuml_model.predict_proba(X).get()

    def score(self, X, y):
        X, y = sk_valid_check_X_y(X, y)
        return self.cuml_model.score(X, y).get()


def plot_evaluation_scores(
    train_scores,
    val_scores,
    title="Scores de validation croisée\n",
    subtitle="",
    figsize=(8, 5),
    verbose=True,
):
    # Scores des dictionnaires de scores qu'on ne veut pas faire figurer dans le graphique
    not_to_plot = ["fp", "fn", "tp", "tn", "fit_time"]
    metric_names = [k for k in val_scores.keys() if k not in not_to_plot]

    # Les dictionnaires en paramètres contiennent des listes de scores pour pour tous les folds,
    # On construit le dictionnaire contenant les scores moyens pour le train et la validation
    mean_scores = {
        "Train": [np.mean(train_scores[k]) for k in metric_names],
        "Validation": [np.mean(val_scores[k]) for k in metric_names],
    }
    # On calcule les écart_types pour les afficher sous forme de barre d'erreur
    errors = {
        "Train": [np.std(train_scores[k]) for k in metric_names],
        "Validation": [np.std(val_scores[k]) for k in metric_names],
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


def plot_roc_curve(
    y_true,
    y_score,
    title="Receiver Operating Characteristic (ROC) Curve",
    subtitle="",
    figsize=(5, 5),
):
    # Taux de faux positifs et taux de faux négatifs
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=figsize)
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
    plt.plot(fpr, tpr, label="ROC Curve (sklearn)")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


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


def plot_recall_mean(
    tn,
    fp,
    fn,
    tp,
    title="Matrice de confusion (% par ligne)",
    subtitle="",
    figsize=(5, 5),
    n_samples=True,
    verbose=False,
):
    """Plotte la matrice de confusion en pourcentage par lignes
    Utilisée pour une cross_evaluation (matrice de confusion moyenne à travers les folds) ou si CUDA

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


def format_percent(x, _):
    return f"{x * 100:.0f}%"


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
