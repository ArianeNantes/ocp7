import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold as SkStratifiedKFold
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from src.p7_constantes import VAL_SEED

"""import os
import gc
import cudf
import cupy as cp
import cuml
from cuml.model_selection import StratifiedKFold as CuStratifiedKFold
from copy import deepcopy
import time
from modeling.src.p7_metric import business_gain_score"""


class RfecvSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        pipe,
        cv=4,
        min_features=100,
        step=1,
        # param_bg=None,
        random_state=VAL_SEED,
        verbose=True,
    ):
        self.pipe = pipe
        self.cv = cv
        self.min_features = min_features
        self.step = step
        if self.min_features <= self.step:
            print(
                f"WARNING : le minimum de features ({self.min_features}) est ignoré, il est > au step {self.step}"
            )

            self.min_features = self.min_features + self.step
        # self.param_bg = param_bg
        self.random_state = random_state
        self.verbose = verbose
        # Si l'argument pipe est de type pipeline, on récupère le modèle dans le dernier step,
        # sinon on considère que c'est le modèle lui-même
        if "pipeline" in str(type(self.pipe)):
            self.model = self.pipe.steps[-1][1]
        else:
            self.model = self.pipe
        print("self.model type", type(self.model))
        self.best_features = []
        self.best_score = -np.inf
        self.mean_scores = []
        self.removed_features = []
        self.history_all = []

    def fit(self, X, y=None, threshold_prob=0.5):
        self.threshold_prob = threshold_prob

        features = list(X.columns)
        best_score = -np.inf
        best_features = features.copy()

        # Nombre de features restantes à chaque fois que l'on élimine une feature
        feature_counts = []
        # Score obtenu à chaque fois que l'on élimine une feature
        mean_scores = []

        # Tant qu'on n'a pas atteint le minimum de features à conserver,
        # on fit le modèle et on score le modèle en validation croisée,
        # puis on élimine la feature la moins importante
        iteration = 0
        while len(features) >= self.min_features:
            iteration += 1

            scores = []
            # Stocker les importances de chaque fold
            importances_folds = []
            folds = SkStratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
            for train_idx, valid_idx in folds.split(X, y):
                X_train, X_val = (
                    X[features].iloc[train_idx].copy(),
                    X[features].iloc[valid_idx].copy(),
                )
                y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]

                self.pipe.fit(X_train, y_train)
                # On met à jour le modèle après le fit du pipeline
                if "pipeline" in str(type(self.pipe)):
                    model_name, _ = self.pipe.steps[-1]
                    self.model = self.pipe.named_steps[model_name]
                else:
                    self.model = self.pipe

                # print("model_fitté ?", self.is_model_fitted())

                # Prédictions et scores
                y_prob = self.pipe.predict_proba(X_val)[:, 1]

                score = roc_auc_score(y_val, y_prob)
                scores.append(score)

                # Importance des features
                if hasattr(self.model, "coef_"):
                    importances = np.abs(self.model.coef_).flatten()
                elif hasattr(self.model, "feature_importances_"):
                    importances = self.model.feature_importances_

                else:
                    raise ValueError(
                        "Le modèle ne fournit pas d'importances de variables."
                    )
                importances_folds.append(importances)

            # Moyenne des scores sur les folds
            mean_score = np.mean(scores)
            feature_counts.append(len(features))
            # On conserve le score obtenu dans la liste des scores obtenus à chaque itération
            mean_scores.append(mean_score)

            # On met à jour l'historique
            self.history_all.append(
                {
                    "iteration": iteration,
                    "n_features": len(features),
                    "auc": mean_score,
                    "features": features.copy(),
                }
            )

            if self.verbose:
                print(
                    # f"Auc {mean_score:.4f}   Remove '{feature_to_remove}' (importance={mean_importances[min_idx]:.4f})"
                    f"{len(features)} features -> Auc = {mean_score:.4f}"
                )

            if mean_score >= best_score:
                best_score = mean_score
                best_features = features.copy()

            # Moyenne des importances sur les folds
            mean_importances = np.mean(importances_folds, axis=0)

            # Si on atteint le minimum de features à conserver, on sort
            if len(features) <= max(self.min_features, self.step):
                break

            for i in range(self.step):
                # On récupère la variable de plus faible importance et on la retire des features pour la prochaine itération
                min_idx = np.argmin(mean_importances).item()
                feature_to_remove = features[min_idx]
                features.pop(min_idx)
                self.removed_features.append(feature_to_remove)

            self.best_features = best_features
            self.best_score = best_score
            self.mean_scores = mean_scores
        return self

    @staticmethod
    def _compute_confusion(y_true, y_pred):
        # Calcul rapide
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tn, fp, fn, tp

    def is_model_fitted(self):
        try:
            check_is_fitted(self.model)
            return True
        except NotFittedError:
            return False

    def get_features(self, n_features=None):
        if n_features:
            history = pd.DataFrame((self.history_all))
            features = history.loc[history["n_features"] == n_features, ["features"]][
                "features"
            ].to_list()[0]
            return features
        else:
            return self.best_features

    def get_history(self, n_features_start=None, n_features_end=None):
        history = pd.DataFrame(self.history_all)
        if n_features_start and n_features_end:
            filtered_history = history[
                (history["n_features"] >= n_features_start)
                & (history["n_features"] <= n_features_end)
            ].sort_values(by="auc", ascending=False)
            return filtered_history
        else:
            return history

    # Trace un graph de type ligne avec sur l'axe des x le nombre de features et sur l'axe des y le score obtenu.
    def plot_scores(self, y_lim=[0.5, 1], model_name=""):
        # On récupère les données dans l'historique
        history = pd.DataFrame(self.history_all)
        fig, ax = plt.subplots(figsize=(15, 5))

        # Tracer la courbe
        ax.plot(
            history["n_features"],
            history["auc"],
            marker="o",
            markersize=2,
            linestyle="-",
            linewidth=0.7,
            label="Score AUC",
        )

        # Étiquettes et titre
        ax.set_xlabel("Nombre de features")
        ax.set_ylabel("Score AUC")
        title = "AUC en fonction du nombre de features"
        if model_name:
            title += f"{model_name}"
        ax.set_title(title)

        # Définir les ticks majeurs et mineurs sur l'axe des X
        ax.xaxis.set_major_locator(MultipleLocator(50))  # Ticks principaux tous les 50
        ax.xaxis.set_minor_locator(MultipleLocator(10))  # Ticks mineurs tous les 10

        # Ligne verticale au meilleur score
        ax.axvline(
            x=len(self.best_features),
            color="black",
            linestyle="--",
            label="Meilleur nombre de features",
        )

        # Grille : seulement sur les ticks mineurs de l'axe X
        ax.grid(
            which="minor", axis="x", color="lightgray", linestyle="--", linewidth=0.8
        )
        ax.grid(which="major", axis="x", color="gray", linestyle="-", linewidth=1)
        ax.grid(
            which="major", axis="y", color="gray", linestyle="-", linewidth=1
        )  # Facultatif : grille sur Y

        # Fixer les limites en Y
        ax.set_ylim(y_lim[0], y_lim[1])

        plt.legend()
        # Afficher
        plt.show()
