import matplotlib.pyplot as plt

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
from sklearn.model_selection import StratifiedKFold as SkStratifiedKFold

from src.modeling.p7_constantes import VAL_SEED


# Calcule les importances de features (nombre de splits) grâce à un model comme un classifier LightGBM en validation croisée
def importance_from_model(model, X, y, n_splits=5, seed=VAL_SEED):
    """clf = lgb.LGBMClassifier(
        n_threads=n_threads,
        class_weight="balanced",
        objective="binary",
        random_state=seed,
        verbosity=-1,  # Pour ne pas voir les logs
    )"""

    folds = SkStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    all_importance_df = pd.DataFrame()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        X_train, y_train = (
            X.iloc[train_idx],
            y.iloc[train_idx],
        )
        X_val, y_val = (
            X.iloc[valid_idx],
            y.iloc[valid_idx],
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            # eval_metric="auc",
        )
        # y_score_val = clf.predict(X_val)
        # A chaque fold, on crée un df contenant les importances du fold avec 3 colonnes :
        # 1: nom de la feature, 2 : son importance sur le fold, 3 : le n° du fold
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = list(X.columns)
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        # On concatène les df de fold les uns en dessous des autres dans un df qui contient toutes les importances
        all_importance_df = pd.concat([all_importance_df, fold_importance_df], axis=0)

    # On calcule la moyenne et l'écart type des importances obtenus sur les folds
    # en les triant de la plus haute importance à la plus basse
    mean_importance = (
        all_importance_df.drop("fold", axis=1)
        .groupby("feature")
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
        )
        .sort_values(by="importance_mean", ascending=False)
    )

    # On ajoute une colonne Rank (pour pouvoir trier / détrier dans n'importe quel sens par la suite)
    ranks = range(1, len(mean_importance) + 1)
    mean_importance["rank_importance"] = ranks
    df_res = mean_importance

    # On ajoute le pourcentage de valeurs manquantes pour chaque feature
    missing_df = pd.DataFrame(X.isna().mean() * 100, columns=["missing_pct"])
    df_res = mean_importance.join(missing_df)

    return df_res


# Dessine sur un même graphique les importances (ordonnées à gauche) sous forme de ligne
# et les pourcentages de valeurs manquantes (ordonnées à droite) sous forme de points
# en fonction du rang d'importance des features (abscisses) de 1 la plus importante à max
def plot_line_importance_with_missing(
    mean_importances,
    col_importance="importance_mean",
    col_rank="rank_importance",
    col_missing="missing_pct",
):
    threshold_missing = 30
    color_missing_inf30 = "green"
    color_missing_sup30 = "orange"
    color_importance = "blue"

    fig, ax_importance = plt.subplots(figsize=(10, 6))

    # Séparer les points selon le seuil 30% de valeurs manquantes
    below_thresh = mean_importances[mean_importances[col_missing] <= threshold_missing]
    above_thresh = mean_importances[mean_importances[col_missing] > threshold_missing]

    # Axe de gauche : importance
    min_rank_importance = mean_importances[col_rank].min()
    max_rank_importance = mean_importances[col_rank].max()

    ax_importance.set_xlabel(
        f"Rang d'Importance de la Feature ({min_rank_importance} = la + plus importante, {max_rank_importance} = la - importante)"
    )

    ax_importance.set_ylabel("Importance", color=color_importance)
    ax_importance.tick_params(axis="y", labelcolor=color_importance)
    ax_importance.set_ylabel("Importance moyenne (nb splits)")

    # Axe de droite : pourcentage de valeurs manquantes
    ax_missing = ax_importance.twinx()

    ax_missing.set_ylabel("Pourcentage de valeurs manquantes")
    # Définir les ticks personnalisés pour l'axe des % de missing (pour que la grille tombe à des endroits "souhaités")
    missing_custom_ticks = [30, 40, 50, 60, 70, 80, 90]
    ax_missing.set_yticks(missing_custom_ticks)

    # Activer la grille uniquement pour l'axe de droite, avec ces ticks
    ax_missing.yaxis.grid(
        True, which="major", color="lightgrey", linestyle="--", linewidth=0.7
    )

    # On trace la ligne qui représente l'importance
    line_importance = ax_importance.plot(
        mean_importances[col_rank],
        mean_importances[col_importance],
        color=color_importance,
        marker="o",
        markersize=3,
        linewidth=0.7,
        label="Importance",
    )

    # On trace les points < 30% de missing en orange
    line_missing_inf30 = ax_missing.plot(
        below_thresh[col_rank],
        below_thresh[col_missing],
        color=color_missing_inf30,
        marker="o",
        linestyle="None",
        markersize=2,
        label=f"% manquantes ≤ {threshold_missing}%",
    )

    # On trace les points au pourcentage de valeurs manquantes >= 30% en orange
    line_missing_sup30 = ax_missing.plot(
        above_thresh[col_rank],
        above_thresh[col_missing],
        color=color_missing_sup30,
        marker="v",
        linestyle="None",
        markersize=5,
        label=f"% manquantes > {threshold_missing}%",
    )

    # Titre et mise en page
    plt.title("Etude des features : Importance & Valeurs manquantes\n\n\n")

    # Combine les légendes
    lines = line_importance + line_missing_inf30 + line_missing_sup30
    labels = [l.get_label() for l in lines]

    # Légende en dehors du graphique (sinon illisible)
    ax_importance.legend(
        lines,
        labels,
        loc="lower left",  # Position dans la boîte définie par bbox
        bbox_to_anchor=(0.0, 1.02),  # Juste au-dessus du graphique, aligné à gauche
        ncol=3,  # Légende sur une seule ligne
        # frameon=True,
        # framealpha=1.0,
        # facecolor="lightgrey",
        edgecolor="black",
        fontsize=9,
        borderpad=0.8,
    )

    fig.tight_layout()
    plt.grid(True)
    return fig


def plot_line_importance(
    mean_importances,
    col_importance="importance_mean",
    col_rank="rank_importance",
):

    fig, ax_importance = plt.subplots(figsize=(10, 6))

    # Axe des abscisses
    min_rank_importance = mean_importances[col_rank].min()
    max_rank_importance = mean_importances[col_rank].max()

    ax_importance.set_xlabel(
        f"Rang d'Importance de la Feature ({min_rank_importance} = la + plus importante, {max_rank_importance} = la - importante)"
    )

    ax_importance.tick_params(axis="y")
    ax_importance.set_ylabel("Importance moyenne (nb splits)")

    # On trace la ligne qui représente l'importance
    line_importance = ax_importance.plot(
        mean_importances[col_rank],
        mean_importances[col_importance],
        # color=color_importance,
        marker="o",
        markersize=3,
        linewidth=0.7,
        label="Importance",
    )

    # Titre et mise en page
    plt.title("Importance des features en nombre de splits")

    fig.tight_layout()
    return fig


# Graphique en barres horizontales les importances pour les features les plus importantes (ex : top 40)
# Avec plus ou moins l'écart type en forme de barres d'erreur
def plot_bar_importances(
    mean_importance,
    col_mean="importance_mean",
    col_std="importance_std",
    top=40,
    sort_by_name=False,
    save_img=False,
):
    best_importance = mean_importance.head(top)
    # Barh inverse l'ordre donc on trie en sens inverse
    if sort_by_name:
        best_importance = best_importance.sort_values(by="feature", ascending=False)
    else:
        best_importance = best_importance.sort_values(by=col_mean, ascending=True)

    # Taille de la figure
    width = 8
    margin = 1
    bar_height = 0.25
    height = margin + bar_height * top

    # print(best_importance)
    fig = plt.figure(figsize=(width, height))
    # Récupère l'axe en cours
    ax = plt.gca()

    barplot = best_importance[col_mean].plot(kind="barh", width=0.85)

    # Récupérer les coordonnées des barres du barplot
    x_positions = [bar.get_width() for bar in barplot.patches]
    y_positions = [bar.get_y() + bar.get_height() / 2 for bar in barplot.patches]

    # Ajouter les barres d'erreur
    plt.errorbar(
        x=x_positions,
        y=y_positions,
        # data=best_importance,
        xerr=[best_importance[col_std], best_importance[col_std]],
        fmt="none",  # Aucun marqueur pour les points de données
        # capsize=5,  # Taille des barres à l'extrémité des lignes d'erreur
        color="black",  # Couleur des barres d'erreur
    )

    fig.suptitle(f"Top {top} des Features les plus importantes par LightGBM\n")
    ax.set(
        ylabel="",
        # title=f"Titre du graphique",
        xlabel=f"Importance moyenne en nombre de splits",
    )
    # ax.set_title("Titre du graphique", fontsize=10)
    plt.tight_layout()
    return fig


# Graphique en barres horizontales pour les features les plus importantes.
# Deux plots dans la même figure :
# A gauche les importances, à droite les % de valeurs manquantes correspondantes
def plot_bar_importances_with_missing(
    mean_importances,
    col_mean="importance_mean",
    col_std="importance_std",
    col_missing="missing_pct",
    top=40,
    sort_by_name=False,
):
    threshold_missing = 30
    best_importance = mean_importances.sort_values(by=col_mean, ascending=False).head(
        top
    )

    # Trier les données
    if sort_by_name:
        best_importance = best_importance.sort_values(by="feature", ascending=False)
    else:
        best_importance = best_importance.sort_values(by=col_mean, ascending=True)

    # Taille de la figure et sous-graphiques
    width = 10
    margin = 1
    bar_height = 0.25
    height = margin + bar_height * top

    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        figsize=(width, height),
        gridspec_kw={"width_ratios": [2, 1]},
        sharey=True,  # Partage les features sur l'axe Y
    )

    # --- Graphique 1 : importance + std ---
    best_importance.plot(
        kind="barh",
        y=col_mean,
        xerr=col_std,
        ax=ax1,
        width=0.85,
        legend=False,
        # grid=True,
        # color="skyblue",
        # edgecolor="black",
    )
    ax1.set_xlabel("Nombre de splits moyen (± écart-type)")
    ax1.set_ylabel("Feature")
    ax1.set_title("Importance moyenne sur les folds en nb de splits")
    # Activer la grille uniquement pour le plot de gauche
    ax1.yaxis.grid(
        True, which="major", color="lightgrey", linestyle="--", linewidth=0.7
    )
    ax1.set_axisbelow(True)  # Force la grille à être dessous les barres

    # --- Graphique 2 : pourcentage de valeurs manquantes ---
    best_importance.plot(
        kind="barh",
        y=col_missing,
        ax=ax2,
        width=0.85,
        legend=False,
        color="grey",
        edgecolor="grey",
    )
    ax2.axvline(x=30, color="grey", linestyle="--", label=f"Seuil {threshold_missing}%")
    # ax2.text(30 + 1, 0.5, "Seuil 30%", color="gray", fontsize=8)
    ax2.set_xlabel("% de valeurs manquantes")
    ax2.set_title("% valeurs manquantes")

    # Ajustements
    fig.suptitle(f"Top {top} des features les plus importantes LightGBM", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig
