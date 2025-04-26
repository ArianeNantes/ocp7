import pandas as pd
import numpy as np
import cudf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.decomposition import PCA

import scipy.stats as stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, boxcox
from collections import defaultdict


# import powerlaw
import pingouin as pg
from sklearn.discriminant_analysis import StandardScaler

# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from bioinfokit.analys import stat
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.patheffects as pe

# from matplotlib.ticker import AutoMinorLocator
from IPython.display import display
import matplotlib.patches as mpatches
import math

import warnings

# Personnalisées
from .p7_preprocess import check_no_nan
from src.p7_regex import sel_item_regex

# from src.p7_regex import sel_lines_regex
# from src.p7_outlier import outliers_one_boxplot_subplot
# from src.p7_outlier import counts_outliers_iqr
# from src.p7_pre_process import LogTranformer, BoxCoxTranformer

from src.p7_color import get_palette_colors

"""***********************************************************************************************************************
RUBRIQUE BALANCE
**************************************************************************************************************************
"""


def plot_default_ratio(series):
    labels = ["Remb. Ok", "Défaut"]
    # Plot en camembert
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"Pourcentage de prêts avec et sans défaut de remboursement\nNombre d'observations : {len(series):_}",
        ha="left",  # Aligne les deux ligne du titre sur leur gauche
        x=0.0,  # Positionne les lignes du titres complètement à gauche
    )
    ratio = series.value_counts(normalize=True)
    # Si on a un cudf la conversion en numpy est nécessaire. Si on a un pd.df, la conversion ne nuit pas.
    ax.pie(
        series.value_counts(normalize=True).to_numpy(),
        labels=labels,
        autopct="%.1f%%",
    )

    fig.legend(loc="lower right")
    fig.tight_layout()
    return fig


"""***********************************************************************************************************************
RUBRIQUE TRANSFORMATIONS
**************************************************************************************************************************
"""


def sort_by_skewness(data, to_process=None):
    # Si to_process est spécifié
    if to_process is not None:
        # On s'assure que les colonnes spécifiées dans to_process font bien partie du df et qu'elles sont
        # numériques
        to_process = [
            col
            for col in to_process
            if col in list(data.select_dtypes(include="number"))
        ]
    # Si to_process n'est pas spécifié, on étudie toutes les variables quantitatives
    else:
        to_process = list(data.select_dtypes(include="number"))

    # On calcule les skewness
    skewness = data[to_process].skew().sort_values(ascending=False)
    display(skewness)
    return skewness


def x_to_ln(data, to_process, inplace=False, verbose=True):
    # Si to_process est une chaîne on considère que c'est un nom de variable (pas une regex)
    if type(to_process) == str:
        to_process = [to_process]

    # Si une colonne contient des données < 0, on ne la traite pas
    for col in to_process.copy():
        n_neg = data[data[col] <= -1].shape[0]
        if n_neg > 0:
            print(
                f"La colonne {col} contient {n_neg} valeurs <=-1, elle ne sera pas transformée"
            )
            to_process.remove(col)

    # S'il reste des colonnes
    if len(to_process) > 0:
        # Si la transformation est demandée sur place, on effectue la transformation dans data
        if inplace == True:
            df = data
        # sinon on copie data dans un nouveau df
        else:
            df = data.copy()

        for col in to_process:
            col_name = "ln_" + col
            """# S'il y a des zéros on ajoute 1 dans toute la colonne avant le log (neperien)
            n_zeros = df[df[col]==0].shape[0]
            if n_zeros > 0:
                df[col_name] = df[col].apply(lambda x: np.log1p(x))
                if verbose:
                    print(f"col {col} : fonction np.log1p")
            # S'il n'y a pas de zéros, on applique le log direct (neperien)
            else:
                df[col_name] = df[col].apply(lambda x: np.log(x))
                if verbose:
                    (f"col {col} : fonction np.log")"""

            # Finalement on applique toujours log (x+1) par souci de simplicité
            df[col_name] = df[col].apply(lambda x: np.log1p(x))
    # Si verbose
    if verbose:
        print(f"{len(to_process)} colonnes transformées :")
        print(to_process)
    return df, to_process


"""***********************************************************************************************************************
RUBRIQUE TESTS ET CORRELATIONS
**************************************************************************************************************************
"""


# Cours Open classroom sur l'anova
def eta_squared(x, y, verbose=False):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x == classe]
        classes.append({"ni": len(yi_classe), "moyenne_classe": yi_classe.mean()})
    SCT = sum([(yj - moyenne_y) ** 2 for yj in y])
    if verbose:
        print(f"Eta squared {y.name} x {x.name} :")
        print("SCT (variation totale / Total sum squares)=", SCT)
    SCE = sum([c["ni"] * (c["moyenne_classe"] - moyenne_y) ** 2 for c in classes])
    if verbose:
        print("SCE (Variation interclasse / Sum of squares of the model)=", SCE)
        print("SCE/SCT (Rapport de corrélation) =", SCE / SCT)
    return SCE / SCT


def bivariate_anova_non_param(data, x, y):
    # Voir https://fr.wikipedia.org/wiki/Test_de_Kruskal-Wallis
    # H0 toutes les médianes sont égales

    # On enlève les NaN sur x et y
    df = data.dropna(subset=[x, y], how="any")

    # On extrait les données y pour chaque modalité (level) de la variable qualitative x
    levels = df[x].sort_values().unique()
    groups = []
    for level in levels:
        # Squeeze permet de transformer en Series.
        groups.append(df.loc[df[x] == level, [y]])

    # Test Kruskall avec pinguin
    print(f"\nTest de Kruskal-Wallis ({y} par modalités de {x})")
    print("H0 : Toutes les médianes sont égales")
    result_kruskal = pg.kruskal(data=df, dv=y, between=x)
    display(result_kruskal)

    """ Trop de RAM nécessaire
    # Tests post-hocs
    print(f"Tests post-hoc Mann-Whitney ({y} par modalités de {x})")
    result_mann_whitney = pg.pairwise_tests(
        dv=y, data=df, between=x, parametric=False
    ).round(3)
    display(result_mann_whitney)
    """
    # welch anova (robuste à la non homogénéité des variances)
    print(f"Welch ANOVA ({y} par modalités de {x})")
    w_anova = pg.welch_anova(dv=y, between=x, data=df)
    display(w_anova)

    # Tests post-hoc Games-Howell, robuste à la non-homogénéité des variances
    print(f"Tests post-hoc Games-Howell ({y} par modalités de {x})")
    result_gameshowell = pg.pairwise_gameshowell(data=df, dv=y, between=x).round(3)
    display(result_gameshowell)
    return


def compare_density_subplot(
    series,
    ax,
    n_bins=40,
    y_label=True,
    x_label=False,
    # palette="muted",
):
    # Si series est un dataframe, on le convertit en series
    if type(series) == pd.DataFrame:
        series = series.squeeze()

    # On calcule la skewness et la kurtosis de la série
    skew = series.skew()
    kurt = series.kurtosis()

    # colors = sns.color_palette(palette)[0:5]

    # On trace l'histogramme
    # fig, ax = plt.subplots(figsize=figsize)
    # label_data = f"Données réelles\nAsymétrie : {skew:.2f}\nApplatis. : {kurt:.2f}"
    label_data = f"Données réelles\nAsymétrie : {skew:.2f}"
    sns.histplot(
        data=series, bins=n_bins, stat="density", kde=False, ax=ax, label=label_data
    )

    # Densité Loi normale de même moyenne et std
    # On récupère les min et max de l'axe des x
    x_min = ax.get_xlim()[0]
    x_max = ax.get_xlim()[1]
    # On récupère la moyenne et l'écart_type
    mean = series.mean()
    std = series.std()
    # On crée un tableau de n valeurs suivant une loi normale
    x_gauss = np.linspace(x_min, x_max, series.shape[0])
    # On calcule les valeurs de la courbe de densité
    y_gauss = stats.norm.pdf(x_gauss, mean, std)
    # On trace la courbe de Gauss
    ax.plot(x_gauss, y_gauss, color="red", label="Densité Loi normale")

    # Estimation de densité des données réelles
    series.plot(kind="density", label="Densité données (est.)")

    # titre des axes
    if y_label:
        ax.set_ylabel("Densité")
    else:
        ax.set_ylabel("")

    if not x_label:
        ax.set_xlabel("")

    ax.legend()

    return ax


def compare_density(
    series, figsize=(10, 6), n_bins=None, compare_to="norm", palette="muted", mode=False
):
    """Trace un histogramme pour une distribution. Affiche la densité, la courbe de l'estimation de densité de la distri et la courbe de
    densité d'une loi normale de même moyenne et écart type. Affiche la skewness.

    Args:
        series (Series ou Dataframe d'une colonne): La Series contenant les données de la distribution
        figsize (tuple, optional): Taille de la figure. Defaults to (10, 6).
        n_bins (int, optional): Nombre de bins, Si None règle de Sturges. Defaults to None.
        compare_to (str, optional): Forcément 'norm'. Defaults to 'norm'. [to do] : prévoir d'autre cas de comparaison que la loi normale
        palette (str, optional): palette de couleur. Defaults to 'muted'.
        mode (bool, optional): Affiche une ligne pour le mode en plus de la médiane et de l'écart type. Defaults to False.

    Returns:
        _type_: l'ax
    """
    # Si series est un dataframe, on le convertit en series
    if type(series) == pd.DataFrame:
        series = series.squeeze()

    # On calcule le nombre de bins optimal avec la règle de Sturges :
    n = series.shape[0]
    if n_bins is None:
        n_bins = int(1 + np.round(np.log2(n)))

    # On calcule la skewness et la kurtosis de la série
    skew = series.skew()
    kurt = series.kurtosis()

    colors = sns.color_palette(palette)[0:5]

    # On trace l'histogramme
    fig, ax = plt.subplots(figsize=figsize)
    # label_data = f"Données réelles\nAsymétrie : {skew:.2f}\nApplatis. : {kurt:.2f}"
    label_data = f"Données réelles\nAsymétrie : {skew:.2f}"
    sns.histplot(
        data=series, bins=n_bins, stat="density", kde=False, ax=ax, label=label_data
    )

    # Densité Loi normale de même moyenne et std
    # On récupère les min et max de l'axe des x
    x_min = ax.get_xlim()[0]
    x_max = ax.get_xlim()[1]
    # On récupère la moyenne et l'écart_type
    mean = series.mean()
    std = series.std()
    # On crée un tableau de n valeurs suivant une loi normale
    x_gauss = np.linspace(x_min, x_max, n)
    # On calcule les valeurs de la courbe de densité
    y_gauss = stats.norm.pdf(x_gauss, mean, std)
    # On trace la courbe de Gauss
    plt.plot(x_gauss, y_gauss, color="red", label="Densité Loi normale")

    # Estimation de densité des données réelles
    series.plot(kind="density", label="Densité données (est.)")

    # titre axe des y
    ax.set_ylabel("Densité")

    ax.legend()
    plt.tight_layout()
    plt.show()

    return ax


def univariate_barh(
    series,
    limit_top=99,
    limit_bottom=1,
    annotate="all",
    figsize=None,
    sort="value",
    n_max=60,
    palette=None,
    colors_idx=[0, 2, 1],
    right_offset=1.05,
    counts_offset=1.09,
    verbose=False,
):
    """Trace un graphique horizontal des % de valeurs manquantes pr colonnes.
    Les colonnes sont divisées en 3 groupes de couleurs différentes pour visualiser les très vides, les intermédiaires
    et les moins de 20%

    Args:
        data (DataFrame): df contenant les données
        missing_values (Series, optional): Series contenant les % de valeurs manuqantes par colonne. Defaults to None.
        limit_top (int, optional): Seuil % de valeurs manquantes pour distinguer le groupe très manquant. Defaults to 100.
        annotate (str, optional): Groupe pour lequel on veut voir les % affichés sur le graphe. Defaults to 'group_middle'.
        figsize (tuple, optional): Taille de la figure. Si non précisé elle est calculée approximativement. Defaults to None.
        sort (str, optional): Trie les barres (colonnes) par valeurs ou par ordre alphabétique des noms de colonnes. Defaults to 'value'.
        palette (palette, optional): palette à utiliser pour les couleurs. Defaults to None.
        colors_idx (list, optional): liste d'index de couleurs dans la palette à utiliser pour les 3 groupes. Defaults to [0, 2, 1]. 1er=groupe du haut (2=vert, 0=bleu et 1=orange)
        verbose (bool, optional): Affiche des messages d'erreurs. Defaults to False.

    Returns:
        3 objets: la Series des valeurs manquantes par colonne, la figure et l'ax
    """
    # ********************************************************
    # CONSTANTES à modifier rapidement
    # ********************************************************
    n_max_ticks_up = 40  # Nombre max de modalités à partir duquel on affiche les ticks en haut en plus du bas
    fontsize = 9  # Taille de police pour les éléments autres que le titre
    fontsize_title = 14  # Taille de police du titre
    decimals = (
        1  # Nombre de décimales à afficher et à prendre en compte pour les seuils
    )
    title = f"{series.name}\n"

    # ********************************************************
    # En fonction des paramètres fournis
    # ********************************************************

    # Limite le nombre de modalités à afficher
    n = series.nunique()
    if n > n_max:
        print(
            f"Le nombre de modalités de {series.name}={n} dépasse le seuil de {n_max}. Graphique non réalisé"
        )
        return

    # Calcul des fréquences par modalités.
    frequencies = series.astype(str).value_counts(normalize=True) * 100
    counts = series.astype(str).value_counts()

    # Tri
    if sort is not None and type(sort) == str:
        if sort == "index":
            frequencies = frequencies.sort_index(ascending=False)
            counts = counts.sort_index(ascending=False)
        # Tri par % de valeur manquantes, les plus vides se retrouveront en haut
        if sort == "value" or sort == "values":
            frequencies = frequencies.sort_values(ascending=True)
            counts = counts.sort_values(ascending=True)

    # Si l'élément palette fourni est une string, on la convertit en liste de couleurs
    # Si None, on prend le cycle de couleurs Matplotlib en cours
    if (
        palette is None
        or type(palette) == str
        or type(palette) == "seaborn.palettes._ColorPalette"
    ):
        # palette = sns.color_palette(palette, n_colors=6)
        palette = sns.color_palette(palette)

    # Si aucune taille de figure n'est fournie, on l'estime grossièrement :
    # Une marge pour le titre en haut et les ticks/labels haut et bas + un nombre de pouces par barre
    # Pour ajuster les constantes, voir avec le temps / l'expérience
    if figsize is None:
        width = 8
        margin = 1
        coefficient = 0.2
        # On met un coefficient différent de hauteur pour les barres en fonction du nombre de variables
        if n < 10:
            coefficient = 0.18
        elif n < 40:
            coefficient = 0.19
        elif n < 100:
            coefficient = 0.2
        elif n < 150:
            coefficient = 0.21
        else:
            coefficient = 0.22
        height = margin + coefficient * n
        figsize = (width, height)
        if verbose:
            print(f"Taille estimée de figure pour {n} variables :", (width, height))

    # Si limit_top est une chaîne (modalité), on prend le seuil de cette modalité
    if type(limit_top) == str:
        # On vérifie que la chaîne fournie est bien dans l'index des modalités
        if limit_top in list(frequencies.index):
            limit_top = np.ceil(frequencies[limit_top])
        else:
            print(f"'{limit_top}' ne figure pas dans les modalités de {series.columns}")

    # ********************************************************
    # Figure et axe
    # ********************************************************
    # Crée la figure
    fig, ax = plt.subplots(figsize=figsize)

    """# Récupère l'axe en cours
    ax = plt.gca()"""

    # ********************************************************
    # TITRE du graphique
    # ********************************************************
    ax.set_title(title, fontsize=fontsize_title)

    # ********************************************************
    # Groupes et couleurs - Dessin des barres
    # ********************************************************

    # On prend trois couleurs dans la palette, pour faire les trois groupes
    colors = []
    for color in colors_idx:
        if type(colors_idx[color]) == int and colors_idx[color] > len(palette):
            colors_idx[color] = color
            if verbose:
                print(
                    f"La couleur d'indice {color} a été remplacée. Dépassement de la longueur de la palette"
                )
        colors.append(palette[color])

    # On définit la couleur pour chaque barre en fonction de leur valeur
    # Et on range les noms de modalités dans group_top, group_middle ou group_bottom
    color_bar = []
    group_top = []
    group_middle = []
    group_bottom = []
    for i in range(len(frequencies)):
        color_bar.append("b")
        if frequencies[i] <= limit_bottom:  # groupe du bas
            color_bar[i] = colors[2]
            group_bottom.append(frequencies.index[i])

        # Attention à l'arrondi, bizarre, ne semble pas hyper hyper fiable
        elif np.round(frequencies[i], decimals) >= limit_top:
            # elif missing_values[i] >= limit_top:
            color_bar[i] = colors[0]
            group_top.append(frequencies.index[i])
        else:
            color_bar[i] = colors[1]
            group_middle.append(frequencies.index[i])

    # ********************************************************
    # Axes et ticks
    # ********************************************************
    # On veut des ticks tous les 10% sur l'axe des X pour les major de l'axe des X
    ax.xaxis.set_major_locator(MultipleLocator(10))
    # On les veut en pourcentage
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    # On veut aussi des ticks sur l'axe des Y collés aux modalités
    ax.tick_params(axis="y", which="both", left=True, pad=0)
    # On définit là où il y a des labels : en bas et en haut si n > n_max_ticks_up, pad = labels collés aux ticks
    if n > n_max_ticks_up:
        top = True
    else:
        top = False
    ax.tick_params(
        axis="x",
        which="both",
        top=top,
        bottom=True,
        labeltop=top,
        labelbottom=True,
        pad=0,
    )
    # On diminue la taille de police des axes (both pour les minor et major ticks)
    ax.tick_params(axis="both", labelsize=fontsize)

    # Grille
    ax.grid(visible=True, which="major", axis="x")
    ax.grid(visible=False, which="both", axis="y")

    # ********************************************************
    # Plot
    # ********************************************************
    frequencies.plot(kind="barh", width=0.85, color=color_bar)
    # Titres des axes
    ax.set_ylabel(f"Modalités {series.name}")
    ax.set_xlabel("Fréquences")

    # ********************************************************
    # Affichage des étiquettes
    # ********************************************************
    # Si annotate n'est ni None ni False, on affiche les valeurs en pourcentage à droite des barres
    if annotate is not None and (
        type(annotate) != bool or type(annotate) == bool and annotate == True
    ):
        # On construit la liste de tous labels (pourcentage à 1 ou 2 chiffres après la virgule)
        if decimals == 1:
            for bars in ax.containers:
                labels = [f"{x:.1f}%" for x in bars.datavalues]
        else:
            for bars in ax.containers:
                labels = [f"{x:.2f}%" for x in bars.datavalues]

        # Si annotate est une chaîne, on considère que annotate est un groupe ou une regex
        # et on efface les labels des variables correspondantes
        if type(annotate) == str:
            if annotate == "group_top":
                annotate = group_top
            elif annotate == "group_middle":
                annotate = group_middle
            elif annotate == "group_bottom":
                annotate = group_bottom
            elif annotate == "all":
                annotate = group_bottom + group_middle + group_top
            else:
                # Si annotate est une chaîne mais pas un groupe, c'est une regex
                # On construit une liste avec les noms de modalités correspondant
                annotate = sel_item_regex(frequencies.index, annotate)
        # Si annotate est une liste ou un ensemble, on efface les labels des modalités de cette liste
        if type(annotate) == list or type(annotate) == set:
            for i, mv in enumerate(frequencies.index):
                if mv not in annotate:
                    labels[i] = ""
        # Si annotate n'est ni false ni None, on affiche les labels
        # (le padding semble n'avoir aucun effet dans la builded fonc. intéressant seulement si on les fait à la main)
        for bars in ax.containers:
            plt.bar_label(bars, labels=labels, padding=0.4, size=fontsize)

    # On recule la limite du graphique vers la droite pour laisser de la place aux annotations
    left, right = ax.get_xlim()
    right = right * right_offset
    ax.set_xlim(right=right)

    # Ajout des effectifs dans la marge de droite
    ax.text(
        counts_offset,
        0.5,
        "Effectifs",
        transform=ax.transAxes,
        rotation=-90,
        va="center",
    )
    for i, num in enumerate(counts):
        ax.text(right * 1.02, i, str(num), va="center", fontsize=fontsize)

    # Ajustement des marges
    plt.subplots_adjust(right=0.75)

    # ********************************************************
    # Légende
    # ********************************************************
    # Crée les artists et les labels pour la légende
    patch_100 = mpatches.Patch(color=colors[0], label=f">= {limit_top} %")
    patch_20 = mpatches.Patch(color=colors[2], label=f"< {limit_bottom} %")
    patch_others = mpatches.Patch(color=colors[1], label="Autres")

    # On choisit les artists à afficher dans la légende
    handles = []
    if len(group_top) > 0:
        handles.append(patch_100)
    if len(group_bottom) > 0:
        handles.append(patch_20)
    if len(handles) >= 1:
        handles.append(patch_others)
    # Si on trie par valeur la légende est en bas à droite, sinon on laisse choisir Matplotlib
    # Nécessaire car il ne choisit pas toujours ok
    if sort == "value":
        location = "lower right"
    else:
        location = "best"
    if len(handles) > 0:
        plt.legend(
            handles=handles,
            title="Fréquences",
            fontsize=fontsize,
            title_fontsize=fontsize,
            loc=location,
        )

    return


def univariate_histogram(
    series, figsize=(10, 6), n_bins=None, palette="muted", mode=False, verbose=False
):
    """Trace un histogramme des effectifs avec la moyenne et la médiane

    Args:
        series (Series ou Dataframe): Séries contenant les données de la distribution
        figsize (tuple, optional): Taille de la figure. Defaults to (10, 6).
        n_bins (_type_, optional): Nombre de bins, règle de Sturges si None. Defaults to None.
        palette (str, optional): Palette de couleurs. Defaults to 'muted'.
        mode (bool, optional): Affiche le mode en plus de la médiane et moyenne. Defaults to False.
        verbose (bool, optional): Verbosité. Defaults to False.
    """
    # Si series est un dataframe, on le transforme en série
    # Le dataframe ne doit avoir qu'une seule colonne
    if type(series) == pd.DataFrame:
        series = series.squeeze()

    # On retient la skewness particulièrement (asymétrie) pour pouvoir proposer des transformations habituelles
    skew = series.skew()

    # On affiche tous les indicteurs habituels
    print(f"Moyenne : {series.mean():.2f}")
    print(f"Médiane : {series.median():.2f}")
    print(f"Skewness : {skew:.2f}")
    print(f"Kurtosis : {series.kurtosis():.2f}")

    if verbose:
        if skew > 0:
            print(
                f"La distribution présente une trainée à droite (Assymétrie : {skew:.2f})"
            )
            print(
                "\nPropositions de correction fréquemment utilisées de l'asymétrie (trainée à droite): "
            )
            print("Sqrt(X), correction faible/moyenne")
            print("ln(X), correction moyenne")
            print("-1/x, correction forte")
            print("-1/x2, correction très forte")
        elif skew < 0:
            print(
                f"\nLa distribution présente une trainée à gauche (Assymétrie : {skew:.2f})"
            )
            print(
                "Propositions de correction fréquemment utilisées de l'asymétrie (trainée à gauche): "
            )
            print("x2, correction faible/moyenne")
            print("x3, correction moyenne/forte")
            print("exp(x), correction forte")
        print("Si x est un pourcentage, penser éventuellement à Arcsin(sqrt(x/100))")
        if series.kurtosis() > 0:
            print(
                "\nLa distribution est plus 'pointue' qu'une loi normale de même paramètres"
            )
        elif series.kurtosis() < 0:
            print(
                "\nLa distribution est plus 'applatie' qu'une loi normale de même paramètres"
            )

    # On calcule le nombre de bins optimal avec la règle de Sturges :
    n = series.shape[0]
    if n_bins is None:
        n_bins = int(1 + np.round(np.log2(n)))

    colors = sns.color_palette(palette)[0:5]

    # On trace l'histogramme
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=series, bins=n_bins, kde=True, ax=ax)

    # On récupère la valeur max de l'axe des y
    ymax = np.ceil(ax.get_ylim()[1])

    # axe des y
    ax.set_ylabel("Effectif")

    # On trace la moyenne, la médiane et le mode
    ax.vlines(
        series.mean(),
        ymax=ymax,
        ymin=0,
        color=colors[1],
        ls="-.",
        lw=2,
        label="Moyenne",
    )
    ax.vlines(
        series.median(),
        ymax=ymax,
        ymin=0,
        color=colors[4],
        ls="-",
        lw=2,
        label="Médiane",
    )
    if mode == True:
        ax.vlines(
            series.mode(),
            ymax=ymax,
            ymin=0,
            color=colors[2],
            ls="--",
            lw=2,
            label="Mode",
        )

    ax.legend()
    plt.tight_layout()
    plt.show()

    return


def eta_map(data, var_cat, var_num, display_table=True, figsize=(10, 20), verbose=True):
    """Affiche une matrice contenant les eta squared"""
    # On filtre les warnings car si beaucoup de features, warning concernant le constraint_layout sur les axex, aucun intérêt
    # warnings.filterwarnings("ignore")

    max_nunique = 100  # Nombre max de modalités pour les variables qualitatives

    # Si var_cat ou var_num sont des chaînes, on considère que c'est un nom de variable (pas une regex)
    if type(var_cat) == str:
        var_cat = [var_cat]
    if type(var_num) == str:
        var_num = [var_num]

    # Pour les messages d'erreurs, on fait la liste des variables qui ne sont pas dans le df
    var_not_in_df = [var for var in var_cat + var_num if var not in list(data.columns)]

    # On vérifie que les variables spécifiées dans var_num sont bien des var numériques du df
    var_not_num = [
        var for var in var_num if var in list(data.select_dtypes(exclude="number"))
    ]
    var_num = [
        var for var in var_num if var in list(data.select_dtypes(include="number"))
    ]
    # On exclut les variables qui comporte des NaN
    var_num_nan = [var for var in var_num if data[var].isnull().sum() > 0]
    var_num = [var for var in var_num if data[var].isnull().sum() == 0]

    # On vérifie que les variables spécifiées dans var_cat font bien partie du df, on ne vérifie pas le type
    # pour pouvoir faire l'analyse sur des variables discrètes ou encodées
    var_cat = [var for var in var_cat if var in list(data.columns)]
    # On vérifie le nombre max de modalités
    var_cat_too_large = [var for var in var_cat if data[var].nunique() > max_nunique]
    var_cat = [var for var in var_cat if data[var].nunique() <= max_nunique]
    # On exclut les variables qui comporte des NaN
    var_cat_nan = [var for var in var_cat if data[var].isnull().sum() > 0]
    var_cat = [var for var in var_cat if data[var].isnull().sum() == 0]

    if verbose:
        print(f"{len(var_cat)} variables qualitatives prises en compte : {var_cat}")
        print(f"{len(var_num)} variables numériques prises en compte : {var_num}")
        if len(var_not_in_df) > 0:
            print(
                f"{len(var_not_in_df)} variables exclues car ne font pas partie du df : {var_not_in_df}"
            )
        if len(var_cat_too_large) > 0:
            print(
                f"{len(var_cat_too_large)} variables exclues car ont trop de modalités (max={max_nunique}): {var_cat_too_large}"
            )
        if len(var_not_num) > 0:
            print(
                f"{len(var_not_num)} variables quantitatives exclues car ne sont pas numériques : {var_not_num}"
            )
            display(data[var_not_num].info())
        var_nan = var_num_nan + var_cat_nan
        if len(var_nan) > 0:
            print(
                f"{len(var_num)} variables exclues car comportent des NaN : {var_nan}"
            )

    # S'il reste des variables à croiser, on calcule la matrice :
    if len(var_num) > 0 and len(var_cat) > 0:
        # On met les variables numériques en colonnes et les variables qualitatives en lignes
        n_rows = len(var_cat)
        n_col = len(var_num)
        # On déclare un tableau vide
        corr_eta = [[np.nan] * n_col for i in range(n_rows)]
        # On remplit la matrice avec les eta_square
        for i, row in enumerate(sorted(var_cat)):
            for j, col in enumerate(sorted(var_num)):
                corr_eta[i][j] = eta_squared(data[row], data[col])
                # print(f"eta_square pour {row} x {col} = {corr_eta[i][j]}")

        # Si display_table est demandé, on construit un df avec la matrice et on l'affiche
        if display_table:
            df_eta = pd.DataFrame(
                corr_eta, index=sorted(var_cat), columns=sorted(var_num)
            )
            print("Rapports de corrélations ETA SQUARE")
            with pd.option_context("display.float_format", "{:.3f}".format):
                display(df_eta)

        # Avec la matrice on dessine la heatmap
        # fig, ax = plt.subplots(figsize=figsize)
        # layout = 'constrained' colle le titre de la fig aux subplot
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        # Régler l'espace que prend les subplots par rapport à la figure, plus top est proche de 1
        # Plus l'espace restant pour le titre de la figure est petit. Nécessite donc des réglages
        # de hauteur de fig.
        # fig.subplots_adjust(top=0.9)

        ax = sns.heatmap(
            corr_eta,
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".3f",
            square=True,
            cmap="Blues",
            # Dessine une ligne pour séparer les carrés
            linewidths=0.1,
            # Ecrit les étiquettes de l'axe des y et des x
            yticklabels=sorted(var_cat),
            xticklabels=sorted(var_num),
            # Colorbar : rétrécie en hauteur et en largeur
            # cbar_kws={"format": "%.2f", "fraction": 0.035},
            cbar_kws={"format": "%.2f", "fraction": 0.045},
            annot_kws={"fontsize": 10},
        )

        # Pour la taille des annotations par rapport à la taille de cellules:
        # essayer annot_kws={"size": 35 / np.sqrt(len(corrmat))},

        # On positionne les ticks sur les axes et on diminue la taille de police des labels
        ax.tick_params(
            axis="both",
            left=True,
            bottom=True,
            labelsize=10,
            gridOn=False,
            grid_alpha=0.3,
            color="grey",
        )

        # Titre du graphique et de la figure
        fig.suptitle(
            "Rapports de corrélation Eta squared\nSCE/SCT = variance inter / variance totale",
            verticalalignment="center",
            fontsize=14,
        )
        # title ='Rapports de corrélation Eta squared : SCE/SCT\nSCE (Variation interclasse / Sum of squares of the model)\nSCT (variation totale / Total sum squares)'
        title = "SCE (Variation interclasse / Sum of squares of the model)\nSCT (variation totale / Total sum squares)"

        # ax.set_title(label=title, fontsize=10, loc="center", y=2.0, pad=30)
        # ax.set_title(label=title, fontsize=10, loc="center")

        # tight_layout incompatible avec la cbar
        # plt.tight_layout()
    # Si il n'y a pas de variables valides à croiser
    else:
        corr_eta = None
        print("Pas assez de variables valides pour calculer la matrice")
    # return corr_eta
    # On remet l'affichage des warnings à la normale
    # warnings.resetwarnings()
    return


def symmetrical_colormap(cmap_settings=("Blues", None), new_name=None):
    """Crée une colormap symétrique (pour les matrices de corrélation) en concaténant une cmap simple qui va d'une couleur vers le blanc. (ex : "Blues")
    avec son inverse

    Args:
        cmap_settings (_type_): _description_
        new_name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # Crée la colormap simple qui va de la couleur vers le blanc
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_" + cmap_settings[0]  # ex: 'sym_Blues'

    # Finesse de définition de la cmap
    n = 128

    # On récupère la liste des couleurs de la cmap simple qui feront la partie droite de la nouvelle cmpa
    colors_r = cmap(np.linspace(0, 1, n))
    # On renverse l'ordre des couleurs pour faire la partie gauche de la nouvelle cmpa
    colors_l = colors_r[::-1]

    # On combine partie gauche et droite pour faire la nouvelle cmap
    colors = np.vstack((colors_l, colors_r))
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap


# Donn un effet barré à un texte. Ne fonctionne pas dans un graphique
def strike(text):
    result = ""
    for c in text:
        result = result + c + "\u0336"
    return result


# [to do] Regrouper corr_sorted_map et corr_map et mettre en paramètre l'otion de tri (alphabétique ou par corr)
def corr_sorted_map(
    data, to_process, method="pearson", figsize=(30, 30), to_mark=[], triangul=False
):
    """Dessine la heatmap des corrélations en regoupant les varaibles les plus corrélées entre elles
    grâce à un dendrogramme non tracé

    Args:
        data (DataFrame): Df contenant les données
        to_process (list): Liste de variables dont veut les corrélations
        method (str, optional): type de corrélations à claculer (pearson, spearman ou kendall). Defaults to 'pearson'.
        figsize (tuple, optional): Taille de la heatmap. Defaults to (30,30).
        to_mark (list, optional): Liste de variables à distinguer des autres (ex : targets). Defaults to [].
        triangul (bool, optional) : Si vrai triangularise la matrice

    Returns:
        array: matrice des corrélations triées
    """
    # inspiré de :
    # https://kobia.fr/automatiser-la-reduction-des-correlations-par-clustering/
    if method == "pearson" or method == "kendall" or method == "spearman":
        corr = data[to_process].corr(method=method)
        # Correction pour avoir une matrice parfaitement symetrique
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr.to_numpy(), 1.0)

        # Définition de la matrice de distance à partir des corrélations
        # 2 variables très corrélées 1 ou -1 ont une distance de 0
        # 2 variables pas corrélées du tout (corr = 0) ont une dist max égale à 1
        dist = squareform(1 - abs(corr))

        # Réalisation du clustering hiérarchique
        corr_linkage = hierarchy.complete(dist)

        # Récupération du dendrogramme
        dendro = hierarchy.dendrogram(
            corr_linkage, orientation="right", labels=to_process, no_plot=True
        )

        # Construction d'une palette symétrique car corrélation en valeur absolue pour similiarités et élimination
        sym_cmap = symmetrical_colormap(cmap_settings=("Blues", None), new_name=None)

        # Matrice des corrélations réordonnées en fonction des regroupements par corrélations
        sorted_corr = data[dendro["ivl"]].corr(method=method)

        if triangul:
            mask = np.zeros_like(sorted_corr)
            mask[np.triu_indices_from(mask)] = True

        else:
            mask = None

        fig, ax = plt.subplots(figsize=figsize)
        plt.grid(visible=False)
        ax = sns.heatmap(
            sorted_corr,
            vmin=-1,
            vmax=1,
            mask=mask,
            annot=True,
            fmt=".2f",
            square=True,
            # cmap='BrBG',
            cmap=sym_cmap,
            # Dessine une ligne pour séparer les carrés
            linewidths=0.1,
            # Ecrit les étiquettes de l'axe des y
            yticklabels=True,
            # Colorbar : rétrécie en hauteur et en largeur
            cbar_kws={"format": "%.2f", "fraction": 0.035},
            annot_kws={"fontsize": 10},
        )

        # Pour la taille des annotations par rapport à la taille de cellules:
        # essayer annot_kws={"size": 35 / np.sqrt(len(corrmat))},

        # S'il ya des variables à distinguer
        # On récupère le texte des labels des axes et on colorie celui qui correspond
        if to_mark is not None and len(to_mark) > 0:
            [
                t.set_color("red")
                for t in ax.yaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]
            [
                t.set_fontweight("bold")
                for t in ax.yaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]
            [
                t.set_color("red")
                for t in ax.xaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]
            [
                t.set_fontweight("bold")
                for t in ax.xaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]

        # Titre du graphique et des axes
        # title =''
        title = f"Matrice des corrélations - Coeff. de {method.capitalize()}\nVariables regroupées selon leurs corrélations 2 à 2"
        if method == "pearson":
            title = "Matrice de corrélations\n Coeff. de Pearson"
        if method == "spearman":
            title = "Matrice de corrélation\n Coefficients de Spearman" ""
        ax.set_title(label=title, fontsize=14)
        # ax.set_ylabel('')
    return sorted_corr


def corr_cluster_map(
    data,
    to_process,
    method="pearson",
    figsize_dendro=(10, 15),
    figsize_map=(30, 30),
    to_mark=[],
    threshold=0.05,
):
    """Dessine un dendogramme qui regroupe les variables les plus corrélées entre elles et
    la heatmap des corrélations en regoupant les varaibles les plus corrélées

    Args:
        data (DataFrame): Df contenant les données
        to_process (list): Liste de variables dont veut les corrélations
        method (str, optional): type de corrélations à claculer (pearson, spearman ou kendall). Defaults to 'pearson'.
        figsize_dendro (tuple, optional): Taille du dendrogramme. Defaults to (10,15).
        figsize_map (tuple, optional): Taille de la heatmap. Defaults to (30,30).
        to_mark (list, optional): Liste de variables à distinguer des autres (ex : targets). Defaults to [].

    Returns:
        list: Liste des variables à supprimer au sein des clusters car trop corrélées
    """
    # inspiré de :
    # https://kobia.fr/automatiser-la-reduction-des-correlations-par-clustering/
    if method == "pearson" or method == "kendall" or method == "spearman":
        corr = data[to_process].corr(method=method)
        # Correction pour avoir une matrice parfaitement symetrique
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr.to_numpy(), 1.0)

        # Définition de la matrice de distance à partir des corrélations
        # 2 variables très corrélées 1 ou -1 ont une distance de 0
        # 2 variables pas corrélées du tout (corr = 0) ont une dist max égale à 1
        dist = squareform(1 - abs(corr))

        # Réalisation du clustering hiérarchique
        # corr_linkage = hierarchy.complete(dist)
        corr_linkage = hierarchy.complete(dist)

        # Récupération du dendrogramme
        fig_dendro, ax_dendro = plt.subplots(figsize=figsize_dendro)
        title_dendro = (
            "Regroupement des variables par corrélations (" + method.capitalize() + ")"
        )
        ax_dendro.set_title(label=title_dendro, fontsize=14)
        ax_dendro.set_xlabel("Métrique : 1-|corr|")
        ax_dendro.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

        dendro = hierarchy.dendrogram(
            corr_linkage,
            orientation="right",
            ax=ax_dendro,
            labels=to_process,
        )
        # Ligne pour tracer le seuil
        ax_dendro.vlines(
            threshold,
            ymin=ax_dendro.get_ylim()[0],
            ymax=ax_dendro.get_ylim()[1],
            color="black",
            ls="--",
            lw=1,
            label="Seuil",
        )
        # S'il ya des variables à distinguer,
        # On récupère le texte des labels de l'axe des y et on colorie celui qui correspond
        if to_mark is not None and len(to_mark) > 0:
            [
                t.set_color("red")
                for t in ax_dendro.yaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]
            [
                t.set_fontweight("bold")
                for t in ax_dendro.yaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]

        # Varaibles à dropper
        # Définition du seuil de corrélation
        # threshold_for_cluster_creation = threshold

        # Récupération des clusters à partir de la hiérarchie
        cluster_ids = hierarchy.fcluster(corr_linkage, threshold, criterion="distance")
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        clusters = [list(v) for v in cluster_id_to_feature_ids.values() if len(v) > 1]
        clusters_col = [list(data[to_process].columns[v]) for v in clusters]

        # On ne conserve que la première variable de chaque cluster, les autres sont retirées
        dropped_features = [data[to_process].columns[v[1:]] for v in clusters]
        dropped_features = [item for sublist in dropped_features for item in sublist]
        print("Variables à supprimer car trop corrélées :", dropped_features)
        print("ordre des var dendro dendro['leaves']", dendro["leaves"])
        print("Longueur dendro['leaves']", len(dendro["leaves"]))
        print("nom des var dendro dendro['ivl']", dendro["ivl"])
        print("Longueur dendro['ivl']", len(dendro["ivl"]))

        # On colorie en gris les labels des varaibles à dropper :
        if dropped_features is not None and len(dropped_features) > 0:
            [
                t.set_color("grey")
                for t in ax_dendro.yaxis.get_ticklabels()
                if t.get_text() in dropped_features
            ]
            # Pas moyen de rayer les labels des variables à éliminer
            # [t.set_path_effects([pe.StrikeThrough()]) for t in ax_dendro.yaxis.get_ticklabels() if t.get_text() in dropped_features]
            # Ne fonctionne pas non plus ci après
            """for t in ax_dendro.yaxis.get_ticklabels():
                if t.get_text() in dropped_features:
                    stroke_label = strike(t.get_text())
                    t.set_text(stroke_label)"""

        # Construction d'une palette symétrique car corrélation en valeur absolue pour similiarités et élimination
        sym_cmap = symmetrical_colormap(cmap_settings=("Blues", None), new_name=None)

        # Matrice des corrélations réordonnées en fonction des regroupements par corrélations
        sorted_corr = data[dendro["ivl"]].corr(method=method)
        fig, ax = plt.subplots(figsize=figsize_map)
        ax = sns.heatmap(
            sorted_corr,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            square=True,
            # cmap='BrBG',
            cmap=sym_cmap,
            # Dessine une ligne pour séparer les carrés
            linewidths=0.1,
            # Ecrit les étiquettes de l'axe des y
            yticklabels=True,
            # Colorbar : rétrécie en hauteur et en largeur
            cbar_kws={"format": "%.2f", "fraction": 0.035},
            annot_kws={"fontsize": 10},
        )

        # Pour la taille des annotations par rapport à la taille de cellules:
        # essayer annot_kws={"size": 35 / np.sqrt(len(corrmat))},

        # S'il ya des variables à distinguer (plus que à éliminer),
        # On récupère le texte des labels de l'axe des y et on colorie celui qui correspond
        if to_mark is not None and len(to_mark) > 0:
            [
                t.set_color("red")
                for t in ax.yaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]
            [
                t.set_fontweight("bold")
                for t in ax_dendro.yaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]

        # On positionne les ticks sur les axes et on diminue la taille de police des labels
        ax.tick_params(
            axis="both",
            left=True,
            bottom=True,
            labelsize=10,
            gridOn=False,
            grid_alpha=0.3,
            color="grey",
        )

        # Titre du graphique et des axes
        # title =''
        title = f"Matrice des corrélations - Coeff. de {method.capitalize()}\nVariables regroupées selon leurs corrélations 2 à 2"
        """if method == 'pearson':
            title = "Matrice de corrélations\n Coeff. de Pearson"
        if method == 'spearman':
            title = "Matrice de corrélation\n Coefficients de Spearman"""
        ax.set_title(label=title, fontsize=14)
        # ax.set_ylabel('')

        """# On transforme la matrice des corrélations entre variables ordonnées en df
        # pour pouvoir s'en resservir (avec les libellés)
        df_corr= pd.DataFrame(sorted_corr, index=dendro['ivl'], columns=dendro['ivl'])"""
    return dropped_features


def corr_map(
    data,
    to_process,
    method="pearson",
    triangul=False,
    to_mark=[],
    display_table=False,
    figsize=(20, 20),
    color_sym_cmap="Blues",
):
    """Calcule et trace une matrice de corrélations. Les variables sont présentées dans l'ordre du DataFrame.

    Args:
        data (DataFrame): df contenant les données
        to_process (list): Liste de variables dont on veut les corrélations
        method (str, optional): Type de corrélations. Defaults to 'pearson'.
        triangul (bool, optional): Triangularise la matrice. Defaults to False.
        to_mark (list, optional): Liste de variables à distinguer des autres (rouge et gras). Defaults to [].
        display_table (bool, optional): Affiche les corrélations sous forme de df. Defaults to False.
        figsize (tuple, optional): Taille de la figure. Defaults to (20, 20).
        color_sym_cmap (str, optional): Réalise une palette symétrique à partir d'une couleur. Defaults to 'Blues'.

    Returns:
        Array: Matrice des corrélations
    """

    # Si analyse de var quantitatives
    # spearman : robuste aux outliers. Pour corrélations non linéaires mais de type fonctions monotones (préservent l'ordre)
    if method == "pearson" or method == "kendall" or method == "spearman":
        corr = data[sorted(to_process)].corr(method=method)

    # Si analyse bivariée de deux variables qualitatives
    elif method == "khi2":
        print("[to do]")
        # scipy.stats.contingency.expected_freq(observed)
        # Calcule les fréquences attendues si les var étaient indep à partir d'un tableau de contingence
        # scipy.stats.chi2_contingency(observed, correction=True, lambda_=None)
        # Fait le test du khi2 à partir d'un tableau de contingence. Callcule les freq avec methode ci-dessus
        # scipy.stats.contingency.crosstab(*args, levels=None, sparse=False)
        # Calcule le tableau de contingences

        # Comme le cours en imaginant que  to_process ne contient que 2 var:
        contingency_table = data[to_process].pivot_table(
            index=to_process[0],
            columns=to_process[1],
            aggfonc=len,
            margins=True,
            margins_name="Total",
        )

        # Equivallent du 2 dim avec scypy
        # from scipy.stats.contingency import crosstab (les param sont des listes)
        # contingency_table = crosstab(to_process[0].to_list(), to_process[1].to_list())
        # On peut mettre plus de deux param.
        # Le résultat est un array
        # Beaucoup d'autres possibilités. Mais j'ai l'impression que necessite de
        # transformer les var qual en numériques
    # Cas Anova
    else:
        print("to do")

    # On a calculé la matrice de corrélation
    if corr is not None:
        # Si demandé, on affiche la matrice de corrélation en df
        if display_table == True:
            with pd.option_context("display.float_format", "{:.2f}".format):
                display(corr)

        # On met une cmap par défaut
        cmap = "BrBG"
        # Si une cmpa symétrique est demandée, on la construit
        if color_sym_cmap is not None:
            cmap = symmetrical_colormap(
                cmap_settings=(color_sym_cmap, None), new_name=None
            )

        # Si la triangularisation est demandée, on affiche la heatmap de la matrice de corrélation
        # Afficher que en dessous de la diagonale
        if triangul:
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
        else:
            mask = None

        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(
            corr,
            mask=mask,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            square=True,
            # cmap='Spectral',
            # cmap='seismic_r',
            # cmap='bwr_r',
            cmap=cmap,
            # Dessine une ligne pour séparer les carrés
            linewidths=0.1,
            # Ecrit les étiquettes de l'axe des y
            yticklabels=True,
            # Colorbar : rétrécie en hauteur et en largeur
            cbar_kws={"format": "%.2f", "fraction": 0.035},
            annot_kws={"fontsize": 10},
        )

        # Pour la taille des annotations par rapport à la taille de cellules:
        # essayer annot_kws={"size": 35 / np.sqrt(len(corrmat))},

        # S'il ya des variables à distinguer,
        # On récupère le texte des labels de l'axe des y et on colorie celui qui correspond
        if to_mark is not None and len(to_mark) > 0:
            [
                t.set_color("red")
                for t in ax.yaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]
            [
                t.set_fontweight("bold")
                for t in ax.yaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]
            [
                t.set_color("red")
                for t in ax.xaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]
            [
                t.set_fontweight("bold")
                for t in ax.xaxis.get_ticklabels()
                if t.get_text() in to_mark
            ]

        # On positionne les ticks sur les axes et on diminue la taille de police des labels
        ax.tick_params(
            axis="both",
            left=True,
            bottom=True,
            labelsize=10,
            gridOn=False,
            grid_alpha=0.3,
            color="grey",
        )

        # Titre du graphique et des axes
        title = ""
        if method == "pearson":
            title = "Matrice de corrélations\n Coeff. de Pearson"
        if method == "spearman":
            title = "Matrice de corrélation\n Coefficients de Spearman"
        ax.set_title(label=title, fontsize=14)
        # ax.set_ylabel('')

    return corr


def grid_scatter_features_cross_target(
    data,
    features,
    target,
    figsize=None,
    n_h=3,
    fontsize=9,
    drop_xticks_in_features=[],
    log_scale=False,
):
    """Trace une figure avec des sous-plots. Un sous-plot (scatter) par feature qui est croisé avec la target.

    Args:
        data (DataFrame): df contenant les données
        features (list): Features dont on veut étudier la corrélation avec la target
        target (str ou list): Le nom de la variable target à croiser avec chaque feature
        figsize (_type_, optional): Largeur et hauteur de la figure. Claculée si non spécifiée. Defaults to None.
        n_h (int, optional): Nombre de plots à placer horizontalement. Defaults to 3.
        fontsize (int, optional): Taille de police pour les ticks. La police pour les titres des axes est celle-ci + 1. Defaults to 9.
        drop_xticks_in_features (list, optional): Elimine les ticks en trop sur l'axe des x pour les features spécifiées. Defaults to [].
        log_scale (bool, optional): Echelle logarithmique. Defaults to False.
    """
    environ = "\u2248"
    palette = "muted"
    colors = sns.color_palette(palette)[0:5]
    # Transparence de la légende
    legend_alpha = 0.4
    # Nombre max de figures autorisé
    max_n_plots = 36
    # Pour afficher une légende si l'échelle logarithmique est demandée mais que celle-ci n'a pas été possible
    display_legend = False
    # On récupère deux couleurs de la palette en cours pour différencier les plots en échelle log et non log
    color_log = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    color_not_log = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]

    # Si target est une liste, on prend le premier élement
    if type(target) == list:
        target_name = target[0]
    else:
        target_name = target

    n_plots = len(features)
    # Si le nombre de plots est trop grand on sort
    if n_plots > max_n_plots:
        print(f"Nombre total de features {n_plots} dépasse le maximum {max_n_plots}")
        return
    print("Nombre total de plots :", n_plots)

    # Si la logscale est demandée mais que la target contient des valeurs négatives, l'échelle logarithmique ne sera pas appliquée
    if log_scale:
        if data[target_name].min() <= 0:
            log_scale = False
            print(
                "La target contient des valeurs négatives ou nulles, l'échelle logarithmique n'est pas appliquée"
            )

    # On calcule le nombre de plots en vertical en fonction du nombre de plots en hrizontal
    n_v = n_plots // n_h
    remainder = n_plots % n_h
    # Si le reste de la division entière du nombre total de plot par le nombre en largeur est existant, on ajoute une ligne de plots (qui ne sera pas complète)
    n_full_rows = n_v
    if remainder > 0:
        n_v += 1

    # Si la taille de la figure n'est pas précisée, on approxime grossièrement en fonction de n_h et n_v
    if figsize is None:
        width_subplot = 4.3
        height_subplot = 3.6
        # On ajoute des marges pour les titres etc.
        width = width_subplot * n_h + 0.1
        height = height_subplot * n_v + 0.2
        figsize = (width, height)
        print("Taille estimée de la figure' :", figsize)

    # On crée la figure
    fig, axes = plt.subplots(n_v, n_h, figsize=figsize)
    print("axes.shape", axes.shape)

    if log_scale:
        title = f"Croisement de variables quantitatives avec la target {target}\nEchelle logarithmique pour les distributions positives\n"
    else:
        title = f"Croisement de variables quantitatives avec la target {target}\n"
    fig.suptitle(title)

    plot_idx = 0
    for i in range(n_v):
        # S'il s'agit d'une ligne de plot qui n'est pas complète on limite le nombre de plots à dessiner horizontalement
        if i < n_full_rows:
            limit_h = n_h
        else:
            limit_h = remainder

        for j in range(limit_h):
            # Selon le nombre de lignes / colonnes de la figure, axes renvoyé n'aura pas la même shape
            if n_h == 1 and n_v == 1:
                ax = axes
            else:
                if n_h == 1:
                    axes = axes.reshape(n_v, 1)
                if n_v == 1:
                    axes = axes.reshape(1, n_h)
                ax = axes[i, j]

            """*********************************************************************************************************
            Dessin scatterplot
            *********************************************************************************************************"""
            feature = features[plot_idx]
            vars = [feature, target_name]

            # Echelle logarithmique si demandé :
            if log_scale:
                if data[feature].min() >= 0:
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    color = color_log
                else:
                    color = color_not_log
                    display_legend = True
            else:
                color = None

            # Taille de police des ticks pour les axes
            # Paramètres des axes
            ax.tick_params(axis="x", labelsize=fontsize)

            ax.tick_params(axis="y", labelsize=fontsize)
            ax.xaxis.label.set_size(fontsize + 1)
            # ax.xaxis.label.set_weight('bold')
            ax.yaxis.label.set_size(fontsize + 1)

            scatter = data[vars].plot(
                kind="scatter",
                x=feature,
                y=target_name,
                alpha=0.6,
                ax=ax,
                c=color,
                # cmap=cmap,
                # colorbar=False
            )

            # Si drop_x_ticks est demandé pour la variable en abcisse on ne garde que le deuxième et l'avant dernier tick sur l'axe des x
            if feature in drop_xticks_in_features:
                # On récupère les ticks actuels sur l'axe des x
                # (sca : interface pour avoir le sublpolt en cours, sinon tous les suplots)
                plt.sca(ax)
                ticks = plt.xticks()[0]
                # On affiche que deux ticks de manière à voir l'étendue
                plt.xticks([ticks[1], ticks[-2]])

            """# On réduit la taille et la police de la colorbar
            colorbar = plt.colorbar(scatter.collections[0], ax=ax, shrink=0.8)
            colorbar = scatter.collections[0].colorbar
            colorbar.ax.tick_params(labelsize=fontsize)
            colorbar.ax.set_ylabel(target, fontsize=fontsize)"""

            # On affiche les R2
            # Calcul du coefficient de corrélation de Pearson
            spearman_corr, _ = spearmanr(data[feature], data[target_name])
            pearson_corr = data[feature].corr(data[target_name])

            # On récupère les handles de la légende déjà créés, cela doit être vide
            handles, legend_labels = ax.get_legend_handles_labels()

            # On ajoute 2 handle transparent pour les deux  R2
            # print("data[feature_row]", data[feature_row])
            # print("corr", data[feature_col, target_sup].corr())
            legend_labels.append(f"Pearson = {pearson_corr:.2f}")
            legend_labels.append(f"Spearman = {spearman_corr:.2f}")
            handles.append(
                Line2D([], [], label=f"Pearson = {pearson_corr}", alpha=0),
            )
            handles.append(Line2D([], [], label=f"Spearman = {spearman_corr}", alpha=0))
            ax.legend(
                handles, legend_labels, framealpha=legend_alpha, fontsize=fontsize
            )

            plot_idx += 1
    # Si la dernière lignes de plots n'est pas complète on supprime les plots videsz (car sinon on voit la grille)
    if remainder > 0:
        n_empty_plots = n_h - remainder
        for k in range(n_empty_plots):
            fig.delaxes(axes[n_v - 1, n_h - 1 - k])

    # Légende
    if display_legend:
        fig.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    color="w",
                    markerfacecolor=color_log,
                    markersize=10,
                ),
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    color="w",
                    markerfacecolor=color_not_log,
                    markersize=10,
                ),
            ],
            labels=["échelle LOG", "échelle normale"],
        )

    plt.tight_layout()
    plt.show()
    return


def grid_scatter_colored_by_2_targets(
    data,
    features,
    target_inf,
    target_sup,
    figsize=None,
    n=3,
    cmap_sup="cool_r",
    cmap_inf="crest",
    fontsize=9,
    drop_xticks_in_features=[],
):
    """Trace dans une grille de sous plot des nuages de points croisant des features 2 à 2 colorés par une target sur la partie supérieure à la diagonale et par une autre
    target sur la partie inférieure. La diagonale est constituée d'histogrammes.
    Affichage des r2 pearson et spearman sur les nuages.
    La figure peut être découpée (nombre de plots en largeurs : n) mais les bords de la matrice ne sont pas regroupés entre eux.

    Args:
        data (DataFrame): df contenant les données
        features (list): liste des variables numériques à plotter en nuages de points
        target_inf (str): Nom de la target servant à colorer les nuages de points sous la diagonale
        target_sup (str): Nom de la target servant à colorer les nuages de points au dessus de la diagonale
        figsize (_type_, optional): Largeur et hauteur de la figure. Si None elle est calculée. Defaults to None.
        n (int, optional): Nombre de plots en largeur. Defaults to 3.
        cmap_sup (str, optional): cmap servant à colorer les nuages sur la partie supérieure. Defaults to "cool_r".
        cmap_inf (str, optional): cmap servant à colorer les nuages sur la partie inférieure. Defaults to "crest".
        fontsize (int, optional): Taille de police des ticks et légende. La taille des titres des axes est celle-ci + 1. Defaults to 9.
        drop_xticks_in_features (list, optional): Ne garde que le 2ème et l'avant dernier ticks de l'axe des x sur les variables spécifiées. (les labels se chevauchent fréquemment). Defaults to [].
    """
    environ = "\u2248"
    palette = "muted"
    colors = sns.color_palette(palette)[0:5]
    # Transparence de la légende
    legend_alpha = 0.4
    # Nombre max de figures autorisé
    max_n_figures = 16

    # Nombre de figures complètement remplies horizontalement
    n_full_figures_h = len(features) // n
    # Nombre de figures complètement remplies verticalement
    n_full_figures_v = n_full_figures_h

    # Nombre de features restantes dans le sens horizontal
    remainder_h = len(features) % n
    # Nombre de features restantes dans le sens vertical
    remainder_v = remainder_h

    # Nombre de figures dans le sens horizontal :
    if remainder_h == 0:
        n_figures_h = n_full_figures_h
    else:
        n_figures_h = n_full_figures_h + 1
    # Nombre de figures dans le sens vertical
    n_figures_v = n_figures_h

    # Nombre total de figures
    if remainder_h == 0:
        n_figures = n_full_figures_h * n_full_figures_h
    else:
        n_figures = (
            n_full_figures_h * n_full_figures_h
            + n_full_figures_h
            + n_full_figures_v
            + 1
        )

    # Si le nombre de figures est trop grand on sort
    if n_figures > max_n_figures:
        print(f"Nombre total de figures {n_figures} dépasse le maximum {max_n_figures}")
        return
    print("Nombre total de figures :", n_figures)

    # Si la taille de la figure n'est pas précisée, on approxime grossièrement en fonction de n
    if figsize is None:
        width_subplot = 4.3
        height_subplot = 3.6
        # On ajoute des marges pour les titres etc.
        width = width_subplot * n + 0.1
        height = height_subplot * n + 0.2
        figsize = (width, height)
        print("Taille estimée des figures 'pleines' :", figsize)
        print("Nombre de figures pleines", n_full_figures_h * n_full_figures_v)
        print("Reste = ", remainder_h)

    # Localisation de la figure par rapport à la diagonale
    location_figure = "diag"

    idx_fig = 0
    for k_h in range(n_figures_h):
        # print("k_h =", k_h)
        for k_v in range(n_figures_v):
            # print(f"k_h = {k_h} et k_v = {k_v}")
            # Si la figure est pleine

            if k_h < n_full_figures_h and k_v < n_full_figures_v:
                # print("fig pleine")
                """features_h = features[k_h * n_full_figures_h : k_h * n_full_figures_h + n]
                features_v = features[k_v * n_full_figures_v : k_v * n_full_figures_v + n]
                """
                features_h = features[k_h * n : k_h * n + n]
                features_v = features[k_v * n : k_v * n + n]

                fig, axes = plt.subplots(n, n, figsize=figsize)

            # Si la figure n'est pas pleine (on est dans le reste), alors on redimensionne la figure
            else:
                # print("fig restante")
                # Cas de la figure pleine en lignes (verticalement) mais pas en lignes => Au dessus (= à droite) de la diag
                if k_v < k_h:
                    # print("Droite de la diag")
                    features_h = features[k_h * n : k_h * n + remainder_h]
                    features_v = features[k_v * n : k_v * n + n]
                    # print(f"étape {(k_h + 1) * (k_v + 1)}/{n_figures} - features horizontales : {features_h} features verticales : {features_v}")
                    width, height = figsize
                    margin_width = 0.02 * width
                    width = width * remainder_h / n + margin_width
                    fig, axes = plt.subplots(n, remainder_h, figsize=(width, height))
                    if remainder_h == 1:
                        axes = axes.reshape(n, 1)

                # Cas de la figure pleine en colonnes (horizontalement) mais pas en lignes => Au dessous (= à gauche LEFT) de la diag
                if k_v > k_h:
                    # print("Gauche de la diag")
                    features_h = features[k_h * n : k_h * n + n]
                    features_v = features[k_v * n : k_v * n + remainder_v]
                    print("features_h", features_h)
                    print("features_v", features_v)
                    width, height = figsize
                    margin_height = 0.02 * height
                    height = height * remainder_v / n + margin_height
                    fig, axes = plt.subplots(remainder_v, n, figsize=(width, height))
                    if remainder_v == 1:
                        axes = axes.reshape(1, n)
                        # print("Le remainder V est 1")
                        # print("Axes.shape", axes.shape)
                # Cas de la dernière figure ni pleine en lignes (verticalement) ni en col => Sur la diag
                elif k_v == k_h:
                    # print("Diag")
                    features_h = features[k_h * n : k_h * n + remainder_h]
                    features_v = features[k_v * n : k_v * n + remainder_v]
                    # print(f"étape {k_h*(k_v + 1) + 1}/{n_figures} - features horizontales : {features_h} features verticales : {features_v}")
                    width = figsize[0]
                    margin_width = 0.02 * width
                    width = width * remainder_h / n + margin_width
                    height = figsize[1]
                    margin_height = 0.02 * height
                    height = height * remainder_v / n + margin_height
                    fig, axes = plt.subplots(
                        remainder_v, remainder_h, figsize=(width, height)
                    )

            # On localise la sous-matrice par rapport à la diagonale
            if k_h == k_v:
                location_figure = "diag"
                targets = [target_inf, target_sup]
            elif k_h > k_v:
                location_figure = "right"
                targets = [target_sup]
            else:
                location_figure = "left"
                targets = [target_inf]

            idx_fig += 1
            fig.suptitle(
                f"ETUDE BIVARIEE Fig. {idx_fig}/{n_figures}\n\nFeatures : {features_h}\n   avec : {features_v}\nNuages colorés par {targets}\n"
            )
            print(
                f"\nétape {idx_fig}/{n_figures} k_h = {k_h}, k_v = {k_v}\nfeatures horizontales : {features_h}\nfeatures verticales : {features_v}"
            )
            print(
                f"Localisation de la figure par rapport à la diagonale : {location_figure}"
            )
            for j, feature_col in enumerate(features_h):
                # print("features_h", features_h)
                for i, feature_row in enumerate(features_v):
                    # print("features_v", features_v)
                    # Si il ne reste qu'un sous-plot dans la toute dernière figure, subplots() ne renvoie pas un tableau d'ax mais un seul ax
                    if len(features_h) == 1 and len(features_v) == 1:
                        ax = axes
                    else:
                        # print("feature_col", feature_col, "i", i, "et j", j)
                        ax = axes[i, j]

                    # Taille de police des ticks pour les axes
                    # Paramètres des axes
                    ax.tick_params(axis="x", labelsize=fontsize)

                    ax.tick_params(axis="y", labelsize=fontsize)
                    ax.xaxis.label.set_size(fontsize + 1)
                    # ax.xaxis.label.set_weight('bold')
                    ax.yaxis.label.set_size(fontsize + 1)
                    # ax.yaxis.label.set_weight('bold')

                    """*********************************************************************************************************
                    Dessin Histogramme
                    *********************************************************************************************************"""
                    ############################ DIAGONALE HISTOGRAMME
                    if location_figure == "diag" and j == i:
                        # print("Histogramme")
                        # ax.set_title("ligne " + str(i) +" col " + str(j) + " DIAG")
                        sns.histplot(data=data[feature_row], bins=40, kde=True, ax=ax)

                        # On récupère la valeur max de l'axe des y
                        ymax = np.ceil(ax.get_ylim()[1])

                        # Paramètres des axes
                        ax.set_ylabel("Effectif")

                        # On trace la moyenne, la médiane et le mode
                        ax.vlines(
                            data[feature_row].mean(),
                            ymax=ymax,
                            ymin=0,
                            color=colors[1],
                            ls="-.",
                            lw=2,
                            label=f"Moyenne"
                            + f" {environ} "
                            + f"{data[feature_row].mean():.0f}",
                        )
                        ax.vlines(
                            data[feature_row].median(),
                            ymax=ymax,
                            ymin=0,
                            color=colors[4],
                            ls="-",
                            lw=2,
                            label=f"Médiane"
                            + f" {environ} "
                            + f"{data[feature_row].median():.0f}",
                        )

                        # On crée un graphisme transparent pour la skewness afin de l'aficher dans la légende sans tracer de courbe correspondante
                        # On récupère les handles de la légende déjà créés pour la moyenne et la médiane
                        handles, legend_labels = ax.get_legend_handles_labels()
                        # On ajoute un handle transparent pour la skewness
                        legend_labels.append(
                            f"Skewness = {data[feature_row].skew():.1f}"
                        )
                        handles.append(
                            Line2D(
                                [],
                                [],
                                label=f"skewness = {data[feature_row].skew():.1f}",
                                alpha=0,
                            )
                        )
                        ax.legend(
                            handles,
                            legend_labels,
                            framealpha=legend_alpha,
                            fontsize=fontsize,
                        )
                    ############################ EN DEHORS DE DIAGONALE => SCATTERPLOT
                    else:
                        # Si triangle supérieur de la matrice globale => NUAGE COLOR SUP par TARGET SUP
                        if location_figure == "right" or (
                            location_figure == "diag" and j > i
                        ):
                            cmap = cmap_sup
                            target = target_sup
                        # Si triangle inférieur de la matrice globale => NUAGE COLOR INF par TARGET INF
                        else:
                            cmap = cmap_inf
                            target = target_inf

                        """*********************************************************************************************************
                        Dessin scatterplot
                        *********************************************************************************************************"""
                        # axes[i, j].set_title("ligne " + str(i) +" col " + str(j) + " SUP")
                        vars = [feature_row, feature_col, target]
                        # Limite le nombre de ticks sur l'axe des x
                        scatter = data[vars].plot(
                            kind="scatter",
                            x=feature_row,
                            y=feature_col,
                            alpha=0.6,
                            ax=ax,
                            c=target,
                            cmap=cmap,
                            colorbar=False,
                        )

                        # Si drop_x_ticks est demandé pour la variable en abcisse on ne garde que le deuxième et l'avant dernier tick sur l'axe des x
                        if feature_row in drop_xticks_in_features:
                            # On récupère les ticks actuels sur l'axe des x
                            # (sca : interface pour avoir le sublpolt en cours, sinon tous les suplots)
                            plt.sca(ax)
                            ticks = plt.xticks()[0]
                            # On affiche que deux ticks de manière à voir l'étendue
                            plt.xticks([ticks[1], ticks[-2]])

                        # On réduit la taille et la police de la colorbar
                        colorbar = plt.colorbar(
                            scatter.collections[0], ax=ax, shrink=0.8
                        )
                        colorbar = scatter.collections[0].colorbar
                        colorbar.ax.tick_params(labelsize=fontsize)
                        colorbar.ax.set_ylabel(target, fontsize=fontsize)

                        # On affiche les R2
                        # Calcul du coefficient de corrélation de Pearson
                        # pearson_corr = np.corrcoef(data[feature_row], data[feature_col])
                        # pearman_corr = data[[feature_row, feature_col]].corr(method="spearman")[0]
                        spearman_corr, _ = spearmanr(
                            data[feature_row], data[feature_col]
                        )
                        # print("spearman_corr", spearman_corr)
                        pearson_corr = data[feature_row].corr(data[feature_col])
                        # print("pearson_corr", pearson_corr)

                        # plt.annotate(f"Pearson {pearson_corr:.2f}", (0.5, 0.9), xycoords='axes fraction', ha='center')

                        # On récupère les handles de la légende déjà créés, cela doit être vide
                        handles, legend_labels = ax.get_legend_handles_labels()

                        # On ajoute 2 handle transparent pour les deux  R2
                        # print("data[feature_row]", data[feature_row])
                        # print("corr", data[feature_col, target_sup].corr())
                        legend_labels.append(f"Pearson = {pearson_corr:.2f}")
                        legend_labels.append(f"Spearman = {spearman_corr:.2f}")
                        handles.append(
                            Line2D([], [], label=f"Pearson = {pearson_corr}", alpha=0),
                        )
                        handles.append(
                            Line2D([], [], label=f"Spearman = {spearman_corr}", alpha=0)
                        )
                        ax.legend(
                            handles,
                            legend_labels,
                            framealpha=legend_alpha,
                            fontsize=fontsize,
                        )

            plt.tight_layout()
            plt.show()
    return


def plot_scatter_3d(df, x_y_z, hue=None, figsize=(9, 7), title=None):
    marker = "o"
    alpha = 0.4
    size = 5
    fontsize_title = 16
    # margin_z = 5

    fig = plt.figure(layout="constrained", figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    n_colors = 1
    if hue:
        n_colors = df[hue].nunique()
    # On récupère les n premières couleurs de la palette en cours
    colors = get_palette_colors(n_colors)

    # Si on croise avec une variable catégorielle
    if hue:
        for i, category in enumerate(df[hue].unique()):
            ax.scatter(
                df.loc[df[hue] == category, x_y_z[0]],
                df.loc[df[hue] == category, x_y_z[1]],
                df.loc[df[hue] == category, x_y_z[2]],
                color=colors[i],
                label=f"{category}",
            )
    # Si on ne croise pas avec une variable catégorielle
    else:
        ax.scatter(
            df[x_y_z[0]],
            df[x_y_z[1]],
            df[x_y_z[2]],
            color=colors[0],
        )

    ax.view_init(
        elev=20, azim=20
    )  # Par exemple, 20 degrés d'élévation et 30 degrés d'azimut

    ax.set_xlabel(x_y_z[0])
    ax.set_ylabel(x_y_z[1])
    ax.set_zlabel(x_y_z[2])
    ax.set_title(title, fontsize=fontsize_title)
    # ax.margins(z=margin_z)

    if hue:
        plt.legend(title=hue)

    # plt.tight_layout()
    return


def add_scatter_3d(
    fig, row_num, col_num, subplot_num, df, x_y_z, hue=None, title=None, legend=False
):
    marker = "o"
    alpha = 0.4
    size = 5
    fontsize_title = 12
    # margin_z = 5

    # fig = plt.figure(layout="constrained", figsize=figsize)
    ax = fig.add_subplot(row_num, col_num, subplot_num, projection="3d")

    n_colors = 1
    if hue:
        n_colors = df[hue].nunique()
    # On récupère les n premières couleurs de la palette en cours
    colors = get_palette_colors(n_colors)

    # Si on croise avec une variable catégorielle
    if hue:
        for i, category in enumerate(df[hue].unique()):
            ax.scatter(
                df.loc[df[hue] == category, x_y_z[0]],
                df.loc[df[hue] == category, x_y_z[1]],
                df.loc[df[hue] == category, x_y_z[2]],
                color=colors[i],
                label=f"{category}",
            )
    # Si on ne croise pas avec une variable catégorielle
    else:
        ax.scatter(
            df[x_y_z[0]],
            df[x_y_z[1]],
            df[x_y_z[2]],
            color=colors[0],
        )

    ax.view_init(
        elev=20, azim=20
    )  # Par exemple, 20 degrés d'élévation et 30 degrés d'azimut

    ax.set_xlabel(x_y_z[0])
    ax.set_ylabel(x_y_z[1])
    ax.set_zlabel(x_y_z[2])
    ax.set_title(title, fontsize=fontsize_title)
    # ax.margins(z=margin_z)

    if legend:
        plt.legend(title=hue)

    # plt.tight_layout()
    return ax
