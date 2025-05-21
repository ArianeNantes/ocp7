import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter, MultipleLocator
import matplotlib.patches as mpatches

from src.modeling.p7_util import search_item_regex


def missing_values_barh(
    data,
    missing_values=None,
    n_max=50,
    annotate="group_middle",
    figsize=None,
    sort="value",
    palette=None,
    colors_idx=[0, 1, 2],
    verbose=False,
):
    """Trace un graphique horizontal des % de valeurs manquantes pr colonnes.
    Les colonnes sont divisées en 3 groupes de couleurs différentes pour visualiser les très vides, les intermédiaires
    et les moins de 20%

    Args:
        data (DataFrame):                   df contenant les données
        missing_values (Series, optional):  Series contenant les % de valeurs manuqantes par colonne. Defaults to None.
        n_max (int, optional):              Nombre maximum de variables à afficher. Default to 50
        annotate (str, optional):           Groupe pour lequel on veut voir les % affichés sur le graphe (group_top, group_middle group_bottom). Defaults to 'group_middle'.
        figsize (tuple, optional):          Taille de la figure. Si non précisé elle est calculée approximativement. Defaults to None.
        sort (str, optional):               Trie les barres (colonnes) par valeurs ou par ordre alphabétique des noms de colonnes. Defaults to 'value'.
        palette (palette, optional):        palette à utiliser pour les couleurs. Defaults to None.
        colors_idx (list, optional):        liste d'index de couleurs dans la palette à utiliser pour les 3 groupes. Defaults to [0, 1, 2].
        verbose (bool, optional):           Affiche des messages d'erreurs. Defaults to False.

    Returns:
        3 objets: la Series des valeurs manquantes par colonne, la figure et l'ax
    """
    # ********************************************************
    # CONSTANTES à modifier rapidement
    # ********************************************************
    limit_top = 100  # Seuil % de valeurs manquantes pour distinguer le groupe très manquant. Defaults to 100.
    n_max_ticks_up = 30  # Nombre max de variables à partir duquel on affiche les ticks en haut en plus du bas
    fontsize = 9  # Taille de police pour les éléments autres que le titre
    fontsize_title = 14  # Taille de police du titre
    limit_bottom = 30  # Le groupe du bas est constitué des variables à moins de 30% de valeurs manquantes
    decimals = (
        1  # Nombre de décimales à afficher et à prendre en compte pour les seuils
    )

    # ********************************************************
    # En fonction des paramètres fournis
    # ********************************************************
    # Calcul des valeurs manquantes.
    # Si missing_values est fourni, on ne recalcule pas (pour éviter du temps de calcul)
    if missing_values is None:
        missing_values = data.isna().mean() * 100

    # Tri
    if sort is not None:
        # Le barh inverse l'ordre donc on  trie en sens inverse et on sélectionnera la queue
        if sort == "index":
            missing_values = missing_values.sort_index(ascending=False)
        # Tri par % de valeur manquantes, les plus vides se retrouveront en haut
        else:
            missing_values = missing_values.sort_values(ascending=True)

    # Limite le nombre de variables à afficher
    n = missing_values.shape[0]
    if n > n_max:
        n = n_max

        if sort == "index":
            title = f"Pourcentage de valeurs manquantes par colonne ({n_max} 1ères)\n"
        else:
            title = f"Pourcentage de valeurs manquantes par colonne (Top {n_max})\n"
    else:
        title = "Pourcentage de valeurs manquantes par colonne\n"

    missing_values = missing_values.tail(n)

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

    # Si limit_top est une chaîne (nom de variable), on prend le seuil de cette variable
    if type(limit_top) == str:
        # On vérifie que la chaîne fournie est bien dans l'index des missing_values
        if limit_top in list(missing_values.index):
            limit_top = np.ceil(missing_values[limit_top])
        else:
            print(
                f"la variable '{limit_top}' ne figure pas dans les index de missing_values"
            )

    # ********************************************************
    # Figure et axe
    # ********************************************************
    # Crée la figure
    fig = plt.figure(figsize=figsize)

    # Récupère l'axe en cours
    ax = plt.gca()

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
    # Et on range les noms de variables dans group_top, group_middle ou group_bottom
    color_bar = []
    group_top = []
    group_middle = []
    group_bottom = []
    for i in range(len(missing_values)):
        color_bar.append("b")
        if missing_values[i] <= limit_bottom:  # groupe du bas
            color_bar[i] = colors[2]
            group_bottom.append(missing_values.index[i])

        # Attention à l'arrondi, bizarre, ne semble pas hyper hyper fiable
        # np.round(missing_values[i], 2) >= limit_top:
        elif np.round(missing_values[i], decimals) >= limit_top:
            # elif missing_values[i] >= limit_top:
            color_bar[i] = colors[0]
            group_top.append(missing_values.index[i])
        else:
            color_bar[i] = colors[1]
            group_middle.append(missing_values.index[i])

    # ********************************************************
    # Axes et ticks
    # ********************************************************
    # On veut des ticks tous les 10% sur l'axe des X pour les major de l'axe des X
    ax.xaxis.set_major_locator(MultipleLocator(10))
    # On les veut en pourcentage
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    # On veut aussi des ticks sur l'axe des Y collés aux noms de variables
    ax.tick_params(axis="y", which="both", left=True, pad=0)
    # On définit là où il y a des labels : en bas et en haut si n_var > n_max_ticks_up, pad = labels collés aux ticks
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
    # Si missing_values n'est pas pandas, on considère que c'est un cuDF et on le convertit en pandas
    if not isinstance(missing_values, (pd.DataFrame, pd.Series)):
        missing_values = missing_values.to_pandas()
    missing_values.plot(kind="barh", width=0.85, color=color_bar)

    # ********************************************************
    # Affichage des étiquettes
    # ********************************************************
    # Si annotate n'est ni None ni False, on affiche les valeurs en pourcentage à droite des barres
    if annotate is not None and (
        type(annotate) != bool or type(annotate) == bool and annotate == True
    ):
        # On construit la liste de tous labels (pourcentage de valeurs manquantes à 1 ou 2 chiffres après la virgule)
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
            else:
                # Si annotate est une chaîne mais pas un groupe, c'est une regex
                # On construit une liste avec les noms de variables correspondant
                annotate = search_item_regex(
                    search_in=missing_values.index, regex=annotate
                )
        # Si annotate est une liste ou un ensemble, on efface les labels des variables de cette liste
        if type(annotate) == list or type(annotate) == set:
            for i, mv in enumerate(missing_values.index):
                if mv not in annotate:
                    labels[i] = ""
        # Si annotate n'est ni false ni None, on affiche les labels
        # (le padding semble n'avoir aucun effet dans la builded fonc. intéressant seulement si on les fait à la main)
        for bars in ax.containers:
            plt.bar_label(bars, labels=labels, padding=0.4, size=fontsize)

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
        location = None
    if len(handles) > 0:
        plt.legend(
            handles=handles,
            title="% Valeurs manquantes",
            fontsize=fontsize,
            title_fontsize=fontsize,
            loc=location,
        )
    return missing_values, fig, ax
