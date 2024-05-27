import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from IPython.display import display
import matplotlib.patches as mpatches

# Perso
from src.p7_regex import sel_item_regex

"""***********************************************************************************************************************
RUBRIQUE MISSING VALUES
**************************************************************************************************************************
"""


def mv_loss_ceils_graph(x, y, ceil=20, type_to_eliminate="columns"):
    """Trace un graphique pour aider à choisir un seuil d'élimination de valeurs manquantes
    par ligne ou par colonne
    utiliser avant une fonction pour calculer les pertes par seuil : mv_loss_ceils_lines

    Args:
        x (list): Liste des seuils à plotter sur l'axe des X
        y (list): Liste des pertes d'information à plotter sur l'axe des Y
        ceil (int, optional): Seuil choisi à plotter sous forme d'une ligne verticale. Defaults to 20.
        type_to_eliminate (str, optional): choix pour élimination de lignes ('lines') ou colonnes ('columns').
            Defaults to 'columns'.

    Returns:
        tuple: fig et ax pour éventuellement modifier le graphique a posteriori
    """
    fig, ax = plt.subplots(figsize=(15, 5))

    sns.lineplot(x=x, y=y, markers=["o", "d", "h", "p", "*"])

    # Définition des titres (en fonction de si on élimine des lignes ou des colonnes)
    if type_to_eliminate == "columns":
        block_label = ["colonne", "colonnes"]
    else:
        block_label = ["ligne", "lignes"]
    ax.set(
        ylabel=f"% de valeurs renseignées\ndans les {block_label[1]} à éliminer",
        title=f"Elimination de {block_label[1].upper()}\nPerte d'information par seuil de valeurs manquantes toléré",
        xlabel=f"Seuil d'élimination de {block_label[1]} (en % de valeurs manquantes par {block_label[0]})",
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.set_xticks(ceils)
    ax.tick_params(axis="both", left=True, bottom=True, labelsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    # Grille
    ax.grid(visible=True, which="both", axis="x")
    ax.grid(visible=True, which="both", axis="y")

    # Ajout d'une ligne verticale
    # On ajoute la ligne verticale uniquement si le seuil est à l'intérieur de la plage des x
    if ceil and ceil >= x[0] and ceil <= x[-1]:
        ax.axvline(
            x=ceil,
            ymin=0.00,
            ymax=0.94,
            color="g",
            linestyle="--",
            label=f"Seuil d'élimination :\n{ceil}% de valeurs manquantes par {block_label[0]}",
        )

        # Légende affichée uniquement si le seuil est dessiné
        plt.legend(
            # title = "",
            # title_fontsize = "10",
            fontsize="10"
        )
    return fig, ax


def mv_by_columns(df, sort=True, verbose=True):
    """Calcule le pourcentage de valeurs manquantes par colonne

    Args:
        df (DataFrame): Le df contenant les données
        sort (bool, optional): Trie les pourcentages de valeurs manquantes avec le plus gros en haut.
            Defaults to True.
        verbose (bool, optional): Affiche le tableau des valeurs manquantes. Defaults to True.

    Returns:
        Series: Series contenant les pourcentages de valeurs manquantes par colonne
    """
    missing_values = df.isnull().mean() * 100
    if sort:
        missing_values = missing_values.sort_values(ascending=False)
    if verbose:
        print("Pourcentage de valeurs manquantes par colonne :")
        with pd.option_context("display.float_format", "{:.2f}%".format):
            display(missing_values)
    return missing_values


def mv_loss_ceils_lines(df, start=None, stop=None, step=5.0, verbose=False):
    """Calcule les pertes d'information provoquées par l'élimination de lignes à différents seuils
    utilisée pour tracer le graphique d'élimination par seuil mv_loss_ceils_graph

    Args:
        df (Dataframe): df contenant les données
        start (_type_, optional): 1er seuil de valeurs manquante à calculer. Defaults to None.
        stop (_type_, optional): Dernier seuil de valeurs manquante à calculer. Defaults to None.
        step (_type_, optional): Cran de seuil souhaité. Defaults to 5..
        verbose (bool, optional): Affiche les pertes à chaque seuil (15 max). Defaults to False.

    Returns:
        lists: seuils, % de valeurs renseignées perdues à chaque seuil, nb de lignes perdues à chaque seuil,
            % de valeurs manquantes par ligne
    """
    # Pourcentage de valeurs manquantes par ligne
    missing_values_by_line = df.isna().mean(axis=1) * 100

    # Liste de seuils de valeurs manquantes par ligne que l'on veut simuler
    # Si start n'est pas précisé on le met au minimum de VM arrondi
    if start is None:
        start = np.floor(missing_values_by_line.min())
    # Si stop n'est pas précisé, on le met au maximum de VM arrondi
    if stop is None:
        stop = np.ceil(missing_values_by_line.max())
    ceils = np.arange(start, stop, step)

    # Liste des pourcentages de valeurs renseignées perdues à chaque seuil
    # (par rapport au nombre de cellules du bloc de lignes)
    loss_filled_values = []

    # Liste du nombre de lignes perdues à chaque seuil
    loss_lines = []

    # On boucle sur les seuils de valeurs manquantes pour calculer l'information perdue
    for ceil in ceils:
        # On dresse la liste des lignes au dessus du seuil en % de valeurs manquantes
        lines_to_drop = []
        lines_to_drop = list(
            missing_values_by_line[missing_values_by_line >= ceil].index
        )
        loss_lines.append(len(lines_to_drop))
        loss_data = 0
        # S'il reste des lignes à dropper pour le seuil en cours
        if len(lines_to_drop) > 0:
            # Le bloc de lignes sur toutes les colonnes comprend x% de valeurs renseignées
            loss_data = df.loc[lines_to_drop, df.columns].notna().sum().sum()
            loss_data = loss_data * 100 / (len(lines_to_drop) * df.shape[1])
        loss_filled_values.append(loss_data)

    # Si verbose est True, on affiche les pertes à chaque seuil (15 premières lignes au max)
    if verbose:
        # On iverse les listes
        ceils_reversed = ceils[::-1]
        loss_lines_reversed = loss_lines[::-1]
        loss_values_reversed = loss_filled_values[::-1]
        for i, ceil in enumerate(ceils_reversed):
            if i < 15:
                print(
                    f"Seuil {ceil:.2f}% : bloc de {loss_lines_reversed[i]} lignes rempli à {loss_values_reversed[i]:.2f}%"
                )

    return ceils, loss_filled_values, loss_lines, missing_values_by_line


# Finalement inutilisée
def mv_loss_ceils_columns(df, start=20.0, stop=100.0, step=5.0, pct="block"):
    """Calcule la perte d'information provoquée par l'élimination de colonnes à différents seuils
    de valeurs manquantes tolérés par colonne (simulation si on élimine toutes les colonnes au dessus de chaque seuil)

    Args:
        df (Dataframe): Dataframe contenant les données
        start (float, optional): Seuil minimum de valeurs manquante à simuler (en % de VM). Defaults to 20.
        stop (float, optional): Seuil maximum (arrondi) de valeurs manquante à simuler (en % de VM). Defaults to 100.
        step (float, optional): Pas d'incrémentation pour construire les seuils entre start et stop. Defaults to 5.
        pct (str, optional): Perte d'information calculée à chaque seuil. 'block' ou 'total filled'. Defaults to 'block'.
                            'block' : pourcentage valeurs renseignées dans le bloc de colonnes
                            'total filled' : pourcentage de valeurs renseignées par rapport au total de valeurs renseignées dans le df

    Returns:
        tuple 2 listes: Liste des seuils construits (ceils), Liste des pertes d'information à chaque seuil (loss_by_ceil)
    """
    # Liste de seuils de valeurs manquantes par colonne que l'on veut simuler
    # ceils = list(range(start, stop, step))
    # Pour que ça fonctionne aussi sur des floats
    ceils = np.arange(start, stop, step)
    # Liste des pourcentages de perte d'information à chaque seuil d'élimination
    loss_by_ceil = []

    # Pourcentage de valeurs manquantes par colonne
    missing_values_by_col = df.isna().mean() * 100
    total_filled_values = df.notna().sum().sum()
    # print("Nombre de valeurs renseignées dans le fichier complet", total_filled_values)

    # On boucle sur les seuils de valeurs manquantes pour calculer l'information perdue
    for ceil in ceils:
        # On dresse la liste des colonnes au dessus du seuil en % de valeurs manquantes
        columns_to_drop = []
        columns_to_drop = list(
            missing_values_by_col[missing_values_by_col >= ceil].index
        )
        # print(f"\nau seuil {ceil}, {len(columns_to_drop)} colonnes à dropper : {columns_to_drop}")

        # On calcule le nombre de valeurs renseignées qui seraient perdues si on droppait les colonnes
        loss_data = df[columns_to_drop].notna().sum().sum()
        # print(f"\nau seuil {ceil}, si on drop les {len(columns_to_drop)} colonnes, on perd: {loss_data} données")

        # Si on veut le pourcentage d'information perdue par rapport au total de valeurs renseignées dans le df
        # = perte d'information en pourcentage de l'information contenue dans la base
        if pct == "total filled":
            # la perte représente x% par rapport au total de valeurs renseignées du fichier
            loss_data_total = loss_data * 100 / total_filled_values
            # print(f"En pourcentage on perd: {loss_data_pct} % de données renseignées")
            loss_by_ceil.append(loss_data_total)

        # Si on veut le pourcentage de valeurs renseignées par rapport au total de cellules dans le bloc de colonnes
        elif pct == "block":
            # le bloc de colonnes qu'on s'apprète à éliminer à ce seuil comporte y% de valeurs renseignées
            if len(columns_to_drop) > 0:
                loss_data_block = loss_data * 100 / (df.shape[0] * len(columns_to_drop))
            else:
                loss_data_block = 0
            loss_by_ceil.append(loss_data_block)

    """    # On formate le tout dans un dictionnaire
    # Précaution car alphabétiquement '12' < '1.1', les seuils ne seraient plus triés en passant
    # par une Series directement. 
    for i, ceil in enumerate(ceils):
        ceils[i] = f"{ceil:.0f}%"
    loss_dict = dict(zip(ceils, loss_by_ceil))
    #print(loss_dict)

    # On fait une Series avec le dico
    loss = pd.Series(loss_dict)"""
    return ceils, loss_by_ceil


# Finalement inutilisée
def missing_values_ceils(df, start=0, stop=100, step=10):
    # Liste de seuils de valeurs manquantes par colonne que l'on veut simuler
    ceils = list(range(start, stop, step))
    # Liste des pourcentages de valeurs renseignées perdues par rapport au total des valeurs renseignées à chaque seuil
    loss_by_ceil_values = []
    # Liste des pourcentages de lignes impactées par rapport au total de lignes à chaque seuil
    loss_by_ceil_lines = []

    # Pourcentage de valeurs manquantes par colonne
    missing_values_by_col = df.isna().mean() * 100
    total_filled_values = df.notna().sum().sum()
    # print("Nombre de valeurs renseignées dans le fichier complet", total_filled_values)

    # On boucle sur les seuils de valeurs manquantes pour calculer l'information perdue
    for ceil in ceils:
        # On dresse la liste des colonnes au dessus du seuil en % de valeurs manquantes
        columns_to_drop = []
        columns_to_drop = list(
            missing_values_by_col[missing_values_by_col >= ceil].index
        )
        # print(f"\nau seuil {ceil}, {len(columns_to_drop)} colonnes à dropper : {columns_to_drop}")

        # On calcule le nombre de valeurs renseignées qui seraient perdues si on droppait les colonnes
        loss_data = df[columns_to_drop].notna().sum().sum()
        # print(f"\nau seuil {ceil}, si on drop les {len(columns_to_drop)} colonnes, on perd: {loss_data} données")

        # la perte représente x% par rapport au total de valeurs renseignées du fichier
        loss_data_values = loss_data * 100 / total_filled_values
        # print(f"En pourcentage on perd: {loss_data_pct} % de données renseignées")
        loss_by_ceil_values.append(loss_data_values)

        # la perte impacte y% de lignes par rapport au nombre total de lignes
        if len(columns_to_drop) > 0:
            loss_data_lines = loss_data * 100 / (df.shape[0] * len(columns_to_drop))
        else:
            loss_data_lines = 0
        loss_by_ceil_lines.append(loss_data_lines)

    """    # On formate le tout dans un dictionnaire
    # Précaution inutile a priori car alphabétiquement, les seuils resteraient triés même en passant
    # par une Series directement. Mais des fois qu'on veuille formater autrement ou qu'on manipule d'autres objets...
    for i, ceil in enumerate(ceils):
        ceils[i] = f"{ceil:.0f}%"
    loss_dict = dict(zip(ceils, loss_by_ceil))
    #print(loss_dict)

    # On fait une Series avec le dico
    loss = pd.Series(loss_dict)"""
    return ceils, loss_by_ceil_values, loss_by_ceil_lines


def missing_values_count_vector(vector):
    """Compte les valeurs manquantes d'un vecteur (une ligne ou une colonne d'un DataFrame)
    Args:
        vector (Series): Ligne ou colonne d'un DataFrame dont on veut compter les valeurs manquantes
    Returns:
        count_null (int): nombre de valeurs manquantes du vecteur (utilise np.isnull())
    """
    # Vecteur de bouléens : Vrai si la valeur est nulle, faux sinon
    bool_null_vector = pd.isnull(vector)
    # On fait la somme du vecteur de booléens, il ne compte que les vrais
    count_null = np.sum(bool_null_vector)
    return count_null


def missing_values_percent_vector(vector):
    """Evalue la proportion de valeurs manquantes d'un vecteur en pourcentage du total du vecteur
    Args:
        vector (Series): Ligne ou colonne d'un DataFrame dont on veut évaluer la proportion de valeurs manquantes
    Returns:
        (float): pourcentage de valeurs manquantes du vecteur
    """
    # Vecteur de bouléens : Vrai si la valeur est nulle, faux sinon
    bool_null_vector = pd.isnull(vector)
    # On fait la somme du vecteur de booléens, il ne compte que les vrais
    count_null = np.sum(bool_null_vector)
    # On divise par le nombre d'éléments du vecteur
    return count_null / vector.size


# Compte les valeurs manquantes (date du 1er projet)
def missing_values_count(
    df, groupby=None, type_count="by_col", display_table=False, sort=False
):
    # Si on ne doit pas regrouper les données du df,
    if groupby is None:
        # Pourcentage de valeurs manquantes par ligne
        if type_count == "by_line":
            df = df.isnull().sum(axis=1) / df.shape[1]
        # Pourcentage de valeurs manquantes par colonne
        elif type_count == "by_col":
            df = df.isnull().sum() / df.shape[0]
        else:
            print("Indiquer le type de comptage : 'by_col' ou 'by_line'")
        if sort:
            df = df.sort_values(ascending=False)

    # Si on doit regrouper les données,
    else:
        if type_count == "by_col":
            df = df.groupby(df[groupby], dropna=False, sort=sort).agg(
                missing_values_percent_vector
            )
        # Pourcentage par lignes, colonnes agrégées en une colonne totale
        elif type_count == "by_line_agg_col":
            # [To do] Voir si je ne peux pas améliorer car je somme
            # les float (pourcentages) Il vaudrait mieux tout sommer,
            # puis diviser seulement ensuite lors du groupby
            df = (
                df.groupby(df[groupby], dropna=False, sort=sort)
                .apply(missing_values_percent_vector)
                .sum(axis=1)
            )

        # Pourcentage par lignes, colonnes intactes (non agrégées en total)
        elif type_count == "by_line_keep_col":
            df = df.groupby(df[groupby], dropna=False, sort=sort).apply(
                missing_values_percent_vector
            )
            # On supprime la colonne groupby qui ne présente aucun intérêt
            if groupby in df.columns:
                df.drop([groupby], axis=1, inplace=True)
        else:
            print(
                "Indiquer le type de comptage : 'by_col' ou 'by_line_keep_col' ou 'by_line_agg_col"
            )

    # on multiplie par 100
    df = df * 100

    # Si l'affichage du DataFrame est demandé,
    # on multiplie par 100
    # on l'affiche avec le signe pourcent, 4 chiffres après la virgule
    if display_table:
        with pd.option_context("display.float_format", "{:.4f}%".format):
            display(df)
    return df


def mv_delete_lines_up(data, missing_values=None, limit_top=99.9, inplace=True):
    """Supprime les lignes au dessus d'un certain pourcentage de valeur manquante par ligne

    Args:
        data (DataFrame): Le df contenant les données
        missing_values (Series, optional): Series contenant les valeurs manquantes. Defaults to None.
        limit_top (float, optional): Seuil % de valeurs manquantes au dessus (strictement) duquel on élimine les lignes.
            Defaults to 99.9.
        inplace (bool, optional): Supprime les ligne sur place. Defaults to True.

    Returns:
        Dataframe: Le df contenant les données avec les lignes supprimées
    """
    # Calcul des valeurs manquantes.
    # Si missing_values est fourni, on ne recalcule pas (pour éviter du temps de calcul)
    if missing_values is None:
        missing_values = data.isna().mean(axis=1) * 100

    max = missing_values.max()
    min = missing_values.min()

    # Sélectionne les lignes à supprimer
    n_raws = data.shape[0]
    to_drop = missing_values[missing_values > limit_top].index
    if len(to_drop) == 0:
        print(f"Aucune ligne à plus de {limit_top} % de valeurs manquantes")
        print(f"Maximum de valeurs manquantes : {max:.2f}%, min : {min:.2f}%")
    else:
        # Supprime les lignes
        data.drop(to_drop, axis=0, inplace=inplace)
        print(
            f"Avant suppression {n_raws}lignes. {len(to_drop)} lignes supprimées. Reste : {data.shape[0]} lignes"
        )
    return data


def mv_delete_col_up(data, missing_values=None, limit_top=99.9, inplace=True):
    """Supprime les colonnes au dessus d'un certain pourcentage de valeur manquante par colonne

    Args:
        data (DataFrame): Le df contenant les données
        missing_values (Series, optional): Series contenant les valeurs manquantes. Defaults to None.
        limit_top (float ou str, optional): Seuil % de valeurs manquantes au dessus (strictement) duquel on élimine les lignes.
            Si str, nom de varaible
            Defaults to 99.9.
        inplace (bool, optional): Supprime les ligne sur place. Defaults to True.

    Returns:
        Dataframe: Le df contenant les données avec les colonnes supprimées
    """
    # Calcul des valeurs manquantes.
    # Si missing_values est fourni, on ne recalcule pas (pour éviter du temps de calcul)
    if missing_values is None:
        missing_values = data.isna().mean() * 100

    # Si limit_top est une chaîne (nom de variable), on prend le seuil de cette variable
    if type(limit_top) == str:
        # On vérifie que la chaîne fournie est bien dans l'index des missing_values
        if limit_top in list(missing_values.index):
            limit_top = missing_values[limit_top]
        else:
            print(
                f"la variable '{limit_top}' ne figure pas dans les index de missing_values"
            )

    # Sélectionne les colonnes à supprimer
    to_drop = []
    to_keep = []
    n_raws = data.shape[0]
    for col in data.columns:
        if missing_values[col] > limit_top:
            to_drop.append(col)
        else:
            to_keep.append(col)

    # Supprime les colonnes
    data.drop(to_drop, axis=1, inplace=inplace)
    print(f"{len(to_drop)} colonnes supprimées : {to_drop}")
    print(f"{len(to_keep)} colonnes non restantes : {to_keep}")

    return data


def missing_values_barh(
    data,
    missing_values=None,
    limit_top=100,
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
        limit_top (int, optional):          Seuil % de valeurs manquantes pour distinguer le groupe très manquant. Defaults to 100.
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
    n_max = 200  # Nombre de variables maximum à afficher
    n_max_ticks_up = 30  # Nombre max de variables à partir duquel on affiche les ticks en haut en plus du bas
    fontsize = 9  # Taille de police pour les éléments autres que le titre
    fontsize_title = 14  # Taille de police du titre
    limit_bottom = 30  # Le groupe du bas est constitué des variables à moins de 30% de valeurs manquantes
    decimals = (
        1  # Nombre de décimales à afficher et à prendre en compte pour les seuils
    )
    title = "Pourcentage de valeurs manquantes par colonne\n"

    # ********************************************************
    # En fonction des paramètres fournis
    # ********************************************************
    # Calcul des valeurs manquantes.
    # Si missing_values est fourni, on ne recalcule pas (pour éviter du temps de calcul)
    if missing_values is None:
        missing_values = data.isna().mean() * 100
        missing_values = missing_values.sort_values(ascending=False)

    # Limite le nombre de variables à afficher
    n = missing_values.shape[0]
    if n > n_max:
        n = n_max
        missing_values = missing_values[:n]
        if verbose:
            print(f"Le nombre de variables a été limité à {n}")

    # Tri
    if sort is not None and type(sort) == str:
        if sort == "index":
            missing_values = missing_values.sort_index(ascending=False)
        # Tri par % de valeur manquantes, les plus vides se retrouveront en haut
        if sort == "value" or sort == "values":
            missing_values = missing_values.sort_values(ascending=True)

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
                annotate = sel_item_regex(missing_values.index, annotate)
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


def mv_cross(
    df,
    groupby,
    display_table=True,
    heatmap=True,
    figsize=(12, 8),
    transpose=False,
    cmap="cool_r",
    fmt_cbar="%d",
):
    # Pourcentage de valeurs manquantes au croisement d'une colonne et d'une modalité de la variable groupby
    key = groupby
    to_display = list(df.columns)
    to_display.remove(key)
    mv = df.set_index(key)
    mv = mv[to_display].isnull()
    mv.reset_index()
    mv = mv.groupby(key).mean() * 100

    if display_table:
        print(
            f"Pourcentage de valeur manquantes au croisement d'une colonne et d'une modalité de {groupby}"
        )
        with pd.option_context("display.float_format", "{:.3f}%".format):
            display(mv.head(30))

    if heatmap:
        # On dessine la heatmap
        annot_positive_only = True
        vmax = mv.max().max()
        fontsize_annot = 9
        fontsize_cbar = 10
        fontsize_ticks = 10

        if transpose:
            mv = mv.transpose()

        ax = sns.heatmap(
            # Dataframe qui contient les data
            mv,
            annot=False,
            # fmt=fmt_annot,
            # Colormap à utiliser
            cmap=cmap,
            # Dessine une ligne pour séparer les carrés (pour la l'épaisseur on repassera...)
            linewidths=0.1,
            # Ecrit les étiquettes de l'axe des y
            yticklabels=True,
            # Contrôle l'étendue de la colorbar
            vmax=vmax,
            vmin=0,
            # Colorbar : rétrécie
            cbar_kws={"format": fmt_cbar, "shrink": 0.8},
            # Taille du texte à l'intérieur des cellules
            annot_kws={"fontsize": fontsize_annot},
        )

        # On récupère la cbar
        colorbar = ax.collections[0].colorbar
        # On fixe la taille de la police des ticks de la cbar
        colorbar.ax.tick_params(labelsize=fontsize_cbar)
        # On donne le titre de la cbar
        colorbar.ax.set_ylabel(
            "Pourcentage de valeurs manquantes", fontsize=fontsize_cbar
        )

        # Affiche les labels de l'axe des x en haut ,
        # (pad=0 pour coller les labels aux ticks)
        ax.tick_params(
            axis="x",
            rotation=0,
            top=True,
            bottom=False,
            labeltop=True,
            labelbottom=False,
            pad=0,
        )

        # On diminue la taille de police sur les deux axes(x, y) en même temps
        ax.tick_params(axis="both", labelsize=fontsize_ticks)

        if annot_positive_only:
            # On essaye de mettre manuellement les annotations pour les formatter comme on veut
            for i in range(len(mv)):
                for j in range(len(mv.columns)):
                    value = mv.iloc[i, j]
                    if value > 0 and value < 100:
                        ax.annotate(
                            f"{value:.3f}",
                            (j + 0.5, i + 0.5),
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
                    if value == 100:
                        ax.annotate(
                            f"{value:.0f}",
                            (j + 0.5, i + 0.5),
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

        # Titre du graphique et des axes
        ax.set_title(
            f"Pourcentage de valeurs manquantes par colonne et par modalités de {groupby}\n"
        )
        ax.set_ylabel("")
    return
