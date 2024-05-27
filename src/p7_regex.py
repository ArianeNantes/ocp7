import re
import pandas as pd
import numpy as np
import itertools


"""***********************************************************************************************************************
RUBRIQUE STRING
**************************************************************************************************************************
"""


def column_names(str_list):
    """Transforme la liste des noms de colonnes en remplaçant tous les tirets par des underscores

    Parameters:
        <str_list>: la liste de chaînes à transformer
        <str_len>:  largeur à prendre en compte pour afficher les chaînes avant et après transformation.
                    si str_len == 0, la fonction n'affiche rien.

    Returns:
        <new_list>: Liste de chaînes transformées"""
    # index_s = 0
    new_list = []

    for s in str_list:
        # Chaîne avant transformation
        s_old = s

        # Liste de séparateur à remplacer par un espace
        separators = ["_"]
        for sep in separators:
            s = s.replace(sep, " ")

        # On sépare

        # Si la chaine est entièrement en majuscules, on la convertit en firstCapitale
        if s.isupper():
            # print('s entièrement en maj =', s)
            s = s.capitalize()
        # Si la chaîne n'est pas entièrement en majuscule on cherche
        # la position des majuscules pour mettre un espace devant mais uniquement si
        # la lettre précédant la majuscule est en minuscule
        else:
            i = 1
            while i < len(s):
                if s[i].isupper() and s[i - 1].islower():
                    s = s.replace(s[i], " " + s[i])
                i += 1
            """ for car in s :
                if car.isupper() :
                    s = s.replace(car, ' '+ car) """

        # Tout en minuscules
        s = s.lower()

        # On sépare en mots
        word_list = s.split()

        # On met le premier mot en firstCap
        word_list[0] = word_list[0].capitalize()

        # On regroupe les mots en les joignant par un espace
        s = " ".join(word_list)
        new_list.append(s)

        if s_old == s:
            transformation = ".."
        else:
            transformation = "->"
        # [Learning] chaine formatée
        # Index sur 5 car aligné à gauche paddé avec des points
        # s_old sur str_len car (par défaut 30) aligné à gauche paddé avec des points
        # s_old sur str_len car (par défaut 30) car aligné à droite paddé avec des points
        """ if str_len > 0 :
            print(f"Index {index_s:.<5} {s_old:.<{str_len}}{transformation}{s:.>{str_len}}" )
 """
    # index_s += 1
    return new_list


@np.vectorize
def delete_duplicated_spaces(str_to_clean, verbose=False):
    """Supprime les espaces inutiles d'une chaîne (début ou fin de chaîne, dupliqués...)
    Args:
        str (string) : la chaîne dont on veut supprimer les espaces inutiles
        verbose (bool) : Default False. Si vrai print le nombre d'espaces supprimés si > 0

    Returns:
        str_cleaned (any) : la chaîne une fois les espaces inutiles supprimés
    """
    cleaned_nb = 0
    clean_str = str_to_clean

    # Si l'objet existe
    if str_to_clean is not None:
        # Si l'objet n'est pas une valeur manquante (dans ce cas ce n'est pas une string donc inutile ?)
        # if (str_to_clean is not np.nan) and (str_to_clean is not pd.NA) and (str_to_clean is not pd.NaT):

        # Si c'est une string
        if type(str_to_clean) == str:
            # Si cette chaine n'est pas vide
            if len(str_to_clean) > 0:
                splitted = str_to_clean.split()
                cleaned_nb = cleaned_nb + len(splitted) - 1
                clean_str = " ".join(splitted)
                if verbose and cleaned_nb > 0:
                    print(f"Nombre d'espaces supprimés : {cleaned_nb}")
    return clean_str


@np.vectorize
def line_breaks_count(vect):
    """Compte le nombres de saut de lignes d'une chaîne multilines (toutes les formes de sauts)
    Args:
        vect (any): Chaînes dont on veut compter les sauts de lignes

    Returns:
        int: Nombre de sauts comptées. 0 si vect n'est pas une string
    """ """"""
    nb_lines = 0
    if vect and isinstance(vect, str):
        nb_lines = len(vect.splitlines()) - 1
    return nb_lines


@np.vectorize
def line_breaks_replace(str_to_clean, replace_with, verbose=False):
    """Remplace les sauts de lignes (\\r, \\n, \\r\\n, \\f, \\v) par une chaîne spécifiée
    Args:
        str_to_clean (any): chaîne dont on veut remplacer les sauts de lignes
        replace_with (str): chaîne avec laquelle on veut remplacer les sauts de lignes
        verbose (bool): default=False, si True, print le nombre de remplacements effectués

    Returns:
        (any): si str retourne la chaîne sans sauts, si non retourne l'objet passé en entrée
    """ """"""
    cleaned = str_to_clean
    break_lines = 0
    # Si l'objet existe
    if str_to_clean is not None:
        # Si l'objet n'est pas une valeur manquante (dans ce cas ce n'est pas une string donc inutile ?)
        if (
            (str_to_clean is not np.nan)
            and (str_to_clean is not pd.NA)
            and (str_to_clean is not pd.NaT)
        ):
            # Si c'est une string
            if type(str_to_clean) == str:
                # Si cette chaine n'est pas vide
                if len(str_to_clean) > 0:
                    splitted_line = str_to_clean.splitlines()
                    cleaned = replace_with.join(splitted_line)
                    if verbose:
                        break_lines = len(splitted_line) - 1
                        print(
                            f"Nombre de sauts remplacés par '{replace_with}' : {break_lines}"
                        )
    return cleaned


"""        # Si c'est une valeur manquante
        else:
            cleaned = np.nan
    else:
        cleaned = np.nan"""

"""***********************************************************************************************************************
RUBRIQUE SERT A TOUT
**************************************************************************************************************************
"""


def sel_var_bins(df, n_bins=20):
    """Sélectionne des colonnes dans un DataFrame qui ont un nombre de valeurs possibles (nunique) < n_bins

    Args:
        df (DataFrame): df contenant les données
        n_bins (int, optional): Nombre de valeurs nunique (ou modalités) maximum pour sélectionner la Series. Defaults to 20.
    """
    nunique_by_col = df.nunique()
    selected_cols = list(nunique_by_col[nunique_by_col <= n_bins].index)

    # retourner le df = df[selected_cols] ou une liste de colonnes ?
    return selected_cols


"""***********************************************************************************************************************
RUBRIQUE REGEX
**************************************************************************************************************************
"""


def sel_var(
    list_var, include_pattern=None, exclude_pattern=None, escape=True, verbose=True
):
    """Renvoie une liste de noms de variables en les sélectionnant dans une liste (ex data.columns)
    d'après un motif de regex à inclure sauf un motif de regex à exclure. Case insensitive

    Args:
        include_pattern (str): Motif de regex à respecter pour que le nom de variable soit sélectionné
            Si None, inclut toutes les variables de la liste.
        exclude_pattern (str): Motif de regex à respecter pour que la variable soit exclue des variables incluses
            Si None, n'exclut aucune variable
        escape (bool, optional): Ajoute le backslash d'échappement devant les car spéciaux (excepté pour le \ lui-même). Defaults to True.
    """
    # Si escape est True on échappe les caractères spéciaux de include_pattern et de exclude_pattern
    if escape:
        if include_pattern is not None:
            # include_pattern = re.sub(regex_char_to_escape, r'\\' +'\1', include_pattern)
            # print("motif après échappement", include_pattern)
            include_pattern = re.sub("\(", r"\\(", include_pattern)
            include_pattern = re.sub("\)", r"\\)", include_pattern)
            include_pattern = re.sub("\[", r"\\[", include_pattern)
            include_pattern = re.sub("\]", r"\\]", include_pattern)
            include_pattern = re.sub("\{", r"\\{", include_pattern)
            include_pattern = re.sub("\}", r"\\}", include_pattern)
            include_pattern = re.sub("\.", r"\\.", include_pattern)
            include_pattern = re.sub("\*", r"\\*", include_pattern)
            include_pattern = re.sub("\?", r"\\?", include_pattern)
            # include_pattern = re.sub("\$", r"\\$", include_pattern)
            # print("motif include après échappement", include_pattern)
        if exclude_pattern is not None:
            exclude_pattern = re.sub("\(", r"\\(", exclude_pattern)
            exclude_pattern = re.sub("\)", r"\\)", exclude_pattern)
            exclude_pattern = re.sub("\[", r"\\[", exclude_pattern)
            exclude_pattern = re.sub("\]", r"\\]", exclude_pattern)
            exclude_pattern = re.sub("\{", r"\\{", exclude_pattern)
            exclude_pattern = re.sub("\}", r"\\}", exclude_pattern)
            exclude_pattern = re.sub("\.", r"\\.", exclude_pattern)
            exclude_pattern = re.sub("\*", r"\\*", exclude_pattern)
            exclude_pattern = re.sub("\?", r"\\?", exclude_pattern)
            # exclude_pattern = re.sub("\$", r"\\$", exclude_pattern)
            # print("motif exlude après échappement", exclude_pattern)

    # Si include_pattern n'est pas précisé, on sélectionne toutes les variables de la liste de variables
    if include_pattern is None:
        to_include = list_var.copy()

    # Si include_pattern est précisé on cherche les var correspondantes au motif
    else:
        # re.compile(include_pattern)
        to_include = []
        for var in list_var:
            if re.search(include_pattern, var, re.IGNORECASE):
                to_include.append(var)

    to_exclude = []
    # Si exclude_pattern est précisé, on exclut les var correspondant au motif
    if exclude_pattern is not None:
        for var in to_include:
            if re.search(exclude_pattern, var, re.IGNORECASE):
                to_exclude.append(var)

    selected_vars = [var for var in to_include if var not in to_exclude]
    if verbose:
        print(
            f"{len(to_include)} variables à inclure correspondant au motif '{include_pattern}' : {to_include}"
        )
        print(
            f"{len(to_exclude)} variables à exclure correspondant au motif '{exclude_pattern}' : {to_exclude}"
        )
        print(f"{len(selected_vars)} variables sélectionnées : {selected_vars}")

    return selected_vars


def regex_or_words(words, plural=True, r=False):
    """Construit une regex pour rechercher la présence de mots (mot1 OU mot2 etc)

    Parameters:
        <words>: Liste de mots
        <plural>: Si plural est True, on cherche aussi les mots au pluriel (simple ajout d'un s à la fin du mot)
        <r>: INUTILE pour le moment. Old : si r est True, la regex construite sera de la forme r'str'

    Returns:
        <regex>: Chaîne de caractères représentant la regex
    """
    regex = ""
    if r:
        regex = "r'"

    if plural:
        for w in words:
            regex = regex + "\\b" + w + "s?\\b" + "|"
    else:
        for w in words:
            regex = regex + "\\b" + w + "\\b" + "|"

    # On supprime le dernier pipe
    if regex.endswith("|"):
        regex = regex[:-1]

    # Si on a ajouté un r devant, on ajoute un guillemet à la fin
    if r:
        regex = regex + "'"
    return regex


def regex_and_words(words, plural=True):
    regex = ""

    i = 0
    # Pour chaque mot de la liste fournie en paramètre, on ajoute le pluriel si nécessaire
    while i < len(words):
        # Si l'option pluriel est choisie, on ajoute un 's?' dans la regex
        if plural:
            words[i] = "\\b" + words[i] + "s?\\b"
        else:
            words[i] = "\\b" + words[i] + "\\b"
        i += 1

    # On dresse la liste des permutations possibles (Mot1 suivi de Mot2 OU Mot2 suivi de Mot1 etc. )
    permutations = list(itertools.permutations(words))
    # print("Permutations :")
    # print(permutations)

    # Pour chaque permutation, (chaque n-uplets de mots)
    for p in permutations:
        for word in p:
            regex = regex + word + ".*"
        if regex.endswith(".*"):
            regex = regex[:-2]
            regex = regex + "|"
    if regex.endswith("|"):
        # On enlève le ou final
        regex = regex[:-1]

    # print(regex)
    return regex


def sel_lines_regex(df, str_regex, col_to_search_in, verbose=True):
    """Sélectionne les lignes d'un dataframe dont la regex a été trouvée dans une liste de colonnes. case insensitive.

    Parameters:
        <df>: Dataframe dont on veut sélectionner des lignes
        <str_regex>: Chaîne de caractère contenant l'expression régulière à rechercher. Exemple : regex = '\bfemale\b'
        <col_to_search_in> : Liste des colonnes où effectuer la recherche
        <verbose> : Si True print le nombre de lignes trouvées avec la regex dans les colonnes

    Returns:
        <selection>: Dataframe contenant les lignes trouvées"""
    mask = (
        df[col_to_search_in]
        .apply(lambda x: x.str.contains(str_regex, case=False, regex=True))
        .any(axis=1)
    )

    selection = df[mask]
    if verbose == True:
        print(
            f"Recherche du motif {str_regex} dans les colonnes {col_to_search_in} :\n{selection.shape[0]} lignes trouvées"
        )
    return selection


def sel_item_regex(search_in, regex_to_search, verbose=True):
    """Sélectionne les éléments d'une liste ou d'un ensemble à partir d'une expression régulière,
    (considérer que case sensitive car case insensitive fonctionne mal)

    Args:
        search_in (list ou set): ensemble dans lequel chercher l'expression régulière
        regex_to_search (String): Expression régulière
        verbose (bool, optional): print les éléments trouvés si true. Defaults to True.

    Returns:
        list ou set: ensemble des éléments trouvés avec l'expression régulière
    """ """"""
    # je n'arrive pas à faire fonctionner IGNORECASE, je ne sais pas pourquoi. Problème de syntaxe ?
    # Ou dépend de paramètres locaux ?
    # Ou spécifique méthode search ? plutôt faire un faire match ou un findall ?
    # essayé avec re.compile(regex_to_search, re.IGNORECASE=True)
    # essayé avec flags=
    # testé aussi dans search avec re.IGNORECASE ou True, ou re.IGNORECASE=True ou même re.IGNORECASE==True
    # essayé aussi de compiler avant d'appeler la fonc. Pas mieux.

    # re.IGNORECASE = True
    re.compile(regex_to_search, re.IGNORECASE)
    # La constante est True même si je ne dis rien, pourtant pas de match si la casse n'est pas bonne
    # print("re.IGNORECASE", re.IGNORECASE)
    items_matched = []

    # Si l'ensemble dans lequel chercher existe et n'est pas vide
    if search_in is not None and len(search_in) > 0:
        for item in search_in:
            if isinstance(item, str):
                # Si la regex est trouvée dans l'élément,
                # on ajoute l'élément à la liste des éléments trouvés
                if re.search(regex_to_search, item):
                    items_matched.append(item)

    if isinstance(search_in, set):
        items_matched = set(items_matched)

    if verbose:
        print(
            f"{len(items_matched)} éléments correspondent au motif {regex_to_search} :"
        )
        print(items_matched)

    return items_matched
