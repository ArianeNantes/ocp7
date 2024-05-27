import numpy as np
import re
import pandas as pd
import subprocess
import glob
import os
from pathlib import Path
import urllib
import tarfile
import zipfile
import requests

import json
from pygments import highlight, lexers, formatters


# Personnalisées
# from src.p7_regex import delete_duplicated_spaces


def print_pretty_json(data_json, colored=True):
    formatted_json = json.dumps(data_json, indent=4)
    if colored:
        colorful_json = highlight(
            formatted_json, lexers.JsonLexer(), formatters.TerminalFormatter()
        )
        print(colorful_json)
    else:
        print(formatted_json)


def make_dir(directories_to_make, verbose=True):
    """Crée les directories spécifiées dans la liste si elles n'existent pas déjà
    (Quand on livre un zip du projet, les directories vides ne sont pas zippées)

    Args:
        directories_to_make (list): liste des directories à créer si nécessaire
    """
    if isinstance(directories_to_make, str):
        directories_to_make = [directories_to_make]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            if verbose:
                print(f"Répertoire {directory} créé")
        else:
            if verbose:
                print(f"{directory} dossier déjà existant")
    return


def download_file(url_download, dir_output="", to_download=None):
    """Télécharge un fichier ou dossier à une url et le dézippe si c'est un .zip ou tgz

    Args:
        url_download (_type_): Lien de téléchargement
        dir_output (str, optional): répertoire où on veut télécharger et dézipper. Defaults to ''.
        to_download (_type_, optional): Fichier ou dossier à télécharger. Defaults to None.
        Si non fourni, prend ce qu'il y a après le dernier slash de l'URL.
    """
    # Si un répertoire dans lequel télécharger les données en sortie est fourni
    if len(dir_output) > 0:
        # S'il n'y a pas de slash à la fin on en ajoute un
        if dir_output[-1:] != "/":
            dir_output = dir_output + "/"
        # Si la directory fournie n'exite pas on la crée
        if not os.path.isdir(dir_output):
            os.makedirs(dir_output)
            print(f"Répertoire {dir_output} créé")

    # Si dir_output n'est pas fourni, c'est le répertoire de travail en cours
    dir_txt = dir_output
    if len(dir_output) == 0:
        dir_txt = os.getcwd()

    # Si le nom du fichier n'est pas spécifié, on le cherche dans l'url donnée
    if to_download is None:
        regex_filename = r"/[\w\.-]+$"
        re.compile(regex_filename)
        found = re.search(regex_filename, url_download)
        # Si on a trouvé, on récupère le nom en enlevant le slasch du début
        if found:
            to_download = found.group()
            to_download = to_download[1:]

    # Si on n'a pas de nom de fichier, on sort
    if to_download is None:
        print("Aucun nom de fichier à télécharger")
    # Si on a un nom de fichier on le télécharge
    else:
        urllib.request.urlretrieve(url_download, dir_output + to_download)
        print(f"{to_download} téléchargé dans le répertoire {dir_txt}")

        # On récupère l'extention
        ext = to_download[-4:]
        # Si c'est point zip on le dézippe
        if ext == ".zip":
            with zipfile.ZipFile(dir_output + to_download, mode="r") as archive:
                archive.extractall(dir_output)
            print(f"Archive zip dézippée dans le répertoire {dir_txt}")
        elif ext == ".tgz":
            to_dezip = tarfile.open(dir_output + to_download)
            to_dezip.extractall(path=dir_output)
            to_dezip.close()
            print(f"Archive tgz dézippée dans le répertoire {dir_txt}")
    return


def dezip(directory, filename):
    filepath = directory + filename

    # On récupère l'extension
    ext = filename[-4:]
    # Si c'est point zip on le dézippe
    if ext == ".zip":
        with zipfile.ZipFile(filepath, mode="r") as archive:
            archive.extractall(directory)
        print(f"Archive zip dézippée dans le répertoire {directory}")
    elif ext == ".tgz":
        to_dezip = tarfile.open(filepath)
        to_dezip.extractall(path=directory)
        to_dezip.close()
        print(f"Archive tgz dézippée dans le répertoire {directory}")
    return


def download_from_url_indirect(dir_output):
    # url = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/download?datasetVersionNumber=2"
    url = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/download"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko"
    }

    params = {"datasetVersionNumber": "2"}
    response = requests.get(url, params=params, headers=headers)

    response = requests.get(url)

    if response.status_code == 200:
        # Trouve le lien de téléchargement direct vers le fichier
        # download_url = response.url.replace('/datasets/olistbr/brazilian-ecommerce', '/datasets/olistbr/brazilian-ecommerce/download') + '.zip'

        # Télécharge le fichier
        response = requests.get(url)
        if response.status_code == 200:
            with open(dir_output, "wb") as file:
                file.write(response.content)
            print("Téléchargement terminé.")
        else:
            print("Erreur lors du téléchargement.")
    else:
        print("Erreur lors de la requête.")


# Affiche les n premières lignes d'un fichier avant d'importer
def display_head(filepath, n_rows=2):
    with open(filepath, "r") as file:
        for _ in range(n_rows):
            line = file.readline()
            if not line:
                break
            print(line, end="")


def files_list_pattern(pattern):
    """Dresse une liste de fichiers respectant un motif

    Args:
        pattern (str): Le motif de fichier recherché (ex : 'EdStats*.csv')

    Returns:
        Liste de string représentant les fichiers qui respectent le motif demandé
    """
    # print("\nMotif de fichiers à importer :", pattern)
    files_list_pattern = glob.glob(pattern)
    # print(f"\nListes de fichiers touvés avec glob.glob({pattern}) : ")
    # print(files_list_pattern)
    return files_list_pattern


def run_command_shell(command="ls -lh Data/EdStats*.csv", output_needed=True):
    """Exécute une commande shell (linux) et affiche ou retourne le résultat selon l'arg output_needed

    Args:
        command (str, optional): La commande shell à exécuter. Defaults to 'ls -lh Data/EdStats*.csv'.
        output_needed (bool, optional): Si True, n'affiche pas la sortie mais la renvoie.
            Si False, affiche la sortie et renvoie None
            Defaults to True.

    Returns:
        _tuple de 2 str_: _Le premier élément est la sortie en str, le 2eme est la commande passée_
            La sortie n'est pas parsée. Décodée en 'utf-8-sig'
    """
    # Si la commande n'est pas spécifiée, on effectue un ls -lha dans le répertoire en cours
    if command is None:
        command = "ls -lha"

    # Si la commande est spécifiée, on la run
    if command is not None:
        # Si la sortie est demandée, on fait un PIPE pour communiquer la sortie écran
        if output_needed:
            proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            # On convertit la sortie de la commande ls en string
            # communicate renvoie un tuple de 2 bytes, la sortie est dans le premier
            # supprimer les éventuels caractères \ufeff en début de chaque ligne 'utf-8-sig', 'ignore' (encodage)
            output_str = proc.communicate()[0].decode("utf-8-sig", "ignore")

        # Si on ne doit pas retourner la sortie de la commande pas besoin de PIPE,
        # juste on affiche dans le terminal
        else:
            subprocess.Popen(command, shell=True)
            output_str = None
    return output_str, command


def parse_output_shell(output, command):
    """Parse la sortie d'une commande shell 'ls -l' effectuée par run_command_shell()

    Args:
        output (str): Sortie de la commande shell, déjà décodée. (ex : renvoyée par run_command_shell())
        command (str): Commande exécutée en shell (ex : 'ls -l'), (ex : renvoyée par run_command_shell())

    Returns:
        liste de str: selon la commande exécutée :
            S'il s'agit d'un 'ls -l', la liste est parsée en détail (File name, Type, Size, Reading right)
            Si la commande n'est pas un 'ls', la sortie est laissée telle quelle
            Si la commande est un ls sans l'option -l, la sortie est parsée en lignes

    """
    output_result = []
    output_lines = []

    # On supprime les espaces inutiles
    # delete_duplicated_spaces(command)
    # On sépare la commande en type de commande et en options
    splitted_command = str.split(command, maxsplit=1)
    command_type = splitted_command[0]

    # S'il y a des options après la commande, on les récupère
    options = []
    if len(splitted_command) > 1:
        options = splitted_command[1]

    # Si l'output existe
    if output:
        output_lines = str.splitlines(output)
        # Si la commande est un ls on split les lignes de l'output
        if command_type == "ls" or command_type == "list":
            # regex : pour matcher l'option -l du ls :
            # une fois un tiret, puis 0 à 8 lettre, puis 0 à 8 lettres,
            # puis espace 0 ou une fois
            regex_option_l = "^-{1}[a-zA-Z]{,8}l{1}[a-zA-Z]{,8} ?"

            # S'il y a l'option -l dans les options, on parse l'interieur des lignes
            if re.match(regex_option_l, options):
                output_result = parse_into_lines_ls(output_lines, df=True)

        # si ce n'est pas un ls ou s'il n'y a pas l'option -l, on ne parse pas l'intérieur
        else:
            output_result = str.splitlines(output)
            # line.decode('utf-8-sig', 'ignore')

    # Si l'output est None (cas par d'un affichage simple à l'écran)
    # [To do] On aurait peut-être dû demander un PIPE systématique
    else:
        output_result = []
    return output_result


def parse_into_lines_ls(output_lines, df=True):
    """A partir d'une sortie parsée en lignes, résultante d'un commande shell 'ls -l',
        parse à l'intérieur de chaque ligne : File name, Type, Size, Reading right

    Args:
        output_lines (liste de str ou Dataframe): sortie d'une commande 'ls' parsée en lignes
        df (bool, optional): pour avoir le résultat sous forme de DataFrame au lieu d'une liste. Defaults to True.

    Returns:
        liste de str ou DataFrame: contient File name, Type, Size, Reading right
    """
    # On va parser les infos suivantes (en supposant que c'est bien un le résultat d'un ls -l):
    file_types = ["Type"]  # File or dir ?
    file_names = ["File name"]
    file_sizes = ["Size"]
    file_rights = ["Reading right"]  # droit en lecture de l'utilisateur en cours

    result = []  # Résultat à retourner sous forme de liste (ou de df plus loin)

    # Pour chaque ligne de la sortie donnée par la commande ls -lh, déjà splittée en lignes
    for line in output_lines:
        # [to do] Voir d'abord si c'est un fichier ou une dir ou autre, ensuite longueur du split, ensuite action
        # On stocke le type (fichier / dir) (position 0)
        # le nom du fichier (8 si les user sont affichés ou 7, dernière pos),
        # le droit en lecture du user (position 1),
        # [to do] Vérifier que le premier droit affiché par ls -l est bien celui du user en cours
        # et la taille du fichier (position 4 si user 3 sinon, 5ème en partant de la fin)
        splitted_line = line.split()
        if len(splitted_line) == 2 and splitted_line[0] == "total":
            file_names.append("total")
            file_sizes.append(splitted_line[1])
            file_rights.append("")
            file_types.append("")
        else:
            file_types.append(line[0])
            if len(splitted_line) > 6:
                file_names.append(splitted_line[-1])
                file_sizes.append(splitted_line[-5])
            else:
                file_names.append("")
                file_sizes.append("")
            file_rights.append(line[1])

    # On construit les éléments du return sous forme d'une liste de listes
    output = []
    output.append(file_types)
    output.append(file_names)
    output.append(file_sizes)
    output.append(file_rights)
    # Le résultat à retourner est pour l'instant une liste de listes
    result = output

    # Si en paramètre, on demande le résultat sous forme de df, on crèe le dataframe
    # Chaque liste est une colonne, et le nom de la colonne est stocké en premère ligne myList[0]
    if df:
        column_names = []
        # On remplit les noms de colonnes du df
        column_names.append(file_names[0])
        column_names.append(file_types[0])
        column_names.append(file_sizes[0])
        column_names.append(file_rights[0])
        # On remplit les données du df
        file_names_data = file_names[1:]
        file_types_data = file_types[1:]
        file_sizes_data = file_sizes[1:]
        file_rights_data = file_rights[1:]
        # [Learning] Sans doute pas la meilleure solution, travail hyper répétitif
        # -> rique erreur +++ copier-coller
        df_to_return = pd.DataFrame(
            list(
                zip(file_names_data, file_types_data, file_sizes_data, file_rights_data)
            ),
            columns=column_names,
        )
        # Si df demandé en retour on retourne la df, pas la liste
        result = df_to_return  # [question] Le type est-il réellement dynamique ?
    return result


def file_size(pattern):
    """renvoie un df contenant la taille (et droits lecture) des fichiers qui respectent le motif pattern

    Args:
        pattern (str): motif des noms de fichiers à étudier

    Returns:
        DataFrame: df contenant les informations sur les fichiers
    """

    # On récupère les tailles par une commande shell ls
    proc = run_command_shell(command="ls -lhS " + pattern)
    files = parse_output_shell(proc[0], proc[1])
    "On ajoute 2 colonnes au df, elles ont les indices 4 et 5"
    files["Lines"] = ""
    files["Columns"] = ""

    i = 0
    for filename in files["File name"]:
        # On récupère le nombre de lignes par une commande shell wc -l
        proc_nb_lines = run_command_shell(command="wc -l " + filename)
        output_nb_lines = parse_output_shell(
            output=proc_nb_lines[0], command=proc_nb_lines[1]
        )
        # La sortie est une liste comprenant le nombre de lignes et le nom de fichier
        # On ne garde que le nombre de lignes
        nb_lines = output_nb_lines[0].split(" ")[0]
        files.iloc[i, 4] = nb_lines

        # On récupère le nombre de colonnes par une commande shell awk -F
        # que l'on interrompt immédiatement, on a la var NF (= le nombre de colonnes).
        # (sinon ça fait toutes les lignes du fichier !)
        command_col = "awk -F',' '{print NF; exit}' " + filename
        proc_nb_columns = run_command_shell(command=command_col)
        output_nb_col = parse_output_shell(
            output=proc_nb_columns[0], command=proc_nb_columns[1]
        )
        # La sortie ne comprend que NF (dans une liste)
        nb_columns = output_nb_col[0]
        files.iloc[i, 5] = nb_columns
        i += 1
    return files
