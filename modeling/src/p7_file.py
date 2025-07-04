import re
import glob
import os
import urllib
import tarfile
import zipfile


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


def files_list_pattern(pattern):
    """Dresse une liste de fichiers respectant un motif

    Args:
        pattern (str): Le motif de fichier recherché (ex : 'EdStats*.csv')

    Returns:
        Liste de string représentant les fichiers qui respectent le motif demandé
    """
    files_list_pattern = glob.glob(pattern)
    return files_list_pattern
