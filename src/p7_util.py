import datetime
import time

# import random
import numpy as np

# import os
# import torch
import matplotlib.pyplot as plt
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(name, time.time() - t0))


# Fonction reprise de https://mccormickml.com/2019/07/22/BERT-fine-tuning/#31-bert-tokenizer
def format_time(duration_seconds):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    duration_rounded = int(round((duration_seconds)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=duration_rounded))


def format_time_min(duration_seconds):
    """
    Takes a time in seconds and returns a string hh:mm
    """
    # Arrondit au nombre de minutes le plus proche
    duration_rounded = int(round(duration_seconds / 60))

    # Calcul des heures et des minutes restantes
    hours = duration_rounded // 60
    minutes = duration_rounded % 60

    # Retourne au format hh:mm
    return "{:02d}:{:02d}".format(hours, minutes)


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


"""def set_all_seeds(seed=42, verbose=True):
    # voir doc pytorch : https://pytorch.org/docs/stable/notes/randomness.html
    # Comme notre version de cuda > 10.1, nous devons définir une variable d'environnement pour pouvoir garantir
    # la reproductibilité (ça peut altèrer les performances)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # On met l'option warn_only à True pour ne pas lever d'erreur chaque fois que l'algo n'est pas déterminé car
    # il est n'est pas possible de rendre l'algo adaptive_avg_pool2d_backward_cuda déterminé.
    # En effet cet algo n'est utilisé que dans alexnet et vgg16 qui sont vieux, aussi ce n'est pas dans les priorités de pytorch.
    # La couche concernée est 'AdaptiveAvgPool2d'
    # Les autres modèles utilisent des 1D
    # Voir fil : https://github.com/pytorch/pytorch/issues/50198
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"Fixation des graines de hazard, valeur : {seed}")
    return"""


def clean_ram(list_vars, dic_vars):

    plt.close("all")

    vars_to_del = [var for var in list_vars if var in dic_vars.keys()]
    for var in vars_to_del:
        del dic_vars[var]
    print(f"{len(vars_to_del)} variables détruites : {vars_to_del}")
    return
