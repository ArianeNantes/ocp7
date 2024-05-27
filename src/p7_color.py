# import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# import matplotlib.patheffects as pe
# import matplotlib.patches as mpatches
# import matplotlib.dates as mdates
# from matplotlib.lines import Line2D
# from matplotlib.ticker import PercentFormatter
# from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib import rcParams

# import seaborn as sns

# from matplotlib.ticker import AutoMinorLocator
from IPython.display import display


"""***********************************************************************************************************************
RUBRIQUE 
**************************************************************************************************************************
"""


# Récupère les n premières couleurs de la palette en cours d'utilisation
def get_palette_colors(n_colors):
    # Récupérer le générateur de couleurs de la palette en cours
    colors_cycle = rcParams["axes.prop_cycle"]
    # Convertir le générateur en liste de couleurs
    colors_list = list(colors_cycle)

    # On récupère la palette en cours
    palette = colors_cycle.by_key()["color"]

    # On récupère les n premières couleurs de la palette
    # first_n_colors = colors_list[:n_colors]
    first_n_colors = palette[:n_colors]

    return first_n_colors


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
