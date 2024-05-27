# from category_encoders import CountEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from sklearn.preprocessing import StandardScaler

# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from IPython.display import display

# Perso
from src.p7_regex import sel_item_regex


"""***********************************************************************************************************************
RUBRIQUE OUTLIERS
**************************************************************************************************************************
"""

"""
Unsupervised Outlier Detection using the Local Outlier Factor (LOF).

The anomaly score of each sample is called the Local Outlier Factor. It measures the local deviation of the density of a given sample 
with respect to its neighbors. It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood. 
More precisely, locality is given by k-nearest neighbors, whose distance is used to estimate the local density. 
By comparing the local density of a sample to the local densities of its neighbors, one can identify samples that have a substantially lower density 
than their neighbors.
"""


def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


"""
Chauvenet's criterion is a specific method for detecting outliers in data. It is based on the idea that, for a given dataset,
the probability of an outlier occurring is relatively low. Chauvenet's criterion can be useful in certain situations,
such as when you want to identify outliers in data that follows a normal distribution.
However, it is not necessarily the best method for detecting outliers in all cases, 
and there are other methods that may be more appropriate in certain situations.

According to Chauvenet’s criterion we reject a measurement (outlier) from a dataset of size N when it’s probability of observation is less than 1/2N. 
A generalization is to replace the value 2 with a parameter C.

Source — Hoogendoorn, M., & Funk, B. (2018). Machine learning for the quantified self. On the art of learning from sensory data.

It's important to note that Chauvenet's criterion is only applicable to datasets that are normally distributed.
If your dataset is not normally distributed, this method may not be suitable for identifying outliers.
"""


def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


def mark_outliers_iqr_cross(df, col, cross_with, factor=1.5):
    df = df.copy()

    if type(cross_with) == list:
        cross_with = cross_with[0]

    if type(col) == str:
        col = [col]

    # Pour chque colonne donnée dans la liste col, on calcule les outliers de la colonne croisée par groupe de la variable cross_with
    for col_name in col:
        # col_name = col[0]

        if df[col_name].dtype == "object":
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        # On calcule les IQR pour chaque groupe séparément
        q1 = df.groupby(cross_with)[col_name].transform("quantile", 0.25)
        q3 = df.groupby(cross_with)[col_name].transform("quantile", 0.75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # On marque les outliers pour chaque groupe séparément
        df[col_name + "_outlier"] = (df[col_name] < lower_bound.squeeze()) | (
            df[col_name] > upper_bound.squeeze()
        )

    return df


"""# Marque les outliers avec Isolation Forest
def mark_outlier_if(
    df,
    col_to_study,
    contamination_rate=0.001,
    encode_as_num=None,
    label_encode=None,
    count_encode=None,
    random_state=42,
    display_outliers=False,
):
    df_encoded = df.copy()

    if encode_as_num:
        for col_name in col_to_study:
            if col_name in encode_as_num and df_encoded[col_name].dtype == "object":
                df_encoded[col_name] = pd.to_numeric(
                    df_encoded[col_name], errors="coerce"
                )

    if label_encode:
        # On applique le label encoding aux colonnes qualitatives
        for col_name in label_encode:
            label_encoder = LabelEncoder()
            df_encoded[col_name] = label_encoder.fit_transform(df_encoded[col_name])

    if count_encode:
        # On applique le count encoding aux colonnes indiquées
        for col_name in count_encode:
            count_encoder = CountEncoder()
            df_encoded[col_name] = count_encoder.fit_transform(df_encoded[col_name])

    # On applique l'algorithme Isolation Forest
    isolation_forest = IsolationForest(
        n_estimators=300,
        max_features=len(col_to_study),
        contamination=contamination_rate,
        random_state=random_state,
    )
    isolation_forest.fit(df_encoded[col_to_study])

    # On prédit les anomalies
    predictions = isolation_forest.predict(df_encoded[col_to_study])

    # On marque les outliers
    df_encoded["outlier_if"] = predictions
    df_encoded["outlier_if"] = df_encoded["outlier_if"].apply(
        lambda x: True if x == -1 else False
    )

    # On replace les variables non encodées
    df_encoded.loc[:, col_to_study] = df.loc[:, col_to_study]

    # On sélectionne les lignes correspondant aux anomalies
    outliers_if = df_encoded[df_encoded["outlier_if"] == True]

    print(
        f"Au taux de contamination {contamination_rate}, {outliers_if.shape[0]} outliers IF identifiés :"
    )
    if display_outliers:
        display(outliers_if.head())
    return df_encoded"""


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# Compte le nompbre d'outliers IQR pour une variable
def counts_outliers_iqr(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    n_outliers_low = df[df[col] < lower_bound].shape[0]
    n_outliers_high = df[df[col] > upper_bound].shape[0]

    return n_outliers_low + n_outliers_high


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


def outliers_scatter(
    data,
    to_process=None,
    cross_with=None,
    figsize=None,
    verbose=True,
    yticks=None,
    xticks=None,
    square=False,
    title=None,
    log_scale=False,
):
    """Trace un nuage de points de deux variables quantitatives croisée avec une variable qualitative par couleur,
    Met des ticks précis (comme papier millimétré pour pouvoir éliminer les outliers)

    Args:
        data (DataFrame): df contenant les données
        to_process (str ou list, optional): noms des variables quantitatives (si str regex dans les noms de variables). Defaults to None.
        cross_with (str ou list, optional): nom de la variable qualitative (si str, regex). Defaults to None.
        figsize (tuple, optional): Dimmensions de la figure. Defaults to None.
        verbose (bool, optional): Affiche des message d'erreurs. Defaults to True.
        yticks (list, optional): Définition des ticks à positionner sur l'axe des y (3 nombres : start, stop et step). Defaults to None.
        xticks (_type_, optional): Définition des ticks à positionner sur l'axe des x (3 nombres : start, stop et step). Defaults to None.
        square (bool, optional): Force à plotter les mêmes limites pour la var en x et la var en y. Defaults to False.
        title (str, optional): Titre du graphique. Defaults to None.
    """
    # max_modal = 10  # Nombre maximum de modalités de la variable qualitative à croiser
    # margin = 1      # Marge pour le titre etc.
    width = 10  # Largeur de figure par défaut
    height = 10  # Hauteur de figure par défaut
    hue = None
    display_legend = True

    all_columns = set(data.columns.values)
    numerical_columns = set(data.select_dtypes(include="number").columns.values)

    # On construit l'ensemble des variables dont on veut dessiner le nuage de points
    # Si to_process n'est pas précisé, on s'arrête et on demande précision
    if to_process is None:
        print("préciser des variables numériques dans le paramètre to_process")
        print("Ensemble des variables numériques :", numerical_columns)
        # to_process = numerical_columns
        return
    # Si to_process est une chaîne, on considère que c'est une regex et on cherche les colonnes correspondantes
    elif type(to_process) == str:
        all_columns = sel_item_regex(all_columns, to_process, verbose)
    # Si c'est une liste ou un ensemble
    elif type(to_process) == list or type(to_process) == set:
        all_columns = set(to_process)
    # Si ce n'est rien de tout ça, on ne calculera rien
    else:
        all_columns = ()

    # On réduit le champ d'étude à deux variables numériques
    numerical_columns = numerical_columns & all_columns
    numerical_columns = list(numerical_columns)
    if len(numerical_columns) > 2:
        numerical_columns = numerical_columns[0:2]
    if verbose:
        print("Colonnes quantitatives sélectionnées :", numerical_columns)

    n_numerical = len(numerical_columns)

    # S'il y a des colonnes numériques,
    if n_numerical > 0:
        # Si cross_with n'est pas précisé, on ne croisera pas
        if cross_with is None:
            cross_with = {}
        # Si cross_with est une chaîne, on considère que c'est une regex et on cherche les colonnes correspondantes
        elif type(cross_with) == str:
            cross_with = sel_item_regex(data.columns, cross_with, verbose)
        cross_with = list(cross_with)
        # On autorise une seule variable à croiser, on prend la premère
        if len(cross_with) > 1:
            cross_with = cross_with[0:1]
        if verbose:
            print("numerical_columns =", numerical_columns)
            print("cross_with :", cross_with)

        n_cross = len(cross_with)

        # On construit les listes à utiliser pour les ticks yticks
        if yticks is not None:
            if type(yticks) == list and len(yticks) >= 3:
                start = yticks[0]
                stop = yticks[1]
                step = yticks[2]
                yticks = list(np.arange(start, stop, step))
                if stop not in yticks:
                    yticks.append(stop)
                ylabels = []
                for stick in yticks:
                    ylabels.append(str(stick))
                if verbose:
                    print("Y ticks major =", yticks)
                    print("Labels pour les Y ticks major=", ylabels)
            else:
                print(
                    "Donner une liste de 3 nombres pour les ticks, yticks=[start, stop, step]"
                )
                yticks = None

        # Si il n'y a qu'1 variable numérique à plotter, on met cross_with sur l'axe des x
        if n_numerical == 1:
            if n_cross == 1:
                x = cross_with[0]
                y = numerical_columns[0]
                # hue = None
                hue = x
                if title is None:
                    title = f"{y} par {x}"
                display_legend = False

        # Si il y a 2 variables numériques à plotter,
        if n_numerical == 2:
            x = numerical_columns[0]
            y = numerical_columns[1]
            title = f""

            # Si des ticks sont demandés pour l'axe des X, on construit la liste
            if xticks is not None:
                if type(xticks) == list and len(xticks) >= 3:
                    start = xticks[0]
                    stop = xticks[1]
                    step = xticks[2]
                    xticks = list(np.arange(start, stop, step))
                    if stop not in xticks:
                        xticks.append(stop)
                    xlabels = []
                    for stick in xticks:
                        xlabels.append(str(stick))
                    if verbose:
                        print("X ticks major =", xticks)
                        print("Labels pour les X ticks major=", xlabels)
                else:
                    print(
                        "Donner une liste de 3 nombres pour les ticks, xticks=[start, stop, step]"
                    )
                    xticks = yticks
                    xlabels = ylabels
            else:
                xticks = yticks
                # xlabels = ylabels
                x_labels = None
                y_labels = None

            # Si il y a 2 variables numériques à plotter et que square = True
            # On met les mêmes xlim et ylim
            if square is not None and square == True:
                min_lim = data[numerical_columns].min().min() - 1
                max_lim = data[numerical_columns].max().max() + 1

            # Si il y a 2 variables numeriques et 1 variable qualitative à croiser
            if n_cross == 1:
                hue = cross_with[0]

            # si 2 variables numériques mais aucune variable qualitative à croiser
            else:
                hue = None

        # Si il y a une variable à croiser, on trie
        if n_cross > 0:
            data_to_plot = data[cross_with + numerical_columns].sort_values(
                by=cross_with[0]
            )
        else:
            data_to_plot = data[numerical_columns]

        # On crée la figure
        if figsize is None:
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig, ax = plt.subplots(figsize=figsize)
        if yticks is not None:
            plt.yticks(yticks, ylabels)
            if n_numerical > 1:
                if xticks is not None:
                    plt.xticks(xticks, xlabels)
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(
                    axis="both",
                    which="minor",
                    size=5,
                    gridOn=True,
                    grid_color="grey",
                    grid_alpha=0.1,
                )

        if square is not None and square == True:
            ax.set_xlim(left=min_lim, right=max_lim)
            ax.set_ylim(bottom=min_lim, top=max_lim)

        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(
            axis="both",
            which="major",
            left=True,
            bottom=True,
            labelsize=9,
            gridOn=True,
            grid_alpha=0.9,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            size=5,
            gridOn=True,
            grid_color="grey",
            grid_alpha=0.1,
        )
        # hue_order = ['a', 'b', 'c', 'd', 'e']
        ax = sns.scatterplot(data=data_to_plot, x=x, y=y, hue=hue)

        # Titre
        if title is not None:
            ax.set_title(title)

        # Echelle logarithmique
        if log_scale:
            plt.xscale("log")
            plt.yscale("log")

        # Légende affichée uniquement si 2 var numérique et 1 cross
        if n_cross > 0:
            title_legend = cross_with[0]
        else:
            title_legend = ""
        plt.legend(title=title_legend, fontsize="11")
        if display_legend == False:
            ax.get_legend().set_visible(False)

    return


def outliers_one_boxplot_subplot(df, feature, x_label=False, y_ticks_labels=False):
    # Propriétés graphiques des boxplots
    medianprops = {"color": "black"}
    meanprops = {
        "marker": "o",
        "markeredgecolor": "black",
        "markerfacecolor": "firebrick",
    }
    flierprops = {
        "marker": "D",
        "markerfacecolor": "black",
        "markersize": 5,
        "linestyle": "none",
    }
    plt.boxplot(
        df[feature],
        # labels=modalities,
        showfliers=True,
        flierprops=flierprops,
        medianprops=medianprops,
        vert=False,
        patch_artist=True,
        showmeans=True,
        meanprops=meanprops,
    )
    if not x_label:
        plt.gca().set_xlabel("")

    # Permet de cacher les ticks ainsi que leur label sur l'axe des y
    if not y_ticks_labels:
        plt.yticks([])
    return


def outliers_boxplot(
    data,
    to_process=None,
    cross_with=None,
    log_scale=False,
    verbose=True,
    title="Outliers de ",
):
    """Trace des boîtes à moustaches horizontales sur un ensemble de variables numériques éventuellement
    croisées avec une variable qualitatives

    Args:
        data (DataFrame): df contenant les données
        to_process (str ou list, optional): noms des variables quantitatives (si str, regex). Defaults to None.
        cross_with (str ou list, optional): nom de la variable qualitative (si str, regex). Defaults to None.
        verbose (bool, optional): Affiche des messages précisant les variables sélectionnées par regex. Defaults to True.
    """
    max_modal = 20  # Nombre maximum de la variable qualitative à croiser
    height_box = 0.5  # Hauteur de chaque box
    margin = 1  # Marge pour le titre etc.
    width = 10  # Largeur de figure
    # title = "Outliers"

    all_columns = set(data.columns.values)
    numerical_columns = set(data.select_dtypes(include="number").columns.values)

    # On construit l'ensemble des variables dont on veut dessiner les boxplot
    # Si to_process n'est pas précisé, on calcule sur toutes les variables quantitatives de data
    if to_process is None:
        to_process = numerical_columns
    # Si columns est une chaîne, on considère que c'est une regex et on cherche les colonnes correspondantes
    elif type(to_process) == str:
        all_columns = sel_item_regex(all_columns, to_process, verbose)
    # Si c'est une liste ou un ensemble
    elif type(to_process) == list or type(to_process) == set:
        all_columns = set(to_process)
    # Si ce n'est rien de tout ça, on ne calculera rien
    else:
        all_columns = ()

    # On réduit le champ d'étude aux variables numériques
    numerical_columns = numerical_columns & all_columns
    if verbose:
        print("Colonnes quantitatives sélectionnées :", numerical_columns)

    # S'il y a des colonnes numériques,
    if len(numerical_columns) > 0:
        # On construit l'ensemble des variables à plotter
        # Si cross_with n'est pas précisé, on ne croisera pas
        if cross_with is None:
            cross_with = {}
        # Si cross_with est une chaîne, on considère que c'est une regex et on cherche les colonnes correspondantes
        elif type(cross_with) == str:
            cross_with = set(sel_item_regex(data.columns, cross_with, verbose))

        # On transforme les ensembles en listes
        numerical_columns = list(numerical_columns)
        cross_with = list(cross_with)
        # On autorise une seule variable à croiser, on prend la premère
        if len(cross_with) > 1:
            cross_with = cross_with[0:1]
        if verbose:
            print("numerical_columns =", numerical_columns)
            print("cross_with :", cross_with)

        # Si il y a plusieurs variables à plotter, on met ces variables dans les lignes
        if len(numerical_columns) > 1:
            to_plot = pd.melt(
                data[numerical_columns + cross_with],
                id_vars=cross_with,
                var_name="variable",
            )
            hue = "variable"
            x = "value"

            # Si il n'y a pas de croisement à faire
            if len(cross_with) == 0:
                y = "variable"
                hue = None
            else:
                y = cross_with[0]

        # Si il y a une seule variable à plotter
        elif len(numerical_columns) == 1:
            to_plot = data[numerical_columns + cross_with]
            hue = None
            x = numerical_columns[0]
            title = title + f"{numerical_columns[0]}"
            # title = f"{title} de {numerical_columns[0]}"

            # Si il n'y a pas de croisement à faire
            if len(cross_with) == 0:
                y = None
            else:
                y = cross_with[0]

        # Si il y a au moins une variable à plotter
        if len(numerical_columns) > 0:
            data_to_plot = to_plot
            if verbose:
                display(data_to_plot)

            # On calcule le nombre de boîtes à dessiner (pour la hauteur de la figure)
            if len(cross_with) > 0:
                # data_to_plot = data_to_plot.sort_values(by=cross_with[0])
                n_modal = data[cross_with[0]].nunique()
            else:
                n_modal = 1
            n_box = len(numerical_columns) * n_modal

            # On évalue la hauteur de la figure
            height = n_box * height_box
            height = height + margin
            if verbose:
                print(f"fig largeur : {width} et hauteur : {height}")

            # fig, ax = plt.subplots(figsize=(17, 8))
            fig, ax = plt.subplots(figsize=(width, height))

            #
            if n_modal > 1:
                # height_box = height_box * len(numerical_columns)
                height_box = 0.7
                data_to_plot = data_to_plot.sort_values(by=cross_with[0])
                y_label = f"Modalités de {cross_with[0]}"
            else:
                y_label = ""

            # Titre de l'axe des x:
            xlabel = ""

            # Si demandé on met une échelle logarithmique
            if log_scale == True:
                # On vérifie qu'il n'y a pas de valeurs négatives
                min_df = data[numerical_columns].min().min()
                if min_df <= 0:
                    print(
                        "Les données numériques contiennent des valeurs négatives ou égales à 0, l'échelle ne sera pas logarithmique"
                    )
                    log_scale = False
                # [to do] ajouter 1 si zéro.
                else:
                    plt.xscale("log")
                    xlabel = "Echelle logarithmique"

            # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,.0f} K'.format(x/1_000)))
            ax = sns.boxplot(
                data=data_to_plot,
                x=x,
                y=y,
                width=height_box,
                # y=cross_with[0],
                hue=hue,
            )

            ax.tick_params(axis="x", labelsize=10)
            ax.set(ylabel=y_label, xlabel=xlabel, title=title)
            plt.tight_layout()
            plt.show()
            # fig.tight_layout(pad=0.03)
    return


def outliers_iqr(data, to_process=None, to_display=None, verbose=True):
    """Renvoie les outliers avec la règle de 1.5 fois les intervalles interquartiles

    Args:
        data (DataFrame): DataFrame contenant les variables à étudier
        to_process (str ou list ou set, optional): Variables dont on veut détecter les outliers. Defaults to None.
            si None calcule pour toutes les variables quantitatives du df
            si str : ex '100g', considéré comme une regex et cherche les colonnes correspondantes dans data.columns
        to_display (str ou list ou set, optional): Variables à afficher pour identifier les outliers. Defaults to None.
            si str : ex 'grade', considéré comme une regex et cherche les colonnes correspondantes dans data.columns
        display_table (bool, optional): _description_. Defaults to True.
        verbose (bool, optional): Affiche les détails aux différentes étapes. Defaults to True.

    Returns:
        Dataframe: Dataframe contenant les outliers identifiés
    """
    all_columns = set(data.columns.values)
    numerical_columns = set(data.select_dtypes(include="number").columns.values)

    # On construit l'ensemble des variables dont on veut calculer les IQR
    # Si to_process n'est pas précisé, on calcule sur toutes les variables quantitatives de data
    if to_process is None:
        to_process = numerical_columns
    # Si columns est une chaîne, on considère que c'est une regex et on cherche les colonnes correspondantes
    elif type(to_process) == str:
        all_columns = sel_item_regex(all_columns, to_process, verbose)
    # Si c'est une liste ou un ensemble
    elif type(to_process) == list or type(to_process) == set:
        all_columns = to_process
    # Si ce n'est rien de tout ça, on ne calculera rien
    else:
        all_columns = ()

    # On réduit le champ d'étude aux variables numériques
    numerical_columns = numerical_columns & all_columns
    if verbose:
        print("Colonnes quantitatives sélectionnées :", numerical_columns)

    # S'il y a des colonnes numériques,
    if len(numerical_columns) > 0:
        # On construit l'ensemble des variables à afficher pour identifier les outliers
        # Si to_display n'est pas précisé, on n'affichera que les variables numériques
        if to_display is None:
            to_display = {}
        # Si to_display est une chaîne, on considère que c'est une regex et on cherche les colonnes correspondantes
        elif type(to_display) == str:
            to_display = set(sel_item_regex(data.columns, to_display, verbose))
        # Si c'est une liste, on la convertit en ensemble
        elif type(to_display) == list:
            to_display = set(to_display)
        # On n'affichera que les colonnes spécifiées qui existent dans data.columns
        to_display = to_display & set(data.columns)

        # On transforme les ensembles en listes
        numerical_columns = list(numerical_columns)
        to_display = list(to_display)

        # On crée un dataframe qui contiendra les outliers de toutes les colonnes
        # columns = list(to_display.union(numerical_columns))
        outliers = pd.DataFrame(
            columns=(
                to_display
                + numerical_columns
                + ["varname", "limit_bottom", "limit_top"]
            ),
            data=None,
        )

        # Pour chaque colonne numériques on trouve les outliers
        for col in numerical_columns:
            # on calcule les intervalles inerquartiles puis les limites min et max
            q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
            iqr = q3 - q1
            limit_bottom, limit_top = q1 - iqr * 1.5, q3 + iqr * 1.5
            # On sélectionne les lignes qui dépassent les limites pour la colonne en cours
            outliers_col = data.loc[
                (data[col] < limit_bottom) | (data[col] > limit_top),
                to_display + numerical_columns,
            ]
            # if verbose:
            print(f"Colonne {col} : {outliers_col.shape[0]} outliers détectés")
            # Pour les outliers de la colonne en cours, on crée 3 var, une avec le nom de la colonne
            # et les 2 autres pour se rappeler des limites iqr
            outliers_col = outliers_col.assign(
                varname=col, limit_bottom=limit_bottom, limit_top=limit_top
            )
            # On ajoute les outliers de la colonne en cours au df de tous les outliers
            outliers = pd.concat([outliers, outliers_col])
    return outliers
