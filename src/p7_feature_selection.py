import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
import os
import gc
from imblearn.under_sampling import NearMiss
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split as sk_train_test_split
from cuml.model_selection import train_test_split as cu_train_test_split
import cudf
from collections import defaultdict
import time

"""
# Pour l'install de nearmiss
# pip install -U imbalanced-learn
# Dépendances critiques : numpy >= 1.19.5, scipy >= 1.6.0 
https://imbalanced-learn.org/stable/install.html
-------
Python (>= 3.8)
NumPy (>= 1.17.3)
SciPy (>= 1.5.0)
Scikit-learn (>= 1.0.2)
https://pypi.org/project/imbalanced-learn/
"""

from src.p7_constantes import DATA_INTERIM
from src.p7_simple_kernel import get_memory_consumed, reduce_memory
from src.p7_preprocess import check_no_nan


class DataSelector:
    def __init__(
        self, num_prj="00", input_file="00_v0_full_data.csv", random_state=42
    ) -> None:
        self.frac_test = 0.25
        self.random_state = random_state
        self.train = None
        self.test = None
        self.num_prj = num_prj
        self.input_dir = DATA_INTERIM
        self.input_file = input_file
        self.output_dir = DATA_INTERIM
        self.id = "SK_ID_CURR"
        self.target = "TARGET"
        self.not_predictors = ["TARGET", "SK_ID_CURR"]
        self.predictors = None
        # self.predictors = [
        #    f for f in self.train.columns.tolist() if f not in self.not_predictors
        # ]
        # On gardera ces features même si elles ont plus de 30 % de valeurs manquantes
        # car on sait qu'elles sont très importantes
        self.features_to_keep = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "PAYMENT_RATE",  # annuity / credit dans simple kernel
            "APP_CREDIT_TO_ANNUITY_RATIO",  # credit / annuity dans full kernel
        ]
        self.resampling = ""

    def drop_rows_to_keep_feature(
        self, threshold=0.3, feature="EXT_SOURCE_1", verbose=True
    ):

        n_rows_before = self.train.shape[0]
        n_missing_before = self.train[feature].isna().sum(axis=0)
        ratio_before = n_missing_before / n_rows_before

        # S'il n'y a aucun besoin de supprimer des lignes (% de valeur manquantes en dessous du seuil toléré), on sort tout de suite
        if ratio_before <= 0.3:
            print(
                f"{feature} comporte {ratio_before:.0%} de valeurs manquantes (seuil toléré : {threshold:.0%}), aucune lignes à supprimer"
            )
            print("Forme du Train :", self.train.shape)
            return self.train

        # Si le % de valeurs manquantes de la feature est trop important,
        # On calcule le nombre de lignes à supprimer pour que la feature tombe à threshold % de valeurs manquantes
        ratio_after = 0.3
        # ratio_2 = n_missing_2 / (n_rows - n_rows_to_del)
        # ratio_2 * (n_rows - n_row_to_del) = n_missing_2
        # ratio_2 * (n_rows - n_missing_1 + n_missing_2) = n_missing_2
        # ratio_2 * (n_rows - n_missing_1) = n_missing_2 * (1 - ratio_2)
        # n_missing_2 = ratio_2 * (n_rows - n_missing_1) / (1 - ratio_2)
        n_missing_after = round(
            ratio_after * (n_rows_before - n_missing_before) / (1 - ratio_after)
        )
        n_rows_to_del = n_missing_before - n_missing_after
        if verbose:
            print(
                f"{feature}, comporte {n_missing_before:_} valeurs manquantes (ratio = {ratio_before:.0%})"
            )
            print(
                f"Pour tomber à {ratio_after:.0%} de valeurs manquantes, il nous faudrait descendre à {round(n_missing_after):_} valeurs manquantes"
            )

            print(f"Il nous faut donc supprimer {n_rows_to_del:_} lignes dans le train")

        # On calcule le pourcentage de valeur manquantes de la feature selon la target pour voir s'il y a une grosse différence
        subset_default = self.train.loc[self.train["TARGET"] == 1, [feature, "TARGET"]]
        ratio_default = subset_default[feature].isna().mean()
        subset_ok = self.train.loc[self.train["TARGET"] == 0, [feature, "TARGET"]]
        ratio_ok = subset_ok[feature].isna().mean()
        print(
            f"{feature} comporte {ratio_default:.0%} de valeurs manquantes pour la catégorie 'Défaut de remboursement' et {ratio_ok:.0%} pour la catégorie 'Remboursement OK'"
        )
        if abs(ratio_default - ratio_ok) < 0.1:
            print(
                "La différence entre les catégories n'étant pas importante, on supprime les lignes uniquement pour la catégorie 'Remboursement OK'"
            )
        else:
            print(
                "WARNING : Supprimer les lignes dans la catégorie 'Remboursement OK' uniquement peut introduire un biais"
            )

        if "SK_ID_CURR" in self.train.columns:
            self.train = self.train.set_index("SK_ID_CURR")

        # On sélectionne un subset dans lequel enlever les lignes qui est feature à NaN et target est OK
        subset_ok = self.train.loc[self.train["TARGET"] == 0, :]

        subset_to_drop_in = subset_ok.loc[subset_ok[feature].isna() == True, :]
        # print("subset_to_drop_in.shape", subset_to_drop_in.shape)

        # On calcule le nombre de manquantes par ligne pour enlever celles qui en ont le moins
        # On ne mélange pas avant car le train a déjà été mélangé lors du split et shuffle(df) n'existe pas dans cudf
        subset_to_drop_in["n_missing"] = subset_to_drop_in.isna().sum(axis=1)
        # On trie par nombre de missing les plus élevés en tête
        subset_to_drop_in = subset_to_drop_in.sort_values(
            by="n_missing", ascending=False
        )
        # On sélectionne le nombre de lignes parmi celles comportant le plus de valeurs manquantes
        idx_rows_to_drop = subset_to_drop_in.head(n_rows_to_del).index

        # On supprime ces lignes du train
        self.train = self.train.drop(idx_rows_to_drop)
        self.train = self.train.reset_index()
        if verbose:
            print("Nouvelle forme du train", self.train.shape)
            print(
                f"Pourcentage de valeurs manquantes pour {feature} dans train après suppression de {len(idx_rows_to_drop):_} lignes : {self.train[feature].isna().mean():.0%}"
            )
        return self.train

    def drop_too_missing_features(self, threshold=0.3, verbose=True):
        """Supprime les features comportant trop de valeurs manquantes,
        exceptées celles qui sont présisées dans l'attribut features_to_keep

        Args:
            threshold (float, optional): seuil toléré de valeurs manquantes. Defaults to 0.3.
            verbose (bool, optional): Verbosité. Defaults to True.

        Returns:
            cuDF: train débarrassé des features comportant trop de valeurs manquantes
        """
        # On identifie les features avec plus de threshold % de valeurs manquantes dans le train
        too_missing_features = [
            f for f in self.train.columns if self.train[f].isna().mean() > threshold
        ]

        # On exclue de cette liste les features à garder même si elles comportent plus de valeurs manquantes que le seuil
        kept_although_too_missing = []
        for feature in self.features_to_keep:
            if feature in too_missing_features:
                too_missing_features.remove(feature)
                kept_although_too_missing.append(feature)

        self.train.drop(too_missing_features, axis=1, inplace=True)
        self.test.drop(too_missing_features, axis=1, inplace=True)

        if verbose:
            print(
                f"{len(too_missing_features)} features supprimées du train et du test comportant plus de {threshold:.0%} de valeurs manquantes"
            )
            if kept_although_too_missing:
                print(
                    f"{len(kept_although_too_missing)} features conservées bien que comportant plus de {threshold:.0%} de valeurs manquantes :"
                )
                print(kept_although_too_missing)
            print("Nouvelle forme de train :", self.train.shape)
        return self.train

    # Divise en train et test avec cuDF
    def split_cudf(self, verbose=True):
        """Divise en jeu de train et jeu de test (avec cuDF).
        Nécessite au moins 2 colonnes dans le jeu global sans valeurs manquantes.

        Args:
            verbose (bool, optional): Verbosité. Defaults to True.

        Returns:
            cuDF: Jeu de train
        """
        df = cudf.read_csv(os.path.join(self.input_dir, self.input_file)).set_index(
            "SK_ID_CURR"
        )
        # train_test_split en version cuda n'accepte pas les NaN,
        # nous allons donc l'utiliser avec des features sans NaN (au moins 2) pour récupérer les index.
        # Même ainsi, c'est beaucoup plus rapide qu'en passant par train_test_split de sklearn
        without_nan = [f for f in df.columns if df[f].isna().sum() == 0]
        if len(without_nan) < 2:
            print(
                "Pas assez de features sans NaN, faire une division manuelle ou utiliser sklearn"
            )
        else:
            X_train, X_test, _, _ = cu_train_test_split(
                df[without_nan[:2]],
                df["TARGET"],
                test_size=self.frac_test,
                stratify=df["TARGET"],
                random_state=self.random_state,
            )

            idx_train = X_train.index
            idx_test = X_test.index
            self.train = df.loc[idx_train].reset_index()
            self.test = df.loc[idx_test].reset_index()

            if verbose:
                print("Forme initiale du dataset :", df.shape)
                print(
                    f"train.shape : {self.train.shape}, test.shape : {self.test.shape}. (Fraction de test : {self.frac_test:.0%})"
                )
            del df
            gc.collect()
        return self.train

    # Elimine les features de variance nulle sur le train
    def drop_null_std(self, verbose=True):
        features = [
            feature
            for feature in self.train.columns
            if feature not in ["SK_ID_CURR", "TARGET"]
        ]
        all_std = self.train[features].std()
        null_std_features = all_std[all_std == 0.0].index.tolist()
        if null_std_features:
            self.train = self.train.drop(null_std_features, axis=1)
            self.test = self.test.drop(null_std_features, axis=1)
        if verbose:
            if null_std_features:
                print(
                    f"{len(null_std_features)} features de variance nulle suppimées dans Train et Test"
                )
                print(
                    f"Nouvelle taille du jeu de Train : {self.train.shape}, {get_memory_consumed(self.train, verbose=False)} Mo"
                )
            else:
                print(
                    f"Aucune feature de variance nulle. Taille du jeu de Train :  {self.train.shape}, {get_memory_consumed(self.train, verbose=False)} Mo"
                )
        return self.train, self.test

    def reduce_memory_usage(self, inplace=False):
        if not inplace:
            df = self.train.copy()
            reduce_memory(df)
            return df
        else:
            reduce_memory(self.train)
            reduce_memory(self.test)
            return

    def save_data(
        self,
        version="v0",
        suffix="",
        replace=False,
        train_rootname="train",
        test_rootname="test",
    ):
        if suffix:
            train_name = f"{self.num_prj}_{version}_{train_rootname}_{suffix}.csv"
            test_name = f"{self.num_prj}_{version}_{test_rootname}_{suffix}.csv"
        else:
            train_name = f"{self.num_prj}_{version}_{train_rootname}.csv"
            test_name = f"{self.num_prj}_{version}_{test_rootname}.csv"
        train_path = os.path.join(self.output_dir, train_name)
        test_path = os.path.join(self.output_dir, test_name)
        save = True
        if not replace:
            if os.path.exists(train_path) or os.path.exists(test_path):
                if os.path.exists(train_path):
                    print(f"Le fichier {train_path} existe déjà")
                if os.path.exists(test_path):
                    print(f"Le fichier {test_path} existe déjà")
                print(
                    "Sauvegarde des données non effectuée. Modifiez la version ou forcer avec replace=True"
                )
                save = False
        if save:
            if "SK_ID_CURR" in self.train.columns:
                index = False
            else:
                index = True
            self.train.to_csv(train_path, index=index)
            print(f"Le fichier {train_path} sauvegardé. Forme {self.train.shape}")
            if "SK_ID_CURR" in self.test.columns:
                index = False
            else:
                index = True
            self.test.to_csv(test_path, index=index)
            print(f"Le fichier {test_path} sauvegardé. Forme {self.test.shape}")

    # [TODO]
    def useless_features(self, fitted_model):
        pass

    """
    I. 
    NearMiss version 1 :
    Sélectionne les échantillons de la classe majoritaire qui ont la plus petite moyenne des distances par rapport aux trois plus proches voisins de la classe minoritaire.
    => Cette version privilégie les échantillons de la classe majoritaire qui sont les plus proches en moyenne des échantillons de la classe minoritaire.
    
    Sélectionne les échantillons de la classe majoritaire pour chaque échantillon de la classe minoritaire
    qui a les plus petits m plus proches voisins de la classe majoritaire. 
    Cette version est différente des deux premières car elle se concentre sur chaque échantillon de la classe minoritaire individuellement.
    
    II. 
    NearMiss version 2:
    Sélectionne les échantillons de la classe majoritaire qui ont la plus petite distance maximale 
    par rapport aux trois plus proches voisins de la classe minoritaire. 
    =>  version privilégie les échantillons de la classe majoritaire qui sont proches des trois plus proches voisins de la classe minoritaire.
    
    Privilégie les échantillons de la classe majoritaire qui
    sont relativement proches de n'importe quel échantillon de la classe minoritaire. 
    Cela peut parfois être plus agressif que NearMiss-1.

    III. 
    NearMiss version 3:
    Sélectionne les échantillons de la classe majoritaire pour chaque échantillon de la classe minoritaire qui a les plus petits m plus proches voisins de la classe majoritaire. 
    Cette version est différente des deux premières car elle se concentre sur chaque échantillon de la classe minoritaire individuellement.
    
    Cette version est plus individualisée, 
    sélectionnant des échantillons de la classe majoritaire basés sur chaque échantillon de la classe minoritaire, 
    ce qui peut aider à capturer des nuances spécifiques de la classe minoritaire.

    """

    # [TODO] Décider enfin si on met l'id SK_ID_CURR en index ou pas, si on le garde ou non danws la target y
    def undersample_nearmiss(self, version=1, inplace=False, verbose=True):
        X = self.train.drop(self.target, axis=1)
        y = self.train[self.target]
        n_rows = X.shape[0]

        near_miss = NearMiss(version=version)
        X_resampled, y_resampled = near_miss.fit_resample(X, y)

        df_resampled = pd.DataFrame(
            X_resampled, columns=X.columns, index=self.train.index
        )
        df_resampled["TARGET"] = y_resampled
        if verbose:
            print(
                f"Taille actuelle de Train : {self.train.shape}, {get_memory_consumed(self.train)} Mo"
            )
            print(
                f"L'UnderSampling version {version} éliminerait {n_rows - self.train.shape[0]} lignes"
            )
            print(
                f"Aboutirait à Train de taille  : {df_resampled.shape}, {get_memory_consumed(df_resampled, verbose=False)} Mo"
            )
        if inplace:
            self.train = df_resampled
            if verbose:
                print("UnderSampling de Train effectué")
            return
        else:
            return X_resampled, y_resampled


def sel_features_pairwise_correlation(
    df, method="spearman", ceil=0.95, sort_alpha=True, verbose=True
):
    check_no_nan(df)
    methods = ["pearson", "spearman", "kendall"]
    if method.lower() not in methods:
        raise ValueError(
            f"Les valeurs possibles de 'method' sont : {methods}, or method = {method}"
        )

    correlations = df.corr(method=method.lower())
    corr_array = correlations.to_numpy()
    # On positionne un zéro sur la diagonale,
    # cela nous permettra de sélectionner les features qui ont une corr trop grande sans se sélectionner elles-mêmes
    np.fill_diagonal(corr_array, 0.0)
    corr_cudf = cudf.DataFrame(
        corr_array, columns=correlations.columns, index=correlations.columns
    )
    corr_ceiled = corr_cudf[corr_cudf > ceil]
    corr_ceiled.dropna(axis=1, how="all", inplace=True)
    corr_ceiled.dropna(axis=0, how="all", inplace=True)
    too_correlated_features = corr_ceiled.columns.tolist()

    # On trie alphabétiquement le résultat si demandé
    if sort_alpha:
        too_correlated_features = sorted(too_correlated_features)
    if verbose:
        print(
            f"{len(too_correlated_features)} features ont un coefficient de corrélation {method}  > {ceil} avec au moins une autre :"
        )
        print(too_correlated_features)

    return too_correlated_features


# # ----------------Note Méthode Ward
# Même en utilisant les distances basées sur les corrélations, avec la méthode Ward (contrairement à Complete),
# On peut aboutir au final à des distances > 1
# En effet, Ward minimise la varaince intra-cluster et pour cela a une définition spécifique de la distance entre 2 clusters.
# Cette définition implique une transformation non linéaire de la distance initiale (comprise entre 0 et 1)
# Complete utilise la distance max entre les éléments de deux clusters, donc ça reste toujours entre 0 et 1
# ----------------Note Métrique
# Pour choisir entre métrique vbasée sur les corrélations ou clustering normal des features représentées dans l'espace des individus :
# Avantage corr : Directement liée à la similarité statistique entre les features (corrélations).
# Inconvénient corr : Nécessite une transformation des corrélations en distances, et les corrélations
#   peuvent ne pas capturer toutes les relations non linéaires entre les features.
# Avantage clust noraml : Peut capturer des similitudes complexes basées sur les valeurs individuelles des features, même si non linéaires.
# Inconvénient clust normal : Peut être plus sensible au bruit et aux valeurs aberrantes, moins intuitive que l'autre.
def cluster_features(
    data,
    to_process,
    corr_coef="spearman",
    threshold_pct=0.05,
    method="complete",
    figsize=None,
    to_mark=[],
):

    corr_coefficients = ["pearson", "spearman", "kendall", None]
    methods = ["ward", "complete", "average"]
    if corr_coef:
        corr_coef = corr_coef.lower()
    if corr_coef not in corr_coefficients:
        raise ValueError(
            f"Les valeurs possibles de 'corr_coef' sont : {corr_coefficients}, or corr_coef = {corr_coef}"
        )

    if method:
        method = method.lower()
    if method not in methods:
        raise ValueError(
            f"Les valeurs possibles de 'method' sont : {methods}, or method = {method}"
        )

    # La méthode n'accepte pas les NaN
    check_no_nan(data[to_process])

    # Pour pouvoir passer le seul en pourcentage sous la forme 0.01 ou 10%:
    if threshold_pct > 1:
        threshold_pct = threshold_pct / 100

    # On trie les features de to_process par ordre alphabétique car cela facilite la lecture et
    # les première features gardées parmi les clusters auront plus de chance d'être simples
    # car nom long = feature compliquée dans la plupart des cas.
    to_process = sorted(to_process)

    # Si corr_coeff n'est pas None, on effectue le regroupement par corrélations
    if corr_coef:
        # Métrique = 1-|corr| (2 features très corrélées sont très proches)
        corr = data[to_process].corr(method=corr_coef)
        # Correction pour avoir une matrice parfaitement symetrique
        corr = (corr + corr.T) / 2
        corr = corr.to_numpy()
        np.fill_diagonal(corr, 1.0)
        corr_dist = squareform(1 - abs(corr))

        # Réalisation du clustering hiérarchique
        if method == "complete":
            linkage_matrix = hierarchy.complete(corr_dist)
        elif method == "ward":
            linkage_matrix = hierarchy.ward(corr_dist)
        else:
            linkage_matrix = hierarchy.average(corr_dist)

    # Si corr_coeff est None, On effectue le clustering des features représentées dans l'espace des individus
    else:
        """agglo = AgglomerativeClustering(
            n_clusters=2, linkage="complete", metric="euclidean", compute_distances=True
        )
        # pred = agglo.fit_predict(X=matrix, y=None)
        agglo.fit(X=data[to_process].to_pandas().T, y=None)

        # On crée la matrice de linkage
        # create the counts of samples under each node
        counts = np.zeros(agglo.children_.shape[0])
        n_samples = len(agglo.labels_)
        for i, merge in enumerate(agglo.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [agglo.children_, agglo.distances_, counts]
        ).astype(float)"""

        # Calcul de la matrice de distance
        # Pour simplifier le code, finalement on utilise pdist de scipy
        X = data[to_process].to_numpy().T
        distance_matrix = pdist(X, metric="euclidean")

        # Calcul de la matrice de liaison
        linkage_matrix = hierarchy.linkage(squareform(distance_matrix), method=method)

        """
        Si on souhaitait normaliser les distances
        # On normalise les distances dans la matrice de liaison pour l'exprimer entre 0 et 1 (en % de la distance max)
        max_distance = np.max(linkage_matrix[:, 2])
        linkage_matrix[:, 2] /= max_distance"""

    # print("distance max :", np.max(linkage_matrix[:, 2]))
    dist_max = np.max(linkage_matrix[:, 2])
    threshold_value = threshold_pct * dist_max

    # Récupération des clusters à partir de la matrice de linkage et du seuil de distance
    cluster_ids = hierarchy.fcluster(
        linkage_matrix, threshold_value, criterion="distance"
    )
    # On crée un dictionnaire pour les clusters
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    # On récupère les clusters constitués de plus d'une variables (pour dropper les inutiles)
    clusters = [list(v) for v in cluster_id_to_feature_ids.values() if len(v) > 1]
    clusters_col = [list(data[to_process].columns[v]) for v in clusters]

    # On ne conserve que la première variable de chaque cluster, les autres sont retirées
    dropped_features = [data[to_process].columns[v[1:]] for v in clusters]
    dropped_features = [item for sublist in dropped_features for item in sublist]

    # On récupère tous les clusters (même si constitués d'une seule var) pour l'imprimer et le renvoyer
    all_clusters = [list(v) for v in cluster_id_to_feature_ids.values()]
    all_clusters_col = [list(data[to_process].columns[v]) for v in all_clusters]

    print(
        f"Au seuil de {threshold_pct:.0%} de la distance maximum (dist={threshold_value:.2f}): {len(all_clusters_col)} clusters :"
    )
    for cluster in all_clusters_col:
        print(f"\t{cluster}")
    print(f"{len(dropped_features)} variables à supprimer :", dropped_features)

    # ------------------ PLOT Dendrogramme

    # Si figsize est None, on calcule grossièrement la taille de la figure
    if not figsize:
        n_features = len(to_process)
        height_by_feature = 0.25
        title_margin = 0.5
        height = n_features * height_by_feature + title_margin
        width = 10
        figsize = (width, height)

    # On définit les titres
    if corr_coef:
        metric_name = f"1 - |corr| ({corr_coef.capitalize()})"
    else:
        metric_name = "distance euclidienne"
    title = f"Regroupement de features par clustering hiérarchique\nMéthode : Linkage={method.capitalize()}, métrique={metric_name}\n"

    fig, ax = plt.subplots(figsize=figsize)

    # plt.suptitle(title)
    ax.set_title(title, ha="left", x=0, fontsize=16)
    # ax.set_xlabel(f"Distance à chaque regroupement. Métrique : {metric_name}")
    ax.set_xlabel(f"Distance à chaque regroupement")

    dendro = hierarchy.dendrogram(
        linkage_matrix,
        orientation="right",
        ax=ax,
        labels=to_process,
    )
    # Ligne pour tracer le seuil
    ax.vlines(
        threshold_value,
        ymin=ax.get_ylim()[0],
        ymax=ax.get_ylim()[1],
        color="black",
        ls="--",
        lw=1,
        label="Seuil",
    )

    # On colorie en gris les labels des varaibles à dropper :
    if dropped_features:
        [
            t.set_color("grey")
            for t in ax.yaxis.get_ticklabels()
            if t.get_text() in dropped_features
        ]

    # S'il ya des variables à distinguer,
    # On récupère le texte des labels de l'axe des y et on colorie celui qui correspond
    for feature in to_mark:
        not_in_to_process = [f for f in to_mark if f not in to_process]
        if not_in_to_process:
            print(
                f"Les features {not_in_to_process} ne seront pas 'marquées' car elles ne font pas partie de 'to_process'"
            )
        to_mark = [f for f in to_mark if f in to_process]

    if to_mark:
        [
            t.set_color("red")
            for t in ax.yaxis.get_ticklabels()
            if t.get_text() in to_mark
        ]
        [
            t.set_fontweight("bold")
            for t in ax.yaxis.get_ticklabels()
            if t.get_text() in to_mark and t.get_text() not in dropped_features
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

    return all_clusters_col, dropped_features
