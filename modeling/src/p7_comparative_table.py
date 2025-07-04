import numpy as np
import pandas as pd
import os
import joblib
import time


from IPython.display import display
import warnings
from copy import deepcopy


from src.p7_evaluate import EvaluatorCV
from src.p7_util import format_time, search_item_regex
from src.p7_constantes import MODEL_DIR
from IPython.display import display


# Tableau comparatif utilisable dans les différents projets pour avoir un tableau synthétique des essais que l'on trace.
# (Différentes évaluation de modèles par exemple).
# Le tableau est géré sous forme de liste de dictionnaires.
# Chaque dictionnaire est une ligne du tableau et chaque clé du dictionnaire est une colonne du tableau
# Pour l'ADAPTER AU PROJET en cours, modifier la fonction get_record qui récupère les informations depuis un objet ou
# déclarer une classe enfant et surcharger get_record (utile si plusieurs types d'objets différents dans le même projet).
# Adapter l'affichage s'effectue à l'utilisation et non dans le code, avec .set_column_format()
class ComparativeTable:
    def __init__(self, title="Comparatif Modèles", table=None):
        self.title = title
        self.train_scores = []
        self.val_scores = []
        self.tags = []
        # Le tableau est une liste de dictionnaires
        # Chaque dictionnaire est une ligne du tableau
        if table:
            self.table = table
        else:
            self.table = []
        # Modifie l'ordre d'affichage des colonnes par la méthode display()
        # Si des colonnes ne font pas partie de laiste, elles ne seront pas affichées
        self.column_order = []
        # Format d'affichage par défaut des colonnes numériques
        self.default_format = "{:.4f}"
        # Modifie le format d'affichage par défaut des colonnes indiquées en clé du dictionnaire
        self.custom_format = {}
        # format d'affichage par défaut des colonnes durées indiquées en clés
        self.duration_format = {"time_fit": "hh:mm:ss", "fit_time": "hh:mm:ss"}

    # Prend les informations d'un objet pour les mettre dans le taleau comparatif
    # A modifier selon le projet ou classe fille et surcharger.
    # tags permet de rajouter des colonnes définie manuellement, par exemple une description...
    def get_record(self, evaluator, tags={}):
        model_name = evaluator.model.__class__.__name__

        # Scores de validation
        # val_scores = {f"{k}_val": v for k, v in evaluator.mean_val_scores.items()}
        if isinstance(evaluator, EvaluatorCV):
            val_scores = evaluator.mean_val_scores
        else:
            val_scores = evaluator.test_scores
        param_bg = evaluator.param_bg
        record = {**tags, **val_scores, **param_bg}

        record["model_name"] = model_name
        record["n_feat."] = evaluator.n_features
        record["n_rows"] = evaluator.n_rows
        record["threshold"] = evaluator.threshold_prob
        record["balancing"] = evaluator.balance_str
        return record

    # Ajouter une ligne depuis un objet dans le tableau comparatif
    def add(self, evaluator, tags={}):
        record = self.get_record(evaluator, tags)
        self.table.append(record)
        return

    # Insère une ligne avant un autre dans le tableau comparatif
    def add_before(self, index, evaluator, tags={}):
        try:
            list_before_index = self.table[:index].copy()
            list_after_index = self.table[index:].copy()
            record = self.get_record(evaluator, tags)
            list_before_index.append(record)
            self.table = list_before_index + list_after_index
        except:
            print(
                f"Impossible d'insérer avant l'index {index}. (Max index = {len(self.table) - 1})"
            )
        return

    # Supprime une ligne du tableau comparatif grâce à son index
    def drop(self, index):
        try:
            self.table.pop(index)
        except:
            print(
                f"Impossible de supprimer l'index {index}. (Max index = {len(self.table) - 1})"
            )
        return

    # Définit les colonnes à afficher et la façon de les ordonner dans le tableau comparatif
    # Les colonnes non indiquées dans la fonction ne seront pas affichées même si elles sont
    # enregistrées grâce à get_record
    def set_column_order(self, column_order):
        self.column_order = column_order

    # Format d'affichage par défaut de tous les float dans le tableau comparatif
    def set_default_format(self, format_string="{:.4f}"):
        self.default_format = format_string
        return

    # Format d'affichage personnalisé qui écrase le format par défaut.
    # A modifier selon le projet ou à surcharger dans une classe fille si plusieurs types d'objets.
    def set_column_format(self, dic_format={"fit_time": "{:.1f}"}):
        self.custom_format = dic_format

    # Définit un format d'affichage par défaut pour les durées,
    # écrase le format d'affichage par défaut d'un float
    def set_duration_format(self, dic_format={"fit_time": "hh:mm:ss"}):
        self.duration_format = dic_format

    # Affiche le tableau comparatif avec le format d'affichage que l'on souhaite
    # A adapter selon le projet ou à surcharger dans une classe fille
    def display(self, sort_by=[], format_duration=False):
        if self.title:
            to_print = self.title
            if sort_by:
                to_print += f" trié par {sort_by}"
            print(to_print)

        df = pd.DataFrame(self.table)

        if self.column_order:
            df = df.reindex(columns=self.column_order)

        if sort_by:
            df = df.sort_values(by=sort_by, ascending=False)

        formatted_df = df.copy()

        # On formate toutes colonnes floats par le formattage par défaut
        float_cols = formatted_df.select_dtypes(include="float").columns
        formatted_df[float_cols] = formatted_df[float_cols].applymap(
            lambda x: self.default_format.format(x)
        )

        # On formate les colonnes de type durées
        for col in self.duration_format.keys():
            if col in formatted_df.columns:
                if self.duration_format[col] == "hh:mm:ss":
                    formatted_df[col] = df[col].apply(lambda x: format_time(x))
                else:
                    formatted_df[col] = df[col].apply(
                        lambda x: self.duration_format[col].format(x)
                    )

        # Formattage des colonnes définies en custom
        if self.custom_format:
            for col in self.custom_format.keys():
                if col in formatted_df.columns:
                    formatted_df[col] = df[col].apply(
                        lambda x: self.custom_format[col].format(x)
                    )
        display(formatted_df)
        return

    # Sauvegarde le tableau uniquement (sous forme liste de dictionnaires)
    # car si la classe est modifiée au cours du projet, on ne pourra pas relire l'objet créé avec cette classe.
    def save(self, filename, directory=MODEL_DIR, save_index=False):
        filepath = os.path.join(directory, f"{filename}.pkl")
        joblib.dump(self.table, filepath)
        print(f"Tableau comparatif enregistré dans {filepath}")
        return
