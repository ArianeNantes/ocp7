"""
Le code pour la construction des données a été repris (et adapté) de :
https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features

La création de nouvelles features a aussi été reprise en partie de :
https://github.com/js-aguiar/home-credit-default-competition
"""

# Lien : https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script
# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables.
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import numpy as np
import pandas as pd
import cudf
import cupy as cp
import lightgbm as lgb
import re
import gc
import os
import time
import psutil
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from cuml.model_selection import train_test_split as cu_train_test_split

from src.p7_util import reduce_memory_cudf
from src.p7_preprocess import VarianceSelector, get_binary_features
from src.p7_preprocess import train_test_split_nan
from src.p7_file import make_dir
from src.p7_util import read_pkl

# from modeling.src.p7_regex import sel_var

warnings.simplefilter(action="ignore", category=FutureWarning)

from src.p7_constantes import (
    DATA_CLEAN_DIR,
    DATA_BASE,
    DATA_INTERIM,
    MODEL_DIR,
    VAL_SEED,
    # GENERATE_SUBMISSION_FILES,
    # STRATIFIED_KFOLD,
    # NUM_FOLDS,
    # EARLY_STOPPING,
)

from src.p7_util import timer


class DataSimple:
    def __init__(self, dataset_num="01", debug=False) -> None:
        self.dataset_num = dataset_num
        self.debug = debug
        self.n_rows = 30_000 if self.debug else None
        self.drop_first = True
        self.frac_test = 0.25
        self.test_is_stratified = True
        self.input_dir = DATA_BASE
        self.output_dir = DATA_INTERIM
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.train_name = f"{self.dataset_num}_v0_built_train.csv"
        self.test_name = f"{self.dataset_num}_v0_built_test.csv"
        # self.na_value = cudf.NA
        self.na_value = np.nan
        # self.oh_columns_name = f"{self.dataset_num}_v0_built_oh_columns.pkl"
        # Features à encoder en label encoding
        self.features_label_encoding = []
        # Features encodées en One_Hot
        # self.features_oh_encoded = []
        # Features encodées en One_Hot
        self.features_oh_encoded = []
        # liste des features provenant de la première modalité des variables encodées OH
        self._first_categories = []
        self.random_state = VAL_SEED
        self._device = "cuda"

        self._train_idx = None
        self._test_idx = None

        # Dictionnaire utilisé pour les features non encodées en One Hot de la table Application,
        # afin d'effectuer toujours les mêmes remplacements
        self.mapping_dicts = {
            "CODE_GENDER": {"M": 0, "F": 1, "XNA": -1},
            "FLAG_OWN_CAR": {"N": 0, "Y": 1},
            "FLAG_OWN_REALTY": {"N": 1, "Y": 0},
            "EMERGENCYSTATE_MODE": {"No": 0, "Yes": 1},
        }

    @property
    def public_attributes(self):
        return [k for k in self.__dict__.keys() if not k.startswith("_")]

    def get_data(self, filename="application_train.csv"):
        if self.debug:
            print(f"DEBUG num_rows={self.n_rows}")

        # On efface les first categories des éventuels appels précédents
        self._first_categories = []

        with timer("Process Application"):
            df = self.get_application(filename=filename)

        with timer("Process bureau and bureau_balance"):
            bureau = self.get_bureau_and_balance()
            print("Bureau df shape:", bureau.shape)
            # pas de join avec 'on' en cuda
            # df = df.join(bureau, how="left", on="SK_ID_CURR")
            df = df.merge(bureau, how="left", on="SK_ID_CURR")
            del bureau
            gc.collect()
            cp._default_memory_pool.free_all_blocks()
        with timer("Process previous_applications"):
            prev = self.get_previous_applications(nan_as_category=True)
            print("Previous applications df shape:", prev.shape)
            df = df.merge(prev, how="left", on="SK_ID_CURR")
            del prev
            gc.collect()
            cp._default_memory_pool.free_all_blocks()
        with timer("Process POS-CASH balance"):
            pos = self.get_pos_cash(nan_as_category=True)
            print("Pos-cash balance df shape:", pos.shape)
            df = df.merge(pos, how="left", on="SK_ID_CURR")
            del pos
            gc.collect()
            cp._default_memory_pool.free_all_blocks()

        with timer("Process installments payments"):
            ins = self.get_installments_payments(nan_as_category=True)
            print("Installments payments df shape:", ins.shape)
            df = df.merge(ins, how="left", on="SK_ID_CURR")
            del ins
            gc.collect()
            cp._default_memory_pool.free_all_blocks()
        with timer("Process credit card balance"):
            cc = self.get_credit_card_balance(nan_as_category=True)
            print("Credit card balance df shape:", cc.shape)
            df = df.merge(cc, how="left", on="SK_ID_CURR")
            del cc
            gc.collect()
            cp._default_memory_pool.free_all_blocks()

            """to_drop = sel_var(df.columns, "Unnamed", verbose=False)
            if to_drop:
                df = df.drop(to_drop, axis=1)"""

        print("All data shape :", df.shape)
        return df

    # Preprocess application_train.csv
    def get_application(self, filename="application_train.csv"):
        # Read data, on ne lit pas application_test réservé à la compétition Kaggle. Nous recrérons un jeu de test avec target
        df = cudf.read_csv(
            os.path.join(DATA_BASE, filename),
            nrows=self.n_rows,
            na_values=["XNA", "Unknown"],
            # Pour avoir des cuDF.NA partout (même si NaN dans le csv)
            keep_default_na=True,
        )
        if filename == "application_train.csv":
            print("Nombre de lignes Application train + test: {}".format(len(df)))

        # Application_test ne possède pas de target. Il n'est pas utilisé pour la modélisation mais
        # pour le data drift et l'API
        elif filename == "application_test.csv":
            print(
                "Nombre de lignes pour les demandes de prêts récentes: {}".format(
                    len(df)
                )
            )

        # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
        # Les valeurs manquantes sont toutes des Non defaut
        # df = df[df["CODE_GENDER"] != "XNA"]

        # Categorical features with Binary encode (0 or 1; two categories)
        # FLAG_OW_CAR et FLAG_OWN_REALTY n'ont aucune valeur manquante
        # On n'utilise pas factorize car l'ordre d'encodage dépend de l'ordre d'apparition des valeurs
        # A la place on utilise un dictionnaire de remplacement qui sera toujours le même quelque soit le nouveau jeu de données.
        """for bin_feature in [
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "EMERGENCYSTATE_MODE",
        ]:
            df[bin_feature], uniques = cudf.factorize(df[bin_feature])"""

        f"""or col, mapping in self.mapping_dicts.items():
            df[col] = df[col].replace(mapping)"""

        f"""or col, mapping in self.mapping_dicts.items():
            if col in df.columns:
                # Convertir les clés ET valeurs en str pour satisfaire cuDF
                mapping_str = 

                # Appliquer le remplacement
                df[col] = df[col].replace(mapping_str)

                # Convertir ensuite les résultats en int (pas possible en cudf de le faire immédiatement)
                # En effet en contrairement à PAndas,cudf.Series.replace() ne supporte pas un mapping dict si les clés sont des chaînes (str) et
                # les valeurs sont des entiers (int64). Il faut tout mettre en str (clés et valeurs puis convertir après coup)
                if self.na_value is not np.nan and not df[col].isnull().any():
                    df[col] = df[col].astype(
                        "int64"
                    )  # attention : .astype(np.int64) peut poser souci avec cudf"""
        for col, mapping in self.mapping_dicts.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        df["CODE_GENDER"] = df["CODE_GENDER"].astype(int)

        # CODE_GENDER et EMERGENCYSTATE_MODE ont des NaN
        # si na_value est np.nan, elle est en float64. On ne remplace les valeurs -1 issus de factorize par des na_value
        # si na_value est cudf.NA, les na sont très bien gérés dans les cudf même pour les bool (pas de type mixte ni object dans les cudf),
        # par contre on retrouvera des object pour les bool contenant des NA quand on transformera en pandas via .to_pandas()
        if self.na_value is not np.nan:
            df["CODE_GENDER"].replace(-1, self.na_value, inplace=True)
            df["EMERGENCYSTATE_MODE"].replace(-1, self.na_value, inplace=True)
            # NaN values for DAYS_LAST_PHONE_CHANGE: 0 -> nan
            df["DAYS_LAST_PHONE_CHANGE"].replace(0, self.na_value, inplace=True)
        else:
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].astype(np.float64)

        # NaN values for DAYS_EMPLOYED: 365_243 -> nan (équivaut à 1_000 ans)
        df["DAYS_EMPLOYED"].replace(365243, self.na_value, inplace=True)

        # Some simple new features (percentages)
        # [TODO] Renommer ces RATIOS calculés
        # df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
        # df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
        # df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

        # Credit ratios
        # Equivallent de PAYMENT_RATE
        df["CREDIT_TO_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
        df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
        # Income ratios
        df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        df["INCOME_TO_BIRTH_RATIO"] = df["AMT_INCOME_TOTAL"] / df["DAYS_BIRTH"]

        # On encode les features categorical en One Hot (toutes les variables de type object qui ne sont pas à encoder avec du labelencoding)
        to_oh_encode = [
            f
            for f in df.select_dtypes(include="object").columns.tolist()
            if f not in self.features_label_encoding
        ]
        df, _ = self.one_hot_encoder(
            df,
            to_encode=to_oh_encode,
            drop_first=self.drop_first,
        )

        # S'il y a des features à encoder en label encoding, on les encode
        if self.features_label_encoding:
            df, _ = self.label_encoder(df, self.features_label_encoding)
        return df

    # Preprocess bureau.csv and bureau_balance.csv
    def get_bureau_and_balance(self):
        bureau = cudf.read_csv(os.path.join(DATA_BASE, "bureau.csv"), nrows=self.n_rows)
        bb = cudf.read_csv(
            os.path.join(DATA_BASE, "bureau_balance.csv"), nrows=self.n_rows
        )
        bb, bb_cat = self.one_hot_encoder(bb, drop_first=False, nan_as_category=True)
        bureau, bureau_cat = self.one_hot_encoder(
            bureau, drop_first=False, nan_as_category=True
        )

        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}
        for col in bb_cat:
            bb_aggregations[col] = ["mean"]
        bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
        # pas de tolist en cuda, on la construit à la main
        bb_agg_columns_list = [f for f in bb_agg.columns]
        bb_agg.columns = cudf.Index(
            [e[0] + "_" + e[1].upper() for e in bb_agg_columns_list]
        )
        # En cuda, le 'on' dans le join n'est pas encore supporté. Donc on remplace par un 'merge'
        bureau = bureau.merge(bb_agg, how="left", on="SK_ID_BUREAU")
        # bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
        bureau.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
        del bb, bb_agg
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

        # Bureau and bureau_balance numeric features
        num_aggregations = {
            "DAYS_CREDIT": ["min", "max", "mean", "var"],
            "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
            "DAYS_CREDIT_UPDATE": ["mean"],
            "CREDIT_DAY_OVERDUE": ["max", "mean"],
            "AMT_CREDIT_MAX_OVERDUE": ["mean"],
            "AMT_CREDIT_SUM": ["max", "mean", "sum"],
            "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
            "AMT_CREDIT_SUM_OVERDUE": ["mean"],
            "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
            "AMT_ANNUITY": ["max", "mean"],
            "CNT_CREDIT_PROLONG": ["sum"],
            "MONTHS_BALANCE_MIN": ["min"],
            "MONTHS_BALANCE_MAX": ["max"],
            "MONTHS_BALANCE_SIZE": ["mean", "sum"],
        }
        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat:
            cat_aggregations[cat] = ["mean"]
        for cat in bb_cat:
            cat_aggregations[cat + "_MEAN"] = ["mean"]

        bureau_agg = bureau.groupby("SK_ID_CURR").agg(
            {**num_aggregations, **cat_aggregations}
        )
        # pas de tolist en cuda on crée donc manuellement la liste bureau_agg.columns.tolist()
        bureau_agg_columns_list = [f for f in bureau_agg.columns]
        bureau_agg.columns = cudf.Index(
            ["BURO_" + e[0] + "_" + e[1].upper() for e in bureau_agg_columns_list]
        )
        # Bureau: Active credits - using only numerical aggregations
        active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
        # active = bureau[bureau["CREDIT_ACTIVE_Active"] == True]
        active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
        # pas de tolist en cuda on crée donc manuellement la liste active_agg.columns.tolist()
        active_agg_columns_list = [f for f in active_agg.columns]
        active_agg.columns = cudf.Index(
            ["ACTIVE_" + e[0] + "_" + e[1].upper() for e in active_agg_columns_list]
        )
        # Pas de join avec On en Cuda, on remplace par un merge
        bureau_agg = bureau_agg.merge(active_agg, how="left", on="SK_ID_CURR")
        # bureau_agg = bureau_agg.join(active_agg, how="left", on="SK_ID_CURR")
        del active, active_agg
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        # Bureau: Closed credits - using only numerical aggregations
        closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == True]
        closed_agg = closed.groupby("SK_ID_CURR").agg(num_aggregations)
        closed_agg_columns_list = [f for f in closed_agg.columns]
        closed_agg.columns = cudf.Index(
            ["CLOSED_" + e[0] + "_" + e[1].upper() for e in closed_agg_columns_list]
        )
        # bureau_agg = bureau_agg.join(closed_agg, how="left", on="SK_ID_CURR")
        bureau_agg = bureau_agg.merge(closed_agg, how="left", on="SK_ID_CURR")
        del closed, closed_agg, bureau
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        return bureau_agg

    # Preprocess previous_applications.csv
    def get_previous_applications(self, nan_as_category=True):
        prev = cudf.read_csv(
            os.path.join(DATA_BASE, "previous_application.csv"), nrows=self.n_rows
        )
        prev, cat_cols = self.one_hot_encoder(
            prev, drop_first=False, nan_as_category=nan_as_category
        )
        # Days 365.243 values -> nan
        prev["DAYS_FIRST_DRAWING"].replace(365243, self.na_value, inplace=True)
        prev["DAYS_FIRST_DUE"].replace(365243, self.na_value, inplace=True)
        prev["DAYS_LAST_DUE_1ST_VERSION"].replace(365243, self.na_value, inplace=True)
        prev["DAYS_LAST_DUE"].replace(365243, self.na_value, inplace=True)
        prev["DAYS_TERMINATION"].replace(365243, self.na_value, inplace=True)
        # Add feature: value ask / value received percentage
        prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
        # Previous applications numeric features
        num_aggregations = {
            "AMT_ANNUITY": ["min", "max", "mean"],
            "AMT_APPLICATION": ["min", "max", "mean"],
            "AMT_CREDIT": ["min", "max", "mean"],
            "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
            "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
            "AMT_GOODS_PRICE": ["min", "max", "mean"],
            "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
            "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
            "DAYS_DECISION": ["min", "max", "mean"],
            "CNT_PAYMENT": ["mean", "sum"],
        }
        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ["mean"]

        prev_agg = prev.groupby("SK_ID_CURR").agg(
            {**num_aggregations, **cat_aggregations}
        )
        prev_agg.columns = cudf.Index(
            ["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()]
        )
        # Previous Applications: Approved Applications - only numerical features
        approved = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
        approved_agg = approved.groupby("SK_ID_CURR").agg(num_aggregations)
        approved_agg_columns_list = [f for f in approved_agg.columns]
        approved_agg.columns = cudf.Index(
            ["APPROVED_" + e[0] + "_" + e[1].upper() for e in approved_agg_columns_list]
        )
        # prev_agg = prev_agg.join(approved_agg, how="left", on="SK_ID_CURR")
        prev_agg = prev_agg.merge(approved_agg, how="left", on="SK_ID_CURR")

        # Previous Applications: Refused Applications - only numerical features
        refused = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1]
        refused_agg = refused.groupby("SK_ID_CURR").agg(num_aggregations)
        refused_agg_columns_list = [f for f in refused_agg.columns]
        refused_agg.columns = cudf.Index(
            ["REFUSED_" + e[0] + "_" + e[1].upper() for e in refused_agg_columns_list]
        )
        # prev_agg = prev_agg.join(refused_agg, how="left", on="SK_ID_CURR")
        prev_agg = prev_agg.merge(refused_agg, how="left", on="SK_ID_CURR")
        del refused, refused_agg, approved, approved_agg, prev
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        return prev_agg

    # Preprocess POS_CASH_balance.csv
    def get_pos_cash(self, nan_as_category=True):
        pos = cudf.read_csv(
            os.path.join(DATA_BASE, "POS_CASH_balance.csv"), nrows=self.n_rows
        )
        pos, cat_cols = self.one_hot_encoder(
            pos, drop_first=False, nan_as_category=nan_as_category
        )
        # Features
        aggregations = {
            "MONTHS_BALANCE": ["max", "mean", "size"],
            "SK_DPD": ["max", "mean"],
            "SK_DPD_DEF": ["max", "mean"],
        }
        for cat in cat_cols:
            aggregations[cat] = ["mean"]

        pos_agg = pos.groupby("SK_ID_CURR").agg(aggregations)
        pos_agg.columns = cudf.Index(
            ["POS_" + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()]
        )
        # Count pos cash accounts
        pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
        del pos
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        return pos_agg

    # Preprocess installments_payments.csv
    def get_installments_payments(self, nan_as_category=True):
        ins = cudf.read_csv(
            os.path.join(DATA_BASE, "installments_payments.csv"), nrows=self.n_rows
        )
        ins, cat_cols = self.one_hot_encoder(
            ins, drop_first=False, nan_as_category=nan_as_category
        )
        # Percentage and difference paid in each installment (amount paid and installment value)
        ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
        ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
        # Days past due and days before due (no negative values)
        ins["DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
        ins["DBD"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]
        # Les cuDF ne supportent pas les .apply avec des fonctions lambdas
        # ins["DPD"] = ins["DPD"].apply(lambda x: x if x > 0 else 0)
        # ins["DBD"] = ins["DBD"].apply(lambda x: x if x > 0 else 0)
        ins["DPD"] = ins["DPD"].clip(lower=0)
        ins["DBD"] = ins["DBD"].clip(lower=0)
        # Features: Perform aggregations
        aggregations = {
            "NUM_INSTALMENT_VERSION": ["nunique"],
            "DPD": ["max", "mean", "sum"],
            "DBD": ["max", "mean", "sum"],
            "PAYMENT_PERC": ["max", "mean", "sum", "var"],
            "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
            "AMT_INSTALMENT": ["max", "mean", "sum"],
            "AMT_PAYMENT": ["min", "max", "mean", "sum"],
            "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
        }
        for cat in cat_cols:
            aggregations[cat] = ["mean"]
        ins_agg = ins.groupby("SK_ID_CURR").agg(aggregations)
        ins_agg.columns = cudf.Index(
            ["INSTAL_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()]
        )
        # Count installments accounts
        ins_agg["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()
        del ins
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        return ins_agg

    # Preprocess credit_card_balance.csv
    def get_credit_card_balance(self, nan_as_category=True):
        cc = cudf.read_csv(
            os.path.join(DATA_BASE, "credit_card_balance.csv"), nrows=self.n_rows
        )

        # General aggregations
        cc.drop(["SK_ID_PREV"], axis=1, inplace=True)
        # Les aggrégations générant des MIN et des MAX sur des variables booléennes n'ont aucun sens,
        # De plus le merge ultérieur transformera ces variables booléennes en
        # object, ce qui causera une erreur pour le fit du lgbm
        # On sépare donc le traitement des variables booléennes générées par le One Hot des autres variables
        cc_agg_num = (
            cc.select_dtypes(exclude="object")
            .groupby("SK_ID_CURR")
            .agg(["min", "max", "mean", "sum", "var"])
        )
        cc, cat_cols = self.one_hot_encoder(
            cc, drop_first=False, nan_as_category=nan_as_category
        )
        cc_agg_cat = (
            cc[["SK_ID_CURR"] + cat_cols]
            .groupby("SK_ID_CURR")
            .agg(["mean", "sum", "var"])
        )
        cc_agg = cc_agg_num.merge(cc_agg_cat, how="left", on="SK_ID_CURR")
        cc_agg.columns = cudf.Index(
            ["CC_" + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()]
        )

        del cc_agg_num
        del cc_agg_cat
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

        # Count credit card lines
        cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
        del cc
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

        return cc_agg

    def inf_to_nan(self, df, verbose=True):
        if verbose:
            # On identifie les colonnes comportant des NaN (parmi les float)
            # Il n'y a des inf que dans les float
            infinite_columns = []
            for col in df.select_dtypes("float").columns:
                if (df[col] == np.inf).any() or (df[col] == -np.inf).any():
                    infinite_columns.append(col)
            print(f"{len(infinite_columns)} features comportent des valeurs infinies :")
            print(infinite_columns)
        # df.select_dtypes("float").replace([np.inf, -np.inf], np.nan, inplace=True)
        df.select_dtypes("float").replace(
            [np.inf, -np.inf], self.na_value, inplace=True
        )
        return df

    def del_null_std(self, df, train_idx, verbose=True):
        train = df.loc[train_idx, df.columns]
        selector = VarianceSelector(verbose=False)
        selector.fit(train)
        del train
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        if verbose:
            print(
                f"{len(selector.to_drop_)} features de variance nulle dans le Train supprimées"
            )
            if selector.to_drop_:
                print(selector.to_drop_)
        if selector.to_drop_:
            return df.drop(selector.to_drop_, axis=1)
        else:
            return df

    def del_empty_col(self, df, train_idx, verbose=True):
        empty_col_train = list(
            df.columns[df.iloc[train_idx].isna().to_numpy().all(axis=0)]
        )
        if empty_col_train is not None:
            if len(empty_col_train) > 0:
                if verbose:
                    print(
                        f"Suppression de {len(empty_col_train)} colonnes vides sur le Train {empty_col_train}"
                    )
                return df.drop[empty_col_train]
            else:
                if verbose:
                    print("Aucune colonne complètement vide sur le Train")
                return df
        else:
            print("WARNING : Les colonnes vides n'ont pas pu être vérifiées")
            return df

    def one_hot_encoder(
        self,
        df,
        to_encode=None,
        nan_as_category=True,
        drop_first=False,
    ):
        """Crée une nouvelle colonne pour chaque modalité des variables qualitatives,
        Stocke le nom des colonnes encodées et le nom des colonnes correspondant à leur première modalité

        Args:
            df (pd.DataFrame ou cudf.DataFrame): DataFrame contenant des variables à encoder
            to_encode (list ou index de colonne, optional): variables à encoder. Si None, toutes les variables de type 'object'. Defaults to None.
            nan_as_category (bool, optional): Si vrai crèe une colonne pour les valeurs manquantes. Defaults to True.
            drop_first (bool, optional): Si vrai Supprime la premère catégorie. Defaults to False.

        Returns:
            (DF ou cuDF, list): DataFrame avec les variables encodées et la liste des noms de colonnes encodées
        """
        original_columns = list(df.columns)
        encoded_columns = []

        # Si to_encode n'est pas spécifié on encode toutes les variables de type object
        if not to_encode:
            to_encode = [col for col in df.columns if df[col].dtype == "object"]
        if to_encode:
            # Supprimer la première catégorie est impératif pour les modèles linéaires
            # first_categories = []
            if isinstance(df, (pd.DataFrame, pd.Series)):
                """first_categories = [
                    f"{col}_{np.unique(df[col].dropna())[0]}" for col in to_encode
                ]"""

                df = pd.get_dummies(
                    df,
                    columns=to_encode,
                    dummy_na=nan_as_category,
                    drop_first=drop_first,
                )

            # Si df n'est pas pandas, on considère qu'il s'agit d'un cudf
            # Drop first n'est pas encore supporté dans RAPIDS, donc on encode avec drop_first=False, puis
            # on supprime la first cat
            else:
                first_categories = [
                    f"{col}_{np.unique(df[col]).to_numpy()[0]}" for col in to_encode
                ]
                df = cudf.get_dummies(
                    df,
                    columns=to_encode,
                    dummy_na=nan_as_category,
                )
                if drop_first:
                    df = df.drop(first_categories, axis=1)

            # Juste pour le debug on stocke les first_catégories qui ont été droppées
            if drop_first:
                self._first_categories = self._first_categories + first_categories
            # On stocke le nom des variables créées par l'encodage
            encoded_columns = [col for col in df.columns if col not in original_columns]
            self.features_oh_encoded = self.features_oh_encoded + encoded_columns

        return df, encoded_columns

    def label_encoder(self, df, to_encode):
        """Encode categorical values as integers (0,1,2,3...) with pandas.factorize."""
        for col in to_encode:
            df[col], uniques = pd.factorize(df[col])

            # Ajout
            """if np.min(df[col]) == -1:
                print(f"NaN rencontrées dans {col} converties en -1")"""
        return df, to_encode

    def cast_and_optimize(self, df, verbose=True):
        # On caste d'abord les features réellement binaires (uniquement 0 ou 1 sur train+test et qui pas fonction d'agrégation)
        # On n'optimise pas la mémoire à caster des int8 en bool, mais cela facilite le travail si on souhaite imputer différemment selon le type
        features = [col for col in df.columns if col not in ["SK_ID_CURR", "TARGET"]]
        not_bool_features = list(df.select_dtypes(exclude="bool").columns)
        bin_not_bool_features = get_binary_features(df[not_bool_features])
        to_cast_bool = [
            f for f in bin_not_bool_features if not f.endswith(("_SUM", "_MIN", "_MAX"))
        ]
        df[to_cast_bool] = df[to_cast_bool].astype("bool")
        if verbose:
            print(
                f"{len(to_cast_bool)} features binaires castées en bool (exclusion _SUM, _MAX, _MIN):"
            )
            print(to_cast_bool)
            print("\nCast des autres types pour optimisation")

        # On optimise la mémoire en castant le reste (on évite les int16 et les fload16 en CUDA)
        df = self.reduce_memory_cudf(df, verbose=verbose)
        return df

    def reduce_memory_cudf(self, df, verbose=True):
        """Réduit l'utilisation mémoire d'un DataFrame cuDF."""
        # Adaptation du code pour cuDF
        # df.loc[:, col] = ... est moins efficace avec cuDF. Il est préférable d’utiliser df[col]
        # df.memory_usage() existe, mais pas toujours avec les mêmes options.
        # object n’est pas utilisé dans cuDF (il n’y a pas de colonnes mixtes de strings ou objets arbitraires)
        # Certaines conversions peuvent ne pas être supportées directement (comme vers float16)
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        print("Utilisation mémoire du cuDF {:.2f} Mo".format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            if col_type == "bool":  # rien à faire, elles sont déjà optimales
                continue

            try:
                c_min = df[col].min()
                c_max = df[col].max()
            except Exception as e:
                print(f"Impossible d'évaluer min/max pour {col}: {e}")
                continue

            if cp.issubdtype(col_type, cp.integer):
                """# On n'a plus de int8, ce sont des bool
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype('int8')
                #int16 sera moins performant que int32 sur GPU et va nous gêner dans les algorithmes cuML (conversion en interne car + performant ou obligation de caster)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype('int16')"""

                if c_min >= cp.iinfo(np.int8).min and c_max <= cp.iinfo(cp.int32).max:
                    df[col] = df[col].astype("int32")
                else:
                    df[col] = df[col].astype("int64")

            elif cp.issubdtype(col_type, cp.floating):
                # float16 est souvent moins supporté ou moins précis en GPU (donc élimine les float16)
                if (
                    c_min >= cp.finfo(cp.float16).min
                    and c_max <= cp.finfo(cp.float32).max
                ):
                    df[col] = df[col].astype("float32")
                else:
                    df[col] = df[col].astype("float64")

            # Pour les strings, cuDF ne permet pas de conversion directe pour réduire la mémoire
            # object n’est pas utilisé dans cuDF (il n’y a pas de colonnes mixtes de strings ou objets arbitraires)
            """elif np.issubdtype(col_type, np.object_):
                pass  # skip, cuDF strings restent en strings"""

        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        if verbose:
            print("Consommation Mémoire après optimisation : {:.2f} Mo".format(end_mem))
            print(
                "Réduction de {:.1f}%".format(100 * (start_mem - end_mem) / start_mem)
            )
        return df

    def clean_data(self, df, verbose=True):
        print("Renommage des colonnes")
        df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
        self.features_oh_encoded = [
            re.sub("[^A-Za-z0-9_]+", "", col) for col in self.features_oh_encoded
        ]
        print("\nTraitement des valeurs inf")
        df = self.inf_to_nan(df, verbose=verbose)

        # Vérification des doublons sur la clé
        to_check = "SK_ID_CURR"
        if df[to_check].duplicated().any():
            print(f"WARNING Des doublons existent dans la colonne {to_check}")
        else:
            print(f"Aucun doublon dans la colonne {to_check}")

        print(f"\nCréation des index Train / Test {self.frac_test:.1%}")
        print("\nSuppression des colonnes de variance nulle sur le Train")
        self.set_train_test_idx(df)
        df = self.del_null_std(df, self._train_idx, verbose=verbose)
        self.features_oh_encoded = [
            col for col in self.features_oh_encoded if col in df.columns
        ]
        print("\nSuppression des colonnes vides sur le Train")
        df = self.del_empty_col(df, self._train_idx, verbose=verbose)

        print("\nall_data.info :")
        df.info()
        return df

    def get_train_and_test(self, df, verbose=True):
        if verbose:
            print("\nSéparation réelle du train et du test")
        if self._train_idx is None or self._test_idx is None:
            print(
                "L'échantillonage n'est pas créé, utiliser d'abord la fonction .set_train_test_idx()"
            )
            return
        else:
            train = df.iloc[self._train_idx]
            test = df.iloc[self._test_idx]
            if verbose:
                print(
                    f"Train : {1 - self.frac_test:.1%} des données - Forme : {train.shape} - Mem : {train.memory_usage().sum() / 1024**2:.2f} Mo"
                )
                print(
                    f"Test : {self.frac_test:.1%} des données - Forme : {test.shape} - Mem : {test.memory_usage().sum() / 1024**2:.2f} Mo"
                )
            return train, test

    # Attention les df de pandas et les cudf de numl n'utilisent pas les mêmes indexers.
    # Risque de bugs pas visibles immédiatement
    # =>  Ne pas utiliser de splits faits en sklearn pour des cudf etc. (Ou convertir les index positionnels/relatifs et vice versa)
    # Or train_test_split de cuml n'accèpte pas les NaN
    def set_train_test_idx(self, df, y=None):
        # Si y n'est pas fournie, on considère qu'il s'agit de la colonne nommée "TARGET" dans le df
        if y is None:
            if "TARGET" not in df.columns:
                print("TARGET n'est pas dans le dataset, veuillez préciser y")
                return
            else:
                y = df["TARGET"]
                X = df.drop("TARGET", axis=1)
        # Si la target a été fournie en param, on n'enlève pas 'TARGET' du df
        else:
            X = df

        # Il nous faut au moins 2 colonnes dans le X sans NaN
        columns_without_nan = [c for c in X.columns if X[c].isna().sum() == 0]
        if len(columns_without_nan) < 2:
            print("Le dataset doit comporter au moins 2 colonnes sans NaN")
            return
        # Si on a suffisemment de colonnes sans nan, on effectue le split sur 2 colonnes pour récupérer les index des splits
        # Attention à ne pas passer par un index pandas, via sklearn, car ce ne sont pas les mêmes indexers
        X_train_tmp, X_test_tmp, y_train, y_test = cu_train_test_split(
            X[columns_without_nan[:2]],
            y,
            test_size=0.25,
            shuffle=True,
            stratify=y,
            random_state=VAL_SEED,
        )
        self._train_idx = X_train_tmp.index
        self._test_idx = X_test_tmp.index
        del y_train
        del y_test
        del X_train_tmp
        del X_test_tmp
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        return

    def save_train_and_test(self, train, test, verbose=True):
        filepath_train = os.path.join(self.output_dir, self.train_name)
        train.to_csv(filepath_train, index=False)
        if verbose:
            print(f"Train enregistré dans {filepath_train}. Forme : {train.shape}")
            print("\nInformations sur le jeu de Train :")
            train.info()

        filepath_test = os.path.join(self.output_dir, self.test_name)
        test.to_csv(filepath_test, index=False)
        if verbose:
            print(f"\nTest enregistré dans {filepath_test}. Forme : {test.shape}")

        """******# On sauvegarde aussi la liste features encodées en OnHot
        joblib.dump(self.features_oh_encoded, os.path.join(self.output_dir, self.oh_columns_name))
        if verbose:
            print("Liste des noms de features encodées en OneHot")"""
        return train, test

    def init_config(self, verbose=True, **kwargs):
        # La liste des paramètres possibles à passer dans kwargs sont
        # les attributs qui ne sont pas protégés ni privés dans l'objet experience ou dans l'objet meta
        valid_args = [arg for arg in kwargs.keys() if arg in self.public_attributes]
        invalid_args = [
            arg for arg in kwargs.keys() if arg not in self.public_attributes
        ]

        # On met à jour les paramètres qui sont autorisés
        for a in valid_args:
            self.__dict__[a] = kwargs[a]

        if valid_args:
            print(f"Paramètres {valid_args} mis à jour")

        if invalid_args:
            print(
                f"Warning : Les paramètres {invalid_args} ne sont pas valides. Liste des paramètres possibles :"
            )
            print(self.public_attributes)

    def print_config(self):
        print("Configuration :")
        for a in self.public_attributes:
            print(f"\t{a} : {self.__dict__[a]}")


def get_memory_consumed(df, verbose=True):
    # Somme de l'utilisation de la mémoire par colonne
    total_memory = df.memory_usage(deep=True).sum()
    memory_usage_mb = round(total_memory / (1024 * 1024))
    if verbose:
        print(f"Taille mémoire du DataFrame : {memory_usage_mb} Mo")
    return memory_usage_mb


def get_available_memory(verbose=True, threshold=0.85):
    total_memory_available_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    if verbose:
        print("RAM totale disponible :", total_memory_available_gb, "Go")
    memory_available_gb = int(total_memory_available_gb * threshold)
    del total_memory_available_gb
    gc.collect()
    if verbose:
        print(f"RAM disponible au seuil de {threshold:.0%} : {memory_available_gb} Go")
    return memory_available_gb


# On ne pensait pas au départ que nous devions utiliser application_test,
# or cela est utile pour l'API, et surtout pour le data drift.
# il est trop tard pour le prévoir dès la classe DataSimple.
# Heureusement on peut construire ces données a posteriori grâce à la classe DataSimple
def build_X_new_loans(directory="", features=[], dic_dtype=None):
    if not directory:
        directory = MODEL_DIR
    # Par défaut nous voulons les features de notre meilleur modèle
    if not features:
        features_path = os.path.join(directory, "features_rfecv_lightgbm")
        features = read_pkl(directory=directory, filename="features_rfecv_lightgbm")

    data_builder = DataSimple()

    # On ne supprime pas les première catégories car celles_ci sont identifiées quand on construit le train.
    data_builder.init_config(debug=False, drop_first=False, na_value=np.nan)
    data_builder.print_config()

    # On agrège les données comme pour le train et le test
    with timer(f"Pipeline d'agrégation des données sur {data_builder._device.upper()}"):
        cu_df_new_loans = data_builder.get_data("application_test.csv")

    print("\nRenommage des colonnes et Traitement des valeurs inf")
    # On fait subir les mêmes traitements basiques que les données du train
    cu_df_new_loans = cu_df_new_loans.rename(
        columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x)
    )
    cu_df_new_loans = data_builder.inf_to_nan(cu_df_new_loans)

    # On supprime les features en trop (Etapes drop_first, variance nulle, VIF, RFECV)
    features_to_drop = [
        f
        for f in cu_df_new_loans.columns
        if f not in features + ["TARGET", "SK_ID_CURR"]
    ]
    cu_df_new_loans = cu_df_new_loans.drop(columns=features_to_drop)
    cu_df_new_loans = data_builder.cast_and_optimize(cu_df_new_loans)

    print(f"\n{len(features_to_drop)} features inutiles supprimées")

    # Il pourrait manquer des features si dans application_test, il n'y avait pas toutes les catégories
    # encodées en OHE pendant la construction du train.
    # Pour notre meilleur modèle, il ne manque rien, mais on ne sait jamais.
    missing_features = [f for f in features if f not in cu_df_new_loans.columns]
    if missing_features:
        print(
            f"{len(missing_features)} features n'ont pas été créées dans le nouveau jeu de données"
        )
        print(
            "On les crée avec la valeur 0 (= l'enregistrement n'est pas de cette catégorie"
        )
        cu_df_new_loans[missing_features] = 0
    else:
        print(
            "Toutes les catégories des features encodées en OHE sont présentes dans le nouveau jeu de données"
        )

    # On place "SK_ID_CURR" en index et on le dit
    cu_df_new_loans = cu_df_new_loans.set_index("SK_ID_CURR")
    print("Indexation par 'SK_ID_CURR'")

    # On transforme en pandas DataFrame
    df_new_loans = cu_df_new_loans.to_pandas()

    # Si un dictionnaire de types pour les features, on caste toutes les features avec ce dictionnaire
    # Les clefs du dictionnaire doivent correspondre aux features présentes
    if dic_dtype is not None:
        # On met un warning si les clefs du dictionnaire ne correspondent pas aux features
        features_in_dic_not_in_features = [
            f for f in dic_dtype.keys() if f not in features
        ]
        features_in_features_not_in_dic = [
            f for f in features if f not in dic_dtype.keys()
        ]
        if features_in_dic_not_in_features:
            print(
                f"WARNING : {len(features_in_dic_not_in_features)} features du dictionnaire dic_dtype ne figurent pas dans la liste des features"
            )
            print(features_in_dic_not_in_features)
        if features_in_features_not_in_dic:
            print(
                f"WARNING : {len(features_in_features_not_in_dic)} features n'ont pas de type dans le dictionnaire dic_dtype"
            )
            print(features_in_features_not_in_dic)

        # On effectue le cast selon le dictionnaire
        df_new_loans = df_new_loans.astype(dic_dtype)

    # On imprime les infos()
    print("\nInfo new loans :")
    print(df_new_loans.info())
    return df_new_loans
