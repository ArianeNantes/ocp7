"""
Lien : https://www.kaggle.com/code/jsaguiar/lightgbm-7th-place-solution

KAGGLE HOME CREDIT DEFAULT RISK COMPETITION
Adapted from one of the models used in 7th place solution ensemble.
For more details about our solution please check this discussion:
https://www.kaggle.com/c/home-credit-default-risk/discussion/64580

Another similar version is also available at GitHub:
https://github.com/js-aguiar/home-credit-default-competition

This model uses LightGBM with goss and label encode for the application's 
categorical features. Other tables are using one-hot encode with mean, 
sum and a few different functions to aggregate. The main ideia was to add 
more time related features like last application and last X months aggregations.
There are also aggregations for specific loan types and status as well as
ratios between tables. Configurations are in line 785
"""

import os
import gc
import numpy as np
import pandas as pd
import cudf
import cuml
from cuml.preprocessing import SimpleImputer as CuSimpleImputer
import cupy as cp
import re

import multiprocessing as mp
from functools import partial
from scipy.stats import kurtosis, iqr, skew
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")

from src.p7_constantes import (
    NUM_THREADS,
    DATA_BASE,
    DATA_INTERIM,
    GENERATE_SUBMISSION_FILES,
    STRATIFIED_KFOLD,
    VAL_SEED,
    NUM_FOLDS,
    EARLY_STOPPING,
    MODEL_DIR,
)
from src.p7_util import timer

SUBMISSION_SUFIX = "_full_04"


# ------------------------- CONFIGURATIONS -------------------------

# INSTALLMENTS TREND PERIODS
INSTALLMENTS_LAST_K_TREND_PERIODS = [12, 24, 60, 120]

# AGGREGATIONS
BUREAU_AGG = {
    "SK_ID_BUREAU": ["nunique"],
    "DAYS_CREDIT": ["min", "max", "mean"],
    "DAYS_CREDIT_ENDDATE": ["min", "max"],
    "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
    "AMT_CREDIT_SUM": ["max", "mean", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
    "AMT_CREDIT_SUM_OVERDUE": ["max", "mean", "sum"],
    "AMT_ANNUITY": ["mean"],
    "DEBT_CREDIT_DIFF": ["mean", "sum"],
    "MONTHS_BALANCE_MEAN": ["mean", "var"],
    "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    # Categorical
    "STATUS_0": ["mean"],
    "STATUS_1": ["mean"],
    "STATUS_12345": ["mean"],
    "STATUS_C": ["mean"],
    "STATUS_X": ["mean"],
    "CREDIT_ACTIVE_Active": ["mean"],
    "CREDIT_ACTIVE_Closed": ["mean"],
    "CREDIT_ACTIVE_Sold": ["mean"],
    "CREDIT_TYPE_Consumer credit": ["mean"],
    "CREDIT_TYPE_Credit card": ["mean"],
    "CREDIT_TYPE_Car loan": ["mean"],
    "CREDIT_TYPE_Mortgage": ["mean"],
    "CREDIT_TYPE_Microloan": ["mean"],
    # Group by loan duration features (months)
    "LL_AMT_CREDIT_SUM_OVERDUE": ["mean"],
    "LL_DEBT_CREDIT_DIFF": ["mean"],
    "LL_STATUS_12345": ["mean"],
}

BUREAU_ACTIVE_AGG = {
    "DAYS_CREDIT": ["max", "mean"],
    "DAYS_CREDIT_ENDDATE": ["min", "max"],
    "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
    "AMT_CREDIT_SUM": ["max", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
    "AMT_CREDIT_SUM_OVERDUE": ["max", "mean"],
    "DAYS_CREDIT_UPDATE": ["min", "mean"],
    "DEBT_PERCENTAGE": ["mean"],
    "DEBT_CREDIT_DIFF": ["mean"],
    "CREDIT_TO_ANNUITY_RATIO": ["mean"],
    "MONTHS_BALANCE_MEAN": ["mean", "var"],
    "MONTHS_BALANCE_SIZE": ["mean", "sum"],
}

BUREAU_CLOSED_AGG = {
    "DAYS_CREDIT": ["max", "var"],
    "DAYS_CREDIT_ENDDATE": ["max"],
    "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
    "AMT_CREDIT_SUM_OVERDUE": ["mean"],
    "AMT_CREDIT_SUM": ["max", "mean", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["max", "sum"],
    "DAYS_CREDIT_UPDATE": ["max"],
    "ENDDATE_DIF": ["mean"],
    "STATUS_12345": ["mean"],
}

BUREAU_LOAN_TYPE_AGG = {
    "DAYS_CREDIT": ["mean", "max"],
    "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
    "AMT_CREDIT_SUM": ["mean", "max"],
    "AMT_CREDIT_SUM_DEBT": ["mean", "max"],
    "DEBT_PERCENTAGE": ["mean"],
    "DEBT_CREDIT_DIFF": ["mean"],
    "DAYS_CREDIT_ENDDATE": ["max"],
}

BUREAU_TIME_AGG = {
    "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
    "AMT_CREDIT_SUM_OVERDUE": ["mean"],
    "AMT_CREDIT_SUM": ["max", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
    "DEBT_PERCENTAGE": ["mean"],
    "DEBT_CREDIT_DIFF": ["mean"],
    "STATUS_0": ["mean"],
    "STATUS_12345": ["mean"],
}

PREVIOUS_AGG = {
    "SK_ID_PREV": ["nunique"],
    "AMT_ANNUITY": ["min", "max", "mean"],
    "AMT_DOWN_PAYMENT": ["max", "mean"],
    "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
    "RATE_DOWN_PAYMENT": ["max", "mean"],
    "DAYS_DECISION": ["min", "max", "mean"],
    "CNT_PAYMENT": ["max", "mean"],
    "DAYS_TERMINATION": ["max"],
    # Engineered features
    "CREDIT_TO_ANNUITY_RATIO": ["mean", "max"],
    "APPLICATION_CREDIT_DIFF": ["min", "max", "mean"],
    "APPLICATION_CREDIT_RATIO": ["min", "max", "mean", "var"],
    "DOWN_PAYMENT_TO_CREDIT": ["mean"],
}

PREVIOUS_ACTIVE_AGG = {
    "SK_ID_PREV": ["nunique"],
    "SIMPLE_INTERESTS": ["mean"],
    "AMT_ANNUITY": ["max", "sum"],
    "AMT_APPLICATION": ["max", "mean"],
    "AMT_CREDIT": ["sum"],
    "AMT_DOWN_PAYMENT": ["max", "mean"],
    "DAYS_DECISION": ["min", "mean"],
    "CNT_PAYMENT": ["mean", "sum"],
    "DAYS_LAST_DUE_1ST_VERSION": ["min", "max", "mean"],
    # Engineered features
    "AMT_PAYMENT": ["sum"],
    "INSTALMENT_PAYMENT_DIFF": ["mean", "max"],
    "REMAINING_DEBT": ["max", "mean", "sum"],
    "REPAYMENT_RATIO": ["mean"],
}

PREVIOUS_APPROVED_AGG = {
    "SK_ID_PREV": ["nunique"],
    "AMT_ANNUITY": ["min", "max", "mean"],
    "AMT_CREDIT": ["min", "max", "mean"],
    "AMT_DOWN_PAYMENT": ["max"],
    "AMT_GOODS_PRICE": ["max"],
    "HOUR_APPR_PROCESS_START": ["min", "max"],
    "DAYS_DECISION": ["min", "mean"],
    "CNT_PAYMENT": ["max", "mean"],
    "DAYS_TERMINATION": ["mean"],
    # Engineered features
    "CREDIT_TO_ANNUITY_RATIO": ["mean", "max"],
    "APPLICATION_CREDIT_DIFF": ["max"],
    "APPLICATION_CREDIT_RATIO": ["min", "max", "mean"],
    # The following features are only for approved applications
    "DAYS_FIRST_DRAWING": ["max", "mean"],
    "DAYS_FIRST_DUE": ["min", "mean"],
    "DAYS_LAST_DUE_1ST_VERSION": ["min", "max", "mean"],
    "DAYS_LAST_DUE": ["max", "mean"],
    "DAYS_LAST_DUE_DIFF": ["min", "max", "mean"],
    "SIMPLE_INTERESTS": ["min", "max", "mean"],
}

PREVIOUS_REFUSED_AGG = {
    "AMT_APPLICATION": ["max", "mean"],
    "AMT_CREDIT": ["min", "max"],
    "DAYS_DECISION": ["min", "max", "mean"],
    "CNT_PAYMENT": ["max", "mean"],
    # Engineered features
    "APPLICATION_CREDIT_DIFF": ["min", "max", "mean", "var"],
    "APPLICATION_CREDIT_RATIO": ["min", "mean"],
    "NAME_CONTRACT_TYPE_Consumer loans": ["mean"],
    "NAME_CONTRACT_TYPE_Cash loans": ["mean"],
    "NAME_CONTRACT_TYPE_Revolving loans": ["mean"],
}

PREVIOUS_LATE_PAYMENTS_AGG = {
    "DAYS_DECISION": ["min", "max", "mean"],
    "DAYS_LAST_DUE_1ST_VERSION": ["min", "max", "mean"],
    # Engineered features
    "APPLICATION_CREDIT_DIFF": ["min"],
    "NAME_CONTRACT_TYPE_Consumer loans": ["mean"],
    "NAME_CONTRACT_TYPE_Cash loans": ["mean"],
    "NAME_CONTRACT_TYPE_Revolving loans": ["mean"],
}

PREVIOUS_LOAN_TYPE_AGG = {
    "AMT_CREDIT": ["sum"],
    "AMT_ANNUITY": ["mean", "max"],
    "SIMPLE_INTERESTS": ["min", "mean", "max", "var"],
    "APPLICATION_CREDIT_DIFF": ["min", "var"],
    "APPLICATION_CREDIT_RATIO": ["min", "max", "mean"],
    "DAYS_DECISION": ["max"],
    "DAYS_LAST_DUE_1ST_VERSION": ["max", "mean"],
    "CNT_PAYMENT": ["mean"],
}

PREVIOUS_TIME_AGG = {
    "AMT_CREDIT": ["sum"],
    "AMT_ANNUITY": ["mean", "max"],
    "SIMPLE_INTERESTS": ["mean", "max"],
    "DAYS_DECISION": ["min", "mean"],
    "DAYS_LAST_DUE_1ST_VERSION": ["min", "max", "mean"],
    # Engineered features
    "APPLICATION_CREDIT_DIFF": ["min"],
    "APPLICATION_CREDIT_RATIO": ["min", "max", "mean"],
    "NAME_CONTRACT_TYPE_Consumer loans": ["mean"],
    "NAME_CONTRACT_TYPE_Cash loans": ["mean"],
    "NAME_CONTRACT_TYPE_Revolving loans": ["mean"],
}

POS_CASH_AGG = {
    "SK_ID_PREV": ["nunique"],
    "MONTHS_BALANCE": ["min", "max", "size"],
    "SK_DPD": ["max", "mean", "sum", "var"],
    "SK_DPD_DEF": ["max", "mean", "sum"],
    "LATE_PAYMENT": ["mean"],
}

INSTALLMENTS_AGG = {
    "SK_ID_PREV": ["size", "nunique"],
    "DAYS_ENTRY_PAYMENT": ["min", "max", "mean"],
    "AMT_INSTALMENT": ["min", "max", "mean", "sum"],
    "AMT_PAYMENT": ["min", "max", "mean", "sum"],
    "DPD": ["max", "mean", "var"],
    "DBD": ["max", "mean", "var"],
    "PAYMENT_DIFFERENCE": ["mean"],
    "PAYMENT_RATIO": ["mean"],
    "LATE_PAYMENT": ["mean", "sum"],
    "SIGNIFICANT_LATE_PAYMENT": ["mean", "sum"],
    "LATE_PAYMENT_RATIO": ["mean"],
    "DPD_7": ["mean"],
    "DPD_15": ["mean"],
    "PAID_OVER": ["mean"],
}

INSTALLMENTS_TIME_AGG = {
    "SK_ID_PREV": ["size"],
    "DAYS_ENTRY_PAYMENT": ["min", "max", "mean"],
    "AMT_INSTALMENT": ["min", "max", "mean", "sum"],
    "AMT_PAYMENT": ["min", "max", "mean", "sum"],
    "DPD": ["max", "mean", "var"],
    "DBD": ["max", "mean", "var"],
    "PAYMENT_DIFFERENCE": ["mean"],
    "PAYMENT_RATIO": ["mean"],
    "LATE_PAYMENT": ["mean"],
    "SIGNIFICANT_LATE_PAYMENT": ["mean"],
    "LATE_PAYMENT_RATIO": ["mean"],
    "DPD_7": ["mean"],
    "DPD_15": ["mean"],
}

CREDIT_CARD_AGG = {
    "MONTHS_BALANCE": ["min"],
    "AMT_BALANCE": ["max"],
    "AMT_CREDIT_LIMIT_ACTUAL": ["max"],
    "AMT_DRAWINGS_ATM_CURRENT": ["max", "sum"],
    "AMT_DRAWINGS_CURRENT": ["max", "sum"],
    "AMT_DRAWINGS_POS_CURRENT": ["max", "sum"],
    "AMT_INST_MIN_REGULARITY": ["max", "mean"],
    "AMT_PAYMENT_TOTAL_CURRENT": ["max", "mean", "sum", "var"],
    "AMT_TOTAL_RECEIVABLE": ["max", "mean"],
    "CNT_DRAWINGS_ATM_CURRENT": ["max", "mean", "sum"],
    "CNT_DRAWINGS_CURRENT": ["max", "mean", "sum"],
    "CNT_DRAWINGS_POS_CURRENT": ["mean"],
    "SK_DPD": ["mean", "max", "sum"],
    "SK_DPD_DEF": ["max", "sum"],
    "LIMIT_USE": ["max", "mean"],
    "PAYMENT_DIV_MIN": ["min", "mean"],
    "LATE_PAYMENT": ["max", "sum"],
}

CREDIT_CARD_TIME_AGG = {
    "CNT_DRAWINGS_ATM_CURRENT": ["mean"],
    "SK_DPD": ["max", "sum"],
    "AMT_BALANCE": ["mean", "max"],
    "LIMIT_USE": ["max", "mean"],
}

LIGHTGBM_PARAMS_FULL = {
    "boosting_type": "goss",
    "n_estimators": 10000,
    "learning_rate": 0.005134,
    "num_leaves": 54,
    "max_depth": 10,
    # Number of samples for constructing bins
    "subsample_for_bin": 240000,
    "reg_alpha": 0.436193,
    "reg_lambda": 0.479169,
    "colsample_bytree": 0.508716,
    "min_split_gain": 0.024766,
    "subsample": 1,
    # The 'balanced' mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
    # as n_samples / (n_classes * np.bincount(y)).
    # If None, all classes are supposed to have weight one.
    # Note, that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
    "is_unbalance": False,
    "silent": -1,
    "verbose": -1,
}

"""if __name__ == "__main__":
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.max_columns", 100)
    with timer("Pipeline total time"):
        main(debug=True)"""


class DataFull:
    def __init__(self, dataset_num="00", debug=False, add_trends=False) -> None:
        self.dataset_num = dataset_num
        self.debug = debug
        self.n_rows = 30_000 if self.debug else None
        self.add_trends = add_trends
        self.input_dir = DATA_BASE
        self.output_dir = DATA_INTERIM
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # Features à encoder en label encoding
        self.features_label_encoding = []
        # Features encodées en One_Hot
        self.features_oh_encoded = []
        # liste des features provenant de la premère modalité des variables encodées OH
        self.first_categories = []
        # Si kernel_useless est vrai on supprime les features jugées inutiles dans le kernel initial
        self.kernel_useless = False

    def get_data(self):
        if self.debug:
            print(f"DEBUG num_rows={self.n_rows}")
        with timer("Application Data"):
            df = self.get_application()
            print("Application dataframe shape: ", df.shape)
        with timer("Bureau and bureau_balance data"):
            bureau_df = self.get_bureau()
            df = cudf.merge(df, bureau_df, on="SK_ID_CURR", how="left")
            print("Bureau dataframe shape: ", bureau_df.shape)
            del bureau_df
            gc.collect()
        with timer("previous_application"):
            prev_df = self.get_previous_applications()
            df = cudf.merge(df, prev_df, on="SK_ID_CURR", how="left")
            print("Previous dataframe shape: ", prev_df.shape)
            del prev_df
            gc.collect()
        with timer("Previous applications balances"):
            pos = self.get_pos_cash()
            df = cudf.merge(df, pos, on="SK_ID_CURR", how="left")
            print("Pos-cash dataframe shape: ", pos.shape)
            del pos
            gc.collect()
        with timer("Installments payments"):
            ins = self.get_installment_payments()
            df = cudf.merge(df, ins, on="SK_ID_CURR", how="left")
            print("Installments dataframe shape: ", ins.shape)
            del ins
            gc.collect()
        with timer("Credit card"):
            cc = self.get_credit_card()
            df = cudf.merge(df, cc, on="SK_ID_CURR", how="left")
            print("Credit card dataframe shape: ", cc.shape)
            del cc
            gc.collect()
        with timer("Add ratios"):
            # Add ratios and groupby between different tables
            df = add_ratios_features(df)
        print("All data shape :", df.shape)

        print("\nTraitement des valeurs inf")
        df = self.inf_to_nan(df)
        print("Suppression des colonnes de variance nulle")
        df = self.del_null_std(df)
        print("Suppression d'une modalité pour les colonnes encodées en One Hot")
        df = self.drop_first_categories(df)

        to_drop = ["SK_ID_PREV", "SK_ID_BUREAU", "index"]
        for f in to_drop:
            if f in df.columns:
                df.drop(f, axis=1, inplace=True)

        df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
        print("all_data.info :")
        print(df.info())
        with timer("Sauvegarde"):
            self.save_all_data(df)
        return df

    # ------------------------- APPLICATION PIPELINE -------------------------

    def get_application(self):
        """Process application_train.csv and application_test.csv and return a pandas dataframe."""
        df = cudf.read_csv(
            os.path.join(self.input_dir, "application_train.csv"), nrows=self.n_rows
        )
        # Data cleaning
        df = df[df["CODE_GENDER"] != "XNA"]  # 4 people with XNA code gender
        df = df[
            df["AMT_INCOME_TOTAL"] < 20000000
        ]  # Max income in test is 4M; train has a 117M value
        # Contraitrement à Pandas, l'introduction de cudf.NA dans une colonne ne modifie pas le type de la colonne en object
        # De même, l'introduire np.nan ne modifie pas le type de la colonne en float64, il faut donc caster avant
        """df[["DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"]] = df[
            ["DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"]
        ].astype("float64")"""
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, cudf.NA)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(0, cudf.NA)
        df["DAYS_LAST_PHONE_CHANGE"] = df["DAYS_LAST_PHONE_CHANGE"].replace(0, cudf.NA)

        # Flag_document features - count and kurtosis
        docs = [f for f in df.columns if "FLAG_DOC" in f]
        df["DOCUMENT_COUNT"] = df[docs].sum(axis=1)
        # df["NEW_DOC_KURT"] = df[docs].kurtosis(axis=1)

        # Categorical age - based on target=1 plot
        # df["AGE_RANGE"] = df["DAYS_BIRTH"].apply(lambda x: get_age_label(x))

        # Les cuDF ne supportent pas l'opération .apply, à la place on utilise des opérations vectorisées
        age_years = -df["DAYS_BIRTH"] / 365
        df["AGE_RANGE"] = (
            (age_years < 27).astype(int) * 1
            + ((age_years >= 27) & (age_years < 40)).astype(int) * 2
            + ((age_years >= 40) & (age_years < 50)).astype(int) * 3
            + ((age_years >= 50) & (age_years < 65)).astype(int) * 4
            + ((age_years >= 65) & (age_years < 99)).astype(int) * 5
            + (age_years >= 99).astype(int) * 0
        )

        # New features based on External sources
        df["EXT_SOURCES_PROD"] = (
            df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        )
        df["EXT_SOURCES_WEIGHTED"] = (
            df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
        )
        # np.seterr(all="ignore", invalid="ignore")

        # nanmedian non implémentée en cudf
        # for function_name in ["min", "max", "mean", "nanmedian", "var"]
        for function_name in ["min", "max", "mean", "var"]:
            feature_name = "EXT_SOURCES_{}".format(function_name.upper())
            df[feature_name] = eval("np.{}".format(function_name))(
                df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]], axis=1
            )

        # Credit ratios
        df["CREDIT_TO_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
        df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
        # Income ratios
        df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        df["INCOME_TO_EMPLOYED_RATIO"] = df["AMT_INCOME_TOTAL"] / df["DAYS_EMPLOYED"]
        df["INCOME_TO_BIRTH_RATIO"] = df["AMT_INCOME_TOTAL"] / df["DAYS_BIRTH"]
        # Time ratios
        df["EMPLOYED_TO_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
        df["ID_TO_BIRTH_RATIO"] = df["DAYS_ID_PUBLISH"] / df["DAYS_BIRTH"]
        df["CAR_TO_BIRTH_RATIO"] = df["OWN_CAR_AGE"] / df["DAYS_BIRTH"]
        df["CAR_TO_EMPLOYED_RATIO"] = df["OWN_CAR_AGE"] / df["DAYS_EMPLOYED"]
        df["PHONE_TO_BIRTH_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]

        # Groupby: Statistics for applications in the same group
        group = [
            "ORGANIZATION_TYPE",
            "NAME_EDUCATION_TYPE",
            "OCCUPATION_TYPE",
            "AGE_RANGE",
            "CODE_GENDER",
        ]
        df = do_median(df, group, "EXT_SOURCES_MEAN", "GROUP_EXT_SOURCES_MEDIAN")
        df = do_std(df, group, "EXT_SOURCES_MEAN", "GROUP_EXT_SOURCES_STD")
        df = do_mean(df, group, "AMT_INCOME_TOTAL", "GROUP_INCOME_MEAN")
        df = do_std(df, group, "AMT_INCOME_TOTAL", "GROUP_INCOME_STD")
        df = do_mean(
            df, group, "CREDIT_TO_ANNUITY_RATIO", "GROUP_CREDIT_TO_ANNUITY_MEAN"
        )
        df = do_std(df, group, "CREDIT_TO_ANNUITY_RATIO", "GROUP_CREDIT_TO_ANNUITY_STD")
        df = do_mean(df, group, "AMT_CREDIT", "GROUP_CREDIT_MEAN")
        df = do_mean(df, group, "AMT_ANNUITY", "GROUP_ANNUITY_MEAN")
        df = do_std(df, group, "AMT_ANNUITY", "GROUP_ANNUITY_STD")

        """# Encode categorical features (LabelEncoder)
        df, le_encoded_cols = label_encoder(df, None)"""
        if self.kernel_useless:
            # On drop avant d'encoder
            df = drop_application_columns(df)

        categorical_columns = df.select_dtypes(include="object").columns.tolist()
        # On encode les features spécifiées en label_encoding et toutes les autres en One Hot
        to_oh_encode = categorical_columns
        if self.features_label_encoding:
            df, le_encoded_cols = label_encoder(
                df, categorical_columns=self.features_label_encoding
            )
            to_oh_encode = [
                col
                for col in categorical_columns
                if col not in self.features_label_encoding
            ]
        df, _ = self.one_hot_encoder(df, categorical_columns=to_oh_encode)

        # 'AMT_ANNUITY' existe dans plusieurs tables, on la renomme en préfixant par le début du nom de la table
        df.rename(
            columns={
                "AMT_ANNUITY": "APP_AMT_ANNUITY",
                "CREDIT_TO_ANNUITY_RATIO": "APP_CREDIT_TO_ANNUITY_RATIO",
            },
            inplace=True,
        )
        return df

    # ------------------------- BUREAU PIPELINE -------------------------

    def get_bureau(self):
        """Process bureau.csv and bureau_balance.csv and return a pandas dataframe."""
        bureau = cudf.read_csv(
            os.path.join(self.input_dir, "bureau.csv"), nrows=self.n_rows
        )
        # Credit duration and credit/account end date difference
        bureau["CREDIT_DURATION"] = (
            -bureau["DAYS_CREDIT"] + bureau["DAYS_CREDIT_ENDDATE"]
        )
        bureau["ENDDATE_DIF"] = (
            bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_ENDDATE_FACT"]
        )
        # Credit to debt ratio and difference
        bureau["DEBT_PERCENTAGE"] = (
            bureau["AMT_CREDIT_SUM"] / bureau["AMT_CREDIT_SUM_DEBT"]
        )
        bureau["DEBT_CREDIT_DIFF"] = (
            bureau["AMT_CREDIT_SUM"] - bureau["AMT_CREDIT_SUM_DEBT"]
        )
        bureau["CREDIT_TO_ANNUITY_RATIO"] = (
            bureau["AMT_CREDIT_SUM"] / bureau["AMT_ANNUITY"]
        )

        # One-hot encoder
        bureau, _ = self.one_hot_encoder(bureau, nan_as_category=False)
        # Join bureau balance features
        bureau = bureau.merge(self.get_bureau_balance(), how="left", on="SK_ID_BUREAU")
        # Flag months with late payments (days past due)
        bureau["STATUS_12345"] = 0
        for i in range(1, 6):
            bureau["STATUS_12345"] += bureau["STATUS_{}".format(i)]

        # Aggregate by number of months in balance and merge with bureau (loan length agg)
        features = [
            "AMT_CREDIT_MAX_OVERDUE",
            "AMT_CREDIT_SUM_OVERDUE",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "DEBT_PERCENTAGE",
            "DEBT_CREDIT_DIFF",
            "STATUS_0",
            "STATUS_12345",
        ]
        agg_length = (
            bureau.groupby("MONTHS_BALANCE_SIZE")[features].mean().reset_index()
        )
        agg_length.rename(
            {feat: "LL_" + feat for feat in features}, axis=1, inplace=True
        )
        bureau = bureau.merge(agg_length, how="left", on="MONTHS_BALANCE_SIZE")
        del agg_length
        gc.collect()

        # General loans aggregations
        agg_bureau = group(bureau, "BUREAU_", BUREAU_AGG)
        # Active and closed loans aggregations
        active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
        agg_bureau = group_and_merge(
            active, agg_bureau, "BUREAU_ACTIVE_", BUREAU_ACTIVE_AGG
        )
        closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]
        agg_bureau = group_and_merge(
            closed, agg_bureau, "BUREAU_CLOSED_", BUREAU_CLOSED_AGG
        )
        del active, closed
        gc.collect()
        # Aggregations for the main loan types
        for credit_type in [
            "Consumer credit",
            "Credit card",
            "Mortgage",
            "Car loan",
            "Microloan",
        ]:
            type_df = bureau[bureau["CREDIT_TYPE_" + credit_type] == 1]
            prefix = "BUREAU_" + credit_type.split(" ")[0].upper() + "_"
            agg_bureau = group_and_merge(
                type_df, agg_bureau, prefix, BUREAU_LOAN_TYPE_AGG
            )
            del type_df
            gc.collect()
        # Time based aggregations: last x months
        for time_frame in [6, 12]:
            prefix = "BUREAU_LAST{}M_".format(time_frame)
            time_frame_df = bureau[bureau["DAYS_CREDIT"] >= -30 * time_frame]
            agg_bureau = group_and_merge(
                time_frame_df, agg_bureau, prefix, BUREAU_TIME_AGG
            )
            del time_frame_df
            gc.collect()

        # Last loan max overdue
        sort_bureau = bureau.sort_values(by=["DAYS_CREDIT"])

        # cudf ne support pas last, on utilise la méthode nth qui permet de retourner la nième ligne de chaque groupe
        gr = sort_bureau.groupby("SK_ID_CURR").nth(-1).reset_index()
        # gr = (
        #    sort_bureau.groupby("SK_ID_CURR")["AMT_CREDIT_MAX_OVERDUE"].last().reset_index()
        # )
        gr = gr.rename(
            columns={"AMT_CREDIT_MAX_OVERDUE": "BUREAU_LAST_LOAN_MAX_OVERDUE"}
        )

        agg_bureau = agg_bureau.merge(gr, on="SK_ID_CURR", how="left")
        # agg_bureau = cudf.merge(agg_bureau, gr, on="SK_ID_CURR", how="left")
        # Ratios: total debt/total credit and active loans debt/ active loans credit
        agg_bureau["BUREAU_DEBT_OVER_CREDIT"] = (
            agg_bureau["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"]
            / agg_bureau["BUREAU_AMT_CREDIT_SUM_SUM"]
        )
        agg_bureau["BUREAU_ACTIVE_DEBT_OVER_CREDIT"] = (
            agg_bureau["BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]
            / agg_bureau["BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM"]
        )
        # 'AMT_ANNUITY' existe dans plusieurs tables, on la renomme en préfixant par le début du nom de la table
        if "AMT_ANNUITY" in agg_bureau:
            agg_bureau.rename(columns={"AMT_ANNUITY": "BUR_AMT_ANNUITY"}, inplace=True)
        if "CREDIT_TO_ANNUITY_RATIO" in agg_bureau:
            agg_bureau.rename(
                columns={"CREDIT_TO_ANNUITY_RATIO": "BUR_CREDIT_TO_ANNUITY_RATIO"},
                inplace=True,
            )
        return agg_bureau

    def get_bureau_balance(self):
        bb = cudf.read_csv(
            os.path.join(self.input_dir, "bureau_balance.csv"), nrows=self.n_rows
        )
        bb, categorical_cols = self.one_hot_encoder(bb, nan_as_category=False)
        # Calculate rate for each category with decay
        bb_processed = bb.groupby("SK_ID_BUREAU")[categorical_cols].mean().reset_index()
        # Min, Max, Count and mean duration of payments (months)
        agg = {"MONTHS_BALANCE": ["min", "max", "mean", "size"]}
        bb_processed = group_and_merge(bb, bb_processed, "", agg, "SK_ID_BUREAU")
        del bb
        gc.collect()
        return bb_processed

    # ------------------------- PREVIOUS PIPELINE -------------------------

    def get_previous_applications(self):
        """Process previous_application.csv and return a pandas dataframe."""
        prev = cudf.read_csv(
            os.path.join(self.input_dir, "previous_application.csv"), nrows=self.n_rows
        )
        pay = cudf.read_csv(
            os.path.join(self.input_dir, "installments_payments.csv"), nrows=self.n_rows
        )

        # One-hot encode most important categorical features
        ohe_columns = [
            "NAME_CONTRACT_STATUS",
            "NAME_CONTRACT_TYPE",
            "CHANNEL_TYPE",
            "NAME_TYPE_SUITE",
            "NAME_YIELD_GROUP",
            "PRODUCT_COMBINATION",
            "NAME_PRODUCT_TYPE",
            "NAME_CLIENT_TYPE",
        ]
        prev, categorical_cols = self.one_hot_encoder(
            prev, ohe_columns, nan_as_category=False
        )

        # Feature engineering: ratios and difference
        prev["APPLICATION_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
        prev["APPLICATION_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
        prev["CREDIT_TO_ANNUITY_RATIO"] = prev["AMT_CREDIT"] / prev["AMT_ANNUITY"]
        prev["DOWN_PAYMENT_TO_CREDIT"] = prev["AMT_DOWN_PAYMENT"] / prev["AMT_CREDIT"]
        # Interest ratio on previous application (simplified)
        total_payment = prev["AMT_ANNUITY"] * prev["CNT_PAYMENT"]
        prev["SIMPLE_INTERESTS"] = (total_payment / prev["AMT_CREDIT"] - 1) / prev[
            "CNT_PAYMENT"
        ]

        # Active loans - approved and not complete yet (last_due 365243)
        approved = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
        active_df = approved[approved["DAYS_LAST_DUE"] == 365243]
        # Find how much was already payed in active loans (using installments csv)
        active_pay = pay[pay["SK_ID_PREV"].isin(active_df["SK_ID_PREV"])]
        active_pay_agg = active_pay.groupby("SK_ID_PREV")[
            ["AMT_INSTALMENT", "AMT_PAYMENT"]
        ].sum()
        active_pay_agg.reset_index(inplace=True)
        # Active loans: difference of what was payed and installments
        active_pay_agg["INSTALMENT_PAYMENT_DIFF"] = (
            active_pay_agg["AMT_INSTALMENT"] - active_pay_agg["AMT_PAYMENT"]
        )
        # Merge with active_df
        active_df = active_df.merge(active_pay_agg, on="SK_ID_PREV", how="left")
        active_df["REMAINING_DEBT"] = active_df["AMT_CREDIT"] - active_df["AMT_PAYMENT"]
        active_df["REPAYMENT_RATIO"] = (
            active_df["AMT_PAYMENT"] / active_df["AMT_CREDIT"]
        )
        # Perform aggregations for active applications
        active_agg_df = group(active_df, "PREV_ACTIVE_", PREVIOUS_ACTIVE_AGG)
        active_agg_df["TOTAL_REPAYMENT_RATIO"] = (
            active_agg_df["PREV_ACTIVE_AMT_PAYMENT_SUM"]
            / active_agg_df["PREV_ACTIVE_AMT_CREDIT_SUM"]
        )
        del active_pay, active_pay_agg, active_df
        gc.collect()

        # Change 365.243 values to nan (missing)
        prev["DAYS_FIRST_DRAWING"].replace(365243, cudf.NA, inplace=True)
        prev["DAYS_FIRST_DUE"].replace(365243, cudf.NA, inplace=True)
        prev["DAYS_LAST_DUE_1ST_VERSION"].replace(365243, cudf.NA, inplace=True)
        prev["DAYS_LAST_DUE"].replace(365243, cudf.NA, inplace=True)
        prev["DAYS_TERMINATION"].replace(365243, cudf.NA, inplace=True)
        # Days last due difference (scheduled x done)
        prev["DAYS_LAST_DUE_DIFF"] = (
            prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_LAST_DUE"]
        )
        approved["DAYS_LAST_DUE_DIFF"] = (
            approved["DAYS_LAST_DUE_1ST_VERSION"] - approved["DAYS_LAST_DUE"]
        )

        # Categorical features
        categorical_agg = {key: ["mean"] for key in categorical_cols}
        # Perform general aggregations
        agg_prev = group(prev, "PREV_", {**PREVIOUS_AGG, **categorical_agg})
        # Merge active loans dataframe on agg_prev
        agg_prev = agg_prev.merge(active_agg_df, how="left", on="SK_ID_CURR")
        del active_agg_df
        gc.collect()
        # Aggregations for approved and refused loans
        agg_prev = group_and_merge(
            approved, agg_prev, "APPROVED_", PREVIOUS_APPROVED_AGG
        )
        refused = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1]
        agg_prev = group_and_merge(refused, agg_prev, "REFUSED_", PREVIOUS_REFUSED_AGG)
        del approved, refused
        gc.collect()
        # Aggregations for Consumer loans and Cash loans
        for loan_type in ["Consumer loans", "Cash loans"]:
            type_df = prev[prev["NAME_CONTRACT_TYPE_{}".format(loan_type)] == 1]
            prefix = "PREV_" + loan_type.split(" ")[0] + "_"
            agg_prev = group_and_merge(
                type_df, agg_prev, prefix, PREVIOUS_LOAN_TYPE_AGG
            )
            del type_df
            gc.collect()

        # Get the SK_ID_PREV for loans with late payments (days past due)
        pay["LATE_PAYMENT"] = pay["DAYS_ENTRY_PAYMENT"] - pay["DAYS_INSTALMENT"]

        pay["LATE_PAYMENT"] = (pay["LATE_PAYMENT"] > 0).astype(int) * 1 + (
            pay["LATE_PAYMENT"] < 0
        ).astype(int) * 0

        # pay["LATE_PAYMENT"] = pay["LATE_PAYMENT"].apply(lambda x: 1 if x > 0 else 0)
        dpd_id = pay[pay["LATE_PAYMENT"] > 0]["SK_ID_PREV"].unique()
        # Aggregations for loans with late payments
        agg_dpd = group_and_merge(
            prev[prev["SK_ID_PREV"].isin(dpd_id)],
            agg_prev,
            "PREV_LATE_",
            PREVIOUS_LATE_PAYMENTS_AGG,
        )
        del agg_dpd, dpd_id
        gc.collect()
        # Aggregations for loans in the last x months
        for time_frame in [12, 24]:
            time_frame_df = prev[prev["DAYS_DECISION"] >= -30 * time_frame]
            prefix = "PREV_LAST{}M_".format(time_frame)
            agg_prev = group_and_merge(
                time_frame_df, agg_prev, prefix, PREVIOUS_TIME_AGG
            )
            del time_frame_df
            gc.collect()
        del prev
        gc.collect()
        # 'AMT_ANNUITY' existe dans plusieurs tables, on la renomme en préfixant par le début du nom de la table
        if "AMT_ANNUITY" in agg_prev:
            agg_prev.rename(columns={"AMT_ANNUITY": "PREV_AMT_ANNUITY"}, inplace=True)
        if "CREDIT_TO_ANNUITY_RATIO" in agg_prev:
            agg_prev.rename(
                columns={"CREDIT_TO_ANNUITY_RATIO": "PREV_CREDIT_TO_ANNUITY_RATIO"},
                inplace=True,
            )
        return agg_prev

    # ------------------------- POS-CASH PIPELINE -------------------------
    """def get_pos_cash(self):
        #Process POS_CASH_balance.csv and return a pandas dataframe.
        pos = cudf.read_csv(
            os.path.join(self.input_dir, "POS_CASH_balance.csv"), nrows=self.n_rows
        )
        pos, categorical_cols = self.one_hot_encoder(pos, nan_as_category=False)
        # Flag months with late payment
        pos["LATE_PAYMENT"] = (pos["SK_DPD"] > 0).astype(int) * 1 + (
            pos["SK_DPD"] < 0
        ).astype(int) * 0
        # pos["LATE_PAYMENT"] = pos["SK_DPD"].apply(lambda x: 1 if x > 0 else 0)

        # Aggregate by SK_ID_CURR
        categorical_agg = {key: ["mean"] for key in categorical_cols}
        pos_agg = group(pos, "POS_", {**POS_CASH_AGG, **categorical_agg})
        # Sort and group by SK_ID_PREV
        sort_pos = pos.sort_values(by=["SK_ID_PREV", "MONTHS_BALANCE"])
        gp = sort_pos.groupby("SK_ID_PREV")

        df = pd.DataFrame()
        df["SK_ID_CURR"] = gp["SK_ID_CURR"].first()

        df["MONTHS_BALANCE_MAX"] = gp["MONTHS_BALANCE"].max()
        # Percentage of previous loans completed and completed before initial term
        df["POS_LOAN_COMPLETED_MEAN"] = gp["NAME_CONTRACT_STATUS_Completed"].mean()
        df["POS_COMPLETED_BEFORE_MEAN"] = (
            gp["CNT_INSTALMENT"].first() - gp["CNT_INSTALMENT"].last()
        )
        df["POS_COMPLETED_BEFORE_MEAN"] = df.apply(
            lambda x: (
                1
                if x["POS_COMPLETED_BEFORE_MEAN"] > 0
                and x["POS_LOAN_COMPLETED_MEAN"] > 0
                else 0
            ),
            axis=1,
        )
        # Number of remaining installments (future installments) and percentage from total
        df["POS_REMAINING_INSTALMENTS"] = gp["CNT_INSTALMENT_FUTURE"].last()
        df["POS_REMAINING_INSTALMENTS_RATIO"] = (
            gp["CNT_INSTALMENT_FUTURE"].last() / gp["CNT_INSTALMENT"].last()
        )
        # Group by SK_ID_CURR and merge
        df_gp = df.groupby("SK_ID_CURR").sum().reset_index()
        df_gp.drop(["MONTHS_BALANCE_MAX"], axis=1, inplace=True)
        pos_agg = pd.merge(pos_agg, df_gp, on="SK_ID_CURR", how="left")
        del df, gp, df_gp, sort_pos
        gc.collect()

        # Percentage of late payments for the 3 most recent applications
        pos = do_sum(pos, ["SK_ID_PREV"], "LATE_PAYMENT", "LATE_PAYMENT_SUM")
        # Last month of each application
        last_month_df = pos.groupby("SK_ID_PREV")["MONTHS_BALANCE"].idxmax()
        # Most recent applications (last 3)
        sort_pos = pos.sort_values(by=["SK_ID_PREV", "MONTHS_BALANCE"])
        gp = sort_pos.iloc[last_month_df].groupby("SK_ID_CURR").tail(3)
        gp_mean = gp.groupby("SK_ID_CURR").mean().reset_index()
        pos_agg = pd.merge(
            pos_agg,
            gp_mean[["SK_ID_CURR", "LATE_PAYMENT_SUM"]],
            on="SK_ID_CURR",
            how="left",
        )

        # print("Colonnes de pos_agg", pos_agg.columns)

        # kernel_useless spécifie s'il faut supprimer les features jugées inutiles dans le kernel d'origine
        if self.kernel_useless:
            drop_features = [
                "POS_NAME_CONTRACT_STATUS_Canceled_MEAN",
                "POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN",
                "POS_NAME_CONTRACT_STATUS_XNA_MEAN",
            ]
            # pos_agg.drop(drop_features, axis=1, inplace=True)
            # Pour pouvoir l'exécuter en mode debug, on check si les colonnes sont présentes
            # car elles sont issues d'un onehot
            drop_features = [
                feature for feature in drop_features if feature in pos_agg.columns
            ]
            if drop_features:
                pos_agg.drop(drop_features, axis=1, inplace=True)

        return cudf.from_pandas(pos_agg)"""

    # [TODO] si le temps : passer en cuDF
    def get_pos_cash(self):
        # Process POS_CASH_balance.csv and return a pandas dataframe.
        pos = pd.read_csv(
            os.path.join(self.input_dir, "POS_CASH_balance.csv"), nrows=self.n_rows
        )
        pos, categorical_cols = self.one_hot_encoder(pos, nan_as_category=False)
        # Flag months with late payment
        pos["LATE_PAYMENT"] = pos["SK_DPD"].apply(lambda x: 1 if x > 0 else 0)

        # pos["LATE_PAYMENT"] = (pos["LATE_PAYMENT"] > 0).astype(int) * 1 + (
        #    pos["LATE_PAYMENT"] < 0
        # ).astype(int) * 0

        # Aggregate by SK_ID_CURR
        categorical_agg = {key: ["mean"] for key in categorical_cols}
        pos_agg = group(pos, "POS_", {**POS_CASH_AGG, **categorical_agg})
        # Sort and group by SK_ID_PREV
        sort_pos = pos.sort_values(by=["SK_ID_PREV", "MONTHS_BALANCE"])
        gp = sort_pos.groupby("SK_ID_PREV")
        df = pd.DataFrame()

        df["SK_ID_CURR"] = gp["SK_ID_CURR"].first()

        df["MONTHS_BALANCE_MAX"] = gp["MONTHS_BALANCE"].max()
        # Percentage of previous loans completed and completed before initial term
        df["POS_LOAN_COMPLETED_MEAN"] = gp["NAME_CONTRACT_STATUS_Completed"].mean()
        df["POS_COMPLETED_BEFORE_MEAN"] = (
            gp["CNT_INSTALMENT"].first() - gp["CNT_INSTALMENT"].last()
        )
        df["POS_COMPLETED_BEFORE_MEAN"] = df.apply(
            lambda x: (
                1
                if x["POS_COMPLETED_BEFORE_MEAN"] > 0
                and x["POS_LOAN_COMPLETED_MEAN"] > 0
                else 0
            ),
            axis=1,
        )
        # Number of remaining installments (future installments) and percentage from total
        df["POS_REMAINING_INSTALMENTS"] = gp["CNT_INSTALMENT_FUTURE"].last()
        df["POS_REMAINING_INSTALMENTS_RATIO"] = (
            gp["CNT_INSTALMENT_FUTURE"].last() / gp["CNT_INSTALMENT"].last()
        )
        # Group by SK_ID_CURR and merge
        df_gp = df.groupby("SK_ID_CURR").sum().reset_index()
        df_gp.drop(["MONTHS_BALANCE_MAX"], axis=1, inplace=True)
        pos_agg = pd.merge(pos_agg, df_gp, on="SK_ID_CURR", how="left")
        del df, gp, df_gp, sort_pos
        gc.collect()

        # Percentage of late payments for the 3 most recent applications
        pos = do_sum(pos, ["SK_ID_PREV"], "LATE_PAYMENT", "LATE_PAYMENT_SUM")
        # Last month of each application
        last_month_df = pos.groupby("SK_ID_PREV")["MONTHS_BALANCE"].idxmax()
        # Most recent applications (last 3)
        sort_pos = pos.sort_values(by=["SK_ID_PREV", "MONTHS_BALANCE"])
        gp = sort_pos.iloc[last_month_df].groupby("SK_ID_CURR").tail(3)
        gp_mean = gp.groupby("SK_ID_CURR").mean().reset_index()
        pos_agg = pd.merge(
            pos_agg,
            gp_mean[["SK_ID_CURR", "LATE_PAYMENT_SUM"]],
            on="SK_ID_CURR",
            how="left",
        )

        # print("Colonnes de pos_agg", pos_agg.columns)

        # kernel_useless spécifie s'il faut supprimer les features jugées inutiles dans le kernel d'origine
        if self.kernel_useless:
            drop_features = [
                "POS_NAME_CONTRACT_STATUS_Canceled_MEAN",
                "POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN",
                "POS_NAME_CONTRACT_STATUS_XNA_MEAN",
            ]
            # pos_agg.drop(drop_features, axis=1, inplace=True)
            # Pour pouvoir l'exécuter en mode debug, on check si les colonnes sont présentes
            # car elles sont issues d'un onehot
            drop_features = [
                feature for feature in drop_features if feature in pos_agg.columns
            ]
            if drop_features:
                pos_agg.drop(drop_features, axis=1, inplace=True)

        return cudf.from_pandas(pos_agg)

    # ------------------------- INSTALLMENTS PIPELINE -------------------------

    def get_installment_payments(self):
        """Process installments_payments.csv and return a pandas dataframe."""
        pay = pd.read_csv(
            os.path.join(self.input_dir, "installments_payments.csv"), nrows=self.n_rows
        )
        # Group payments and get Payment difference
        pay = do_sum(
            pay,
            ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"],
            "AMT_PAYMENT",
            "AMT_PAYMENT_GROUPED",
        )
        pay["PAYMENT_DIFFERENCE"] = pay["AMT_INSTALMENT"] - pay["AMT_PAYMENT_GROUPED"]
        pay["PAYMENT_RATIO"] = pay["AMT_INSTALMENT"] / pay["AMT_PAYMENT_GROUPED"]
        pay["PAID_OVER_AMOUNT"] = pay["AMT_PAYMENT"] - pay["AMT_INSTALMENT"]
        pay["PAID_OVER"] = (pay["PAID_OVER_AMOUNT"] > 0).astype(int)
        # Payment Entry: Days past due and Days before due
        pay["DPD"] = pay["DAYS_ENTRY_PAYMENT"] - pay["DAYS_INSTALMENT"]
        pay["DPD"] = pay["DPD"].apply(lambda x: 0 if x <= 0 else x)
        pay["DBD"] = pay["DAYS_INSTALMENT"] - pay["DAYS_ENTRY_PAYMENT"]
        pay["DBD"] = pay["DBD"].apply(lambda x: 0 if x <= 0 else x)
        # Flag late payment
        pay["LATE_PAYMENT"] = pay["DBD"].apply(lambda x: 1 if x > 0 else 0)
        # Percentage of payments that were late
        pay["INSTALMENT_PAYMENT_RATIO"] = pay["AMT_PAYMENT"] / pay["AMT_INSTALMENT"]
        pay["LATE_PAYMENT_RATIO"] = pay.apply(
            lambda x: x["INSTALMENT_PAYMENT_RATIO"] if x["LATE_PAYMENT"] == 1 else 0,
            axis=1,
        )
        # Flag late payments that have a significant amount
        pay["SIGNIFICANT_LATE_PAYMENT"] = pay["LATE_PAYMENT_RATIO"].apply(
            lambda x: 1 if x > 0.05 else 0
        )
        # Flag k threshold late payments
        pay["DPD_7"] = pay["DPD"].apply(lambda x: 1 if x >= 7 else 0)
        pay["DPD_15"] = pay["DPD"].apply(lambda x: 1 if x >= 15 else 0)
        # Aggregations by SK_ID_CURR
        pay_agg = group(pay, "INS_", INSTALLMENTS_AGG)

        # Installments in the last x months
        for months in [36, 60]:
            recent_prev_id = pay[pay["DAYS_INSTALMENT"] >= -30 * months][
                "SK_ID_PREV"
            ].unique()
            pay_recent = pay[pay["SK_ID_PREV"].isin(recent_prev_id)]
            prefix = "INS_{}M_".format(months)
            pay_agg = group_and_merge(
                pay_recent, pay_agg, prefix, INSTALLMENTS_TIME_AGG
            )

        # Le calcul des tendances est long (nécessite une régression par individu),
        # On ne les calcule que si demandé
        if self.add_trends:
            # Last x periods trend features
            group_features = [
                "SK_ID_CURR",
                "SK_ID_PREV",
                "DPD",
                "LATE_PAYMENT",
                "PAID_OVER_AMOUNT",
                "PAID_OVER",
                "DAYS_INSTALMENT",
            ]
            gp = pay[group_features].groupby("SK_ID_CURR")
            func = partial(
                trend_in_last_k_instalment_features,
                periods=INSTALLMENTS_LAST_K_TREND_PERIODS,
            )
            g = parallel_apply(
                gp, func, index_name="SK_ID_CURR", chunk_size=10000
            ).reset_index()
            pay_agg = pay_agg.merge(g, on="SK_ID_CURR", how="left")

            # Last loan features
            g = parallel_apply(
                gp,
                installments_last_loan_features,
                index_name="SK_ID_CURR",
                chunk_size=10000,
            ).reset_index()
            pay_agg = pay_agg.merge(g, on="SK_ID_CURR", how="left")

        return cudf.from_pandas(pay_agg)

    # ------------------------- CREDIT CARD PIPELINE -------------------------

    def get_credit_card(self):
        """Process credit_card_balance.csv and return a pandas dataframe."""
        cc = cudf.read_csv(
            os.path.join(self.input_dir, "credit_card_balance.csv"), nrows=self.n_rows
        )
        cc, _ = self.one_hot_encoder(cc, nan_as_category=False)
        cc.rename(columns={"AMT_RECIVABLE": "AMT_RECEIVABLE"}, inplace=True)
        # Amount used from limit
        cc["LIMIT_USE"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]
        # Current payment / Min payment
        cc["PAYMENT_DIV_MIN"] = (
            cc["AMT_PAYMENT_CURRENT"] / cc["AMT_INST_MIN_REGULARITY"]
        )
        # Late payment
        cc["LATE_PAYMENT"] = (cc["SK_DPD"] > 0).astype(int) * 1 + (
            cc["SK_DPD"] < 0
        ).astype(int) * 0
        # cc["LATE_PAYMENT"] = cc["SK_DPD"].apply(lambda x: 1 if x > 0 else 0)

        # How much drawing of limit
        cc["DRAWING_LIMIT_RATIO"] = (
            cc["AMT_DRAWINGS_ATM_CURRENT"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]
        )
        # Aggregations by SK_ID_CURR
        cc_agg = cc.groupby("SK_ID_CURR").agg(CREDIT_CARD_AGG)
        cc_agg.columns = cudf.Index(
            ["CC_" + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()]
        )
        cc_agg.reset_index(inplace=True)

        # Last month balance of each credit card application
        last_ids = cc.groupby("SK_ID_PREV")["MONTHS_BALANCE"].idxmax()
        last_months_df = cc[cc.index.isin(last_ids)]
        cc_agg = group_and_merge(
            last_months_df, cc_agg, "CC_LAST_", {"AMT_BALANCE": ["mean", "max"]}
        )

        # Aggregations for last x months
        for months in [12, 24, 48]:
            cc_prev_id = cc[cc["MONTHS_BALANCE"] >= -months]["SK_ID_PREV"].unique()
            cc_recent = cc[cc["SK_ID_PREV"].isin(cc_prev_id)]
            prefix = "INS_{}M_".format(months)
            cc_agg = group_and_merge(cc_recent, cc_agg, prefix, CREDIT_CARD_TIME_AGG)
        return cc_agg

    def inf_to_nan_old(self, df):
        if not isinstance(df, (pd.DataFrame, pd.Series)):
            # On convertit en Pandas
            # Attention np.nan est un float64 (pd.NA tansformerait la colonne en dtype object)
            X = df.to_pandas()
            X = X.replace([np.inf, -np.inf], np.nan)
            X = cudf.from_pandas(X)

        else:
            X = X.replace([np.inf, -np.inf], np.nan)
        # Pour les cuDF on peut directement travailler les inf, mais à condition qu'il n'y ait pas d'inf de type int
        # Contrairement à pandas, cudf.NA ne modifie pas le type de la colonne
        # Provoque malheureusement une erreur : Potentially unsafe cast for non-equivalent float32 to int64
        # X = df.replace([float("inf"), -float("inf")], np.nan)
        return X

    def inf_to_nan(self, df, verbose=True):
        if verbose:
            # On identifie les colonnes comportant des NaN (parmi les float)
            # Il n'y a des inf que dans les float64
            infinite_columns = []
            for col in df.select_dtypes("float").columns:
                if (df[col] == np.inf).any() or (df[col] == -np.inf).any():
                    infinite_columns.append(col)
            print(f"{len(infinite_columns)} features comportent des valeurs infinies :")
            print(infinite_columns)
        df.select_dtypes("float").replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    def del_null_std(self, df, verbose=True):
        to_drop = []
        # Avec un cuDF, nous devons différencier les types

        # On calcule les variances null sur les types number :
        std = df.select_dtypes(include=["number"]).std()
        # Colonnes de type number avec un écart type de 0.0 (variance nulle)
        to_drop = to_drop + std[std == 0.0].index.to_numpy().tolist()

        # On identifie les colonnes de variance nulle pour les types bool ou object
        # On ne considère pas la présence de types unshable comme les listes etc.
        for col in df.select_dtypes(exclude="number").columns:
            if df[col].nunique() == 1:
                to_drop.append(col)

        # On supprime les colonnes avec un écart type de 0.0
        X = df.drop(to_drop, axis=1)

        if verbose:
            print(f"{len(to_drop)} features de variance nulle")
        return X

    def drop_first_categories(self, df, verbose=True):
        to_drop = [
            feature for feature in self.first_categories if feature in df.columns
        ]
        X = df.drop(to_drop, axis=1)
        if verbose:
            print(
                f"{len(to_drop)} features issues des premières modalités des variables qualitatives encodées en One Hot"
            )
        return X

    def one_hot_encoder(
        self,
        df,
        categorical_columns=None,
        nan_as_category=True,
    ):
        """Crée une nouvelle colonne pour chaque modalité des variables qualitatives,
        Stocke le nom des colonnes encodées et le nom des colonnes correspondant à leur première modalité

        Args:
            df (pd.DataFrame ou cudf.DataFrame): DataFrame contenant des variables à encoder
            categorical_columns (list ou index de colonne, optional): variables à encoder. Si None, toutes les variables de type 'object'. Defaults to None.
            nan_as_category (bool, optional): Si vrai crèe une colonne pour les valeurs manquantes. Defaults to True.

        Returns:
            (DF ou cuDF, list): DataFrame avec les variables encodées et la liste des noms de colonnes encodées
        """
        original_columns = list(df.columns)
        # Si categorical_columns n'est pas spécifié on encode toutes les variables de type object
        if not categorical_columns:
            categorical_columns = [
                col for col in df.columns if df[col].dtype == "object"
            ]
        if categorical_columns:
            # On encode et on stocke la première modalité des variables qualitatives.
            # Nous ne pouvons pas la suprimer avant de faire les aggrégations mais
            # il sera impératif de la supprimer pour les modèles linéaires.
            first_categories = []
            if isinstance(df, (pd.DataFrame, pd.Series)):
                first_categories = [
                    f"{col}_{np.unique(df[col].dropna())[0]}"
                    for col in categorical_columns
                ]

                df = pd.get_dummies(
                    df, columns=categorical_columns, dummy_na=nan_as_category
                )

            # Si df n'est pas pandas, on considère qu'il s'agit d'un cudf
            else:
                first_categories = [
                    f"{col}_{np.unique(df[col]).to_numpy()[0]}"
                    for col in categorical_columns
                ]
                df = cudf.get_dummies(
                    df, columns=categorical_columns, dummy_na=nan_as_category
                )

            self.first_categories = self.first_categories + first_categories
            # On stocke le nom des variables créées par l'encodage
            encoded_columns = [col for col in df.columns if col not in original_columns]
            self.features_oh_encoded = self.features_oh_encoded + encoded_columns

        return df, encoded_columns

    def save_all_data(self, df, verbose=True):
        filename = f"{self.dataset_num}_v0_full_data.csv"
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        if verbose:
            print(f"Données enregistrées dans {filepath}")


def add_ratios_features(df):
    # CREDIT TO INCOME RATIO
    df["BUREAU_INCOME_CREDIT_RATIO"] = (
        df["BUREAU_AMT_CREDIT_SUM_MEAN"] / df["AMT_INCOME_TOTAL"]
    )
    df["BUREAU_ACTIVE_CREDIT_TO_INCOME_RATIO"] = (
        df["BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM"] / df["AMT_INCOME_TOTAL"]
    )
    # PREVIOUS TO CURRENT CREDIT RATIO
    df["CURRENT_TO_APPROVED_CREDIT_MIN_RATIO"] = (
        df["APPROVED_AMT_CREDIT_MIN"] / df["AMT_CREDIT"]
    )
    df["CURRENT_TO_APPROVED_CREDIT_MAX_RATIO"] = (
        df["APPROVED_AMT_CREDIT_MAX"] / df["AMT_CREDIT"]
    )
    df["CURRENT_TO_APPROVED_CREDIT_MEAN_RATIO"] = (
        df["APPROVED_AMT_CREDIT_MEAN"] / df["AMT_CREDIT"]
    )
    # PREVIOUS TO CURRENT ANNUITY RATIO
    df["CURRENT_TO_APPROVED_ANNUITY_MAX_RATIO"] = (
        df["APPROVED_AMT_ANNUITY_MAX"] / df["APP_AMT_ANNUITY"]
    )
    df["CURRENT_TO_APPROVED_ANNUITY_MEAN_RATIO"] = (
        df["APPROVED_AMT_ANNUITY_MEAN"] / df["APP_AMT_ANNUITY"]
    )
    df["PAYMENT_MIN_TO_ANNUITY_RATIO"] = (
        df["INS_AMT_PAYMENT_MIN"] / df["APP_AMT_ANNUITY"]
    )
    df["PAYMENT_MAX_TO_ANNUITY_RATIO"] = (
        df["INS_AMT_PAYMENT_MAX"] / df["APP_AMT_ANNUITY"]
    )
    df["PAYMENT_MEAN_TO_ANNUITY_RATIO"] = (
        df["INS_AMT_PAYMENT_MEAN"] / df["APP_AMT_ANNUITY"]
    )
    # PREVIOUS TO CURRENT CREDIT TO ANNUITY RATIO
    df["CTA_CREDIT_TO_ANNUITY_MAX_RATIO"] = (
        df["APPROVED_CREDIT_TO_ANNUITY_RATIO_MAX"] / df["APP_CREDIT_TO_ANNUITY_RATIO"]
    )
    df["CTA_CREDIT_TO_ANNUITY_MEAN_RATIO"] = (
        df["APPROVED_CREDIT_TO_ANNUITY_RATIO_MEAN"] / df["APP_CREDIT_TO_ANNUITY_RATIO"]
    )
    # DAYS DIFFERENCES AND RATIOS
    df["DAYS_DECISION_MEAN_TO_BIRTH"] = (
        df["APPROVED_DAYS_DECISION_MEAN"] / df["DAYS_BIRTH"]
    )
    df["DAYS_CREDIT_MEAN_TO_BIRTH"] = df["BUREAU_DAYS_CREDIT_MEAN"] / df["DAYS_BIRTH"]
    df["DAYS_DECISION_MEAN_TO_EMPLOYED"] = (
        df["APPROVED_DAYS_DECISION_MEAN"] / df["DAYS_EMPLOYED"]
    )
    df["DAYS_CREDIT_MEAN_TO_EMPLOYED"] = (
        df["BUREAU_DAYS_CREDIT_MEAN"] / df["DAYS_EMPLOYED"]
    )
    return df


# ------------------------- LIGHTGBM MODEL -------------------------


def kfold_lightgbm_sklearn(data, categorical_feature=None):
    # Ajout pour error : [LightGBM] [Fatal] Do not support special JSON characters in feature name.
    data = data.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

    df = data[data["TARGET"].notnull()]
    test = data[data["TARGET"].isnull()]
    print("Train/valid shape: {}, test shape: {}".format(df.shape, test.shape))
    del_features = [
        "TARGET",
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "SK_ID_PREV",
        "index",
        "level_0",
    ]
    predictors = list(filter(lambda v: v not in del_features, df.columns))

    if not STRATIFIED_KFOLD:
        folds = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=VAL_SEED)
    else:
        folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=VAL_SEED)

    # Hold oof predictions, test predictions, feature importance and training/valid auc
    oof_preds = np.zeros(df.shape[0])
    sub_preds = np.zeros(test.shape[0])
    importance_df = pd.DataFrame()
    eval_results = dict()

    for n_fold, (train_idx, valid_idx) in enumerate(
        folds.split(df[predictors], df["TARGET"])
    ):
        train_x, train_y = df[predictors].iloc[train_idx], df["TARGET"].iloc[train_idx]
        valid_x, valid_y = df[predictors].iloc[valid_idx], df["TARGET"].iloc[valid_idx]

        params = {"random_state": VAL_SEED, "nthread": NUM_THREADS}
        clf = LGBMClassifier(**{**params, **LIGHTGBM_PARAMS_FULL})

        if not categorical_feature:
            clf.fit(
                train_x,
                train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric="auc",
                # verbose=400,
                callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING)],
            )
        else:
            clf.fit(
                train_x,
                train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric="auc",
                # verbose=400,
                callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING)],
                feature_name=list(df[predictors].columns),
                categorical_feature=categorical_feature,
            )

        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_
        )[:, 1]
        sub_preds += (
            clf.predict_proba(test[predictors], num_iteration=clf.best_iteration_)[:, 1]
            / folds.n_splits
        )

        # Feature importance by GAIN and SPLIT
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = predictors
        fold_importance["gain"] = clf.booster_.feature_importance(
            importance_type="gain"
        )
        fold_importance["split"] = clf.booster_.feature_importance(
            importance_type="split"
        )
        importance_df = pd.concat([importance_df, fold_importance], axis=0)
        eval_results["train_{}".format(n_fold + 1)] = clf.evals_result_["training"][
            "auc"
        ]
        eval_results["valid_{}".format(n_fold + 1)] = clf.evals_result_["valid_1"][
            "auc"
        ]

        print(
            "Fold %2d AUC : %.6f"
            % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]))
        )
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print("Full AUC score %.6f" % roc_auc_score(df["TARGET"], oof_preds))
    # test["TARGET"] = sub_preds.copy()
    test.loc[:, ["TARGET"]] = sub_preds.copy()

    # Get the average feature importance between folds
    mean_importance = importance_df.groupby("feature").mean().reset_index()
    mean_importance.sort_values(by="gain", ascending=False, inplace=True)
    # Save feature importance, test predictions and oof predictions as csv
    if GENERATE_SUBMISSION_FILES:

        # Generate oof csv
        oof = pd.DataFrame()
        # oof["SK_ID_CURR"] = df["SK_ID_CURR"].copy()
        oof.loc[:, ["SK_ID_CURR"]] = df["SK_ID_CURR"].copy()
        # df["PREDICTIONS"] = oof_preds.copy()
        df.loc[:, ["PREDICTIONS"]] = oof_preds.copy()
        # df["TARGET"] = df["TARGET"].copy()
        df.loc[:, ["TARGET"]] = df["TARGET"].copy()
        df.to_csv(
            os.path.join(MODEL_DIR, "oof{}.csv".format(SUBMISSION_SUFIX)), index=False
        )
        # Save submission (test data) and feature importance
        test[["SK_ID_CURR", "TARGET"]].to_csv(
            os.path.join(MODEL_DIR, "submission{}.csv".format(SUBMISSION_SUFIX)),
            index=False,
        )
        mean_importance.to_csv(
            os.path.join(
                MODEL_DIR, "feature_importance{}.csv".format(SUBMISSION_SUFIX)
            ),
            index=False,
        )
    return mean_importance


def drop_application_columns(df):
    """Drop features based on permutation feature importance."""
    drop_list = [
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "HOUR_APPR_PROCESS_START",
        "FLAG_EMP_PHONE",
        "FLAG_MOBIL",
        "FLAG_CONT_MOBILE",
        "FLAG_EMAIL",
        "FLAG_PHONE",
        "FLAG_OWN_REALTY",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_WORK_CITY",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "COMMONAREA_MODE",
        "NONLIVINGAREA_MODE",
        "ELEVATORS_MODE",
        "NONLIVINGAREA_AVG",
        "FLOORSMIN_MEDI",
        "LANDAREA_MODE",
        "NONLIVINGAREA_MEDI",
        "LIVINGAPARTMENTS_MODE",
        "FLOORSMIN_AVG",
        "LANDAREA_AVG",
        "FLOORSMIN_MODE",
        "LANDAREA_MEDI",
        "COMMONAREA_MEDI",
        "YEARS_BUILD_AVG",
        "COMMONAREA_AVG",
        "BASEMENTAREA_AVG",
        "BASEMENTAREA_MODE",
        "NONLIVINGAPARTMENTS_MEDI",
        "BASEMENTAREA_MEDI",
        "LIVINGAPARTMENTS_AVG",
        "ELEVATORS_AVG",
        "YEARS_BUILD_MEDI",
        "ENTRANCES_MODE",
        "NONLIVINGAPARTMENTS_MODE",
        "LIVINGAREA_MODE",
        "LIVINGAPARTMENTS_MEDI",
        "YEARS_BUILD_MODE",
        "YEARS_BEGINEXPLUATATION_AVG",
        "ELEVATORS_MEDI",
        "LIVINGAREA_MEDI",
        "YEARS_BEGINEXPLUATATION_MODE",
        "NONLIVINGAPARTMENTS_AVG",
        "HOUSETYPE_MODE",
        "FONDKAPREMONT_MODE",
        "EMERGENCYSTATE_MODE",
    ]
    # Drop most flag document columns
    for doc_num in [2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21]:
        drop_list.append("FLAG_DOCUMENT_{}".format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df


def get_age_label(days_birth):
    """Return the age group label (int)."""
    age_years = -days_birth / 365
    if age_years < 27:
        return 1
    elif age_years < 40:
        return 2
    elif age_years < 50:
        return 3
    elif age_years < 65:
        return 4
    elif age_years < 99:
        return 5
    else:
        return 0


def trend_in_last_k_instalment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(["DAYS_INSTALMENT"], ascending=False, inplace=True)
    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]
        features = add_trend_feature(
            features, gr_period, "DPD", "{}_TREND_".format(period)
        )
        features = add_trend_feature(
            features, gr_period, "PAID_OVER_AMOUNT", "{}_TREND_".format(period)
        )
    return features


def installments_last_loan_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(["DAYS_INSTALMENT"], ascending=False, inplace=True)
    last_installment_id = gr_["SK_ID_PREV"].iloc[0]
    gr_ = gr_[gr_["SK_ID_PREV"] == last_installment_id]

    features = {}
    features = add_features_in_group(
        features, gr_, "DPD", ["sum", "mean", "max", "std"], "LAST_LOAN_"
    )
    features = add_features_in_group(
        features, gr_, "LATE_PAYMENT", ["count", "mean"], "LAST_LOAN_"
    )
    features = add_features_in_group(
        features,
        gr_,
        "PAID_OVER_AMOUNT",
        ["sum", "mean", "max", "min", "std"],
        "LAST_LOAN_",
    )
    features = add_features_in_group(
        features, gr_, "PAID_OVER", ["count", "mean"], "LAST_LOAN_"
    )
    return features


# ------------------------- UTILITY FUNCTIONS -------------------------


def group(df_to_agg, prefix, aggregations, aggregate_by="SK_ID_CURR"):
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(
        ["{}{}_{}".format(prefix, e[0], e[1].upper()) for e in agg_df.columns.tolist()]
    )
    return agg_df.reset_index()


def group_and_merge(
    df_to_agg, df_to_merge, prefix, aggregations, aggregate_by="SK_ID_CURR"
):
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by=aggregate_by)
    return df_to_merge.merge(agg_df, how="left", on=aggregate_by)


def do_mean(df, group_cols, counted, agg_name):
    gp = (
        df[group_cols + [counted]]
        .groupby(group_cols)[counted]
        .mean()
        .reset_index()
        .rename(columns={counted: agg_name})
    )
    df = df.merge(gp, on=group_cols, how="left")
    del gp
    gc.collect()
    return df


def do_median(df, group_cols, counted, agg_name):
    gp = (
        df[group_cols + [counted]]
        .groupby(group_cols)[counted]
        .median()
        .reset_index()
        .rename(columns={counted: agg_name})
    )
    df = df.merge(gp, on=group_cols, how="left")
    del gp
    gc.collect()
    return df


def do_std(df, group_cols, counted, agg_name):
    gp = (
        df[group_cols + [counted]]
        .groupby(group_cols)[counted]
        .std()
        .reset_index()
        .rename(columns={counted: agg_name})
    )
    df = df.merge(gp, on=group_cols, how="left")
    del gp
    gc.collect()
    return df


def do_sum(df, group_cols, counted, agg_name):
    gp = (
        df[group_cols + [counted]]
        .groupby(group_cols)[counted]
        .sum()
        .reset_index()
        .rename(columns={counted: agg_name})
    )
    df = df.merge(gp, on=group_cols, how="left")
    del gp
    gc.collect()
    return df


def pd_one_hot_encoder(
    df,
    categorical_columns=None,
    nan_as_category=True,
):
    """Create a new column for each categorical value in categorical columns."""
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns


def label_encoder(df, categorical_columns=None):
    """Encode categorical values as integers (0,1,2,3...) with pandas.factorize."""
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])

        # Ajout
        """if np.min(df[col]) == -1:
            print(f"NaN rencontrées dans {col} converties en -1")"""

    return df, categorical_columns


def add_features(feature_name, aggs, features, feature_names, groupby):
    feature_names.extend(["{}_{}".format(feature_name, agg) for agg in aggs])

    for agg in aggs:
        if agg == "kurt":
            agg_func = kurtosis
        elif agg == "iqr":
            agg_func = iqr
        else:
            agg_func = agg

        g = (
            groupby[feature_name]
            .agg(agg_func)
            .reset_index()
            .rename(
                index=str, columns={feature_name: "{}_{}".format(feature_name, agg)}
            )
        )
        features = features.merge(g, on="SK_ID_CURR", how="left")
    return features, feature_names


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == "sum":
            features["{}{}_sum".format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == "mean":
            features["{}{}_mean".format(prefix, feature_name)] = gr_[
                feature_name
            ].mean()
        elif agg == "max":
            features["{}{}_max".format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == "min":
            features["{}{}_min".format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == "std":
            features["{}{}_std".format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == "count":
            features["{}{}_count".format(prefix, feature_name)] = gr_[
                feature_name
            ].count()
        elif agg == "skew":
            features["{}{}_skew".format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == "kurt":
            features["{}{}_kurt".format(prefix, feature_name)] = kurtosis(
                gr_[feature_name]
            )
        elif agg == "iqr":
            features["{}{}_iqr".format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == "median":
            features["{}{}_median".format(prefix, feature_name)] = gr_[
                feature_name
            ].median()
    return features


def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        # Attention à ne pas assigner pd.NA car cela créerait des types de données mixtes
        # Tandis que np.nan est un float.
        trend = np.nan
    features["{}{}".format(prefix, feature_name)] = trend
    return features


def parallel_apply(groups, func, index_name="Index", num_workers=0, chunk_size=100000):
    if num_workers <= 0:
        num_workers = NUM_THREADS
    # n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in chunk_groups(groups, chunk_size):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)
        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_


# Display/plot feature importance
def display_full_importances(feature_importance_df_, importance="gain", top=40):
    cols = (
        feature_importance_df_[["feature", importance]]
        .groupby("feature")
        .mean()
        .sort_values(by=importance, ascending=False)[:top]
        .index
    )
    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)
    ]
    plt.figure(figsize=(8, 10))
    sns.barplot(
        x=importance,
        y="feature",
        data=best_features.sort_values(by=importance, ascending=False),
    )
    plt.title(
        f"Top {top} Importance ({importance} - avg over folds) of LightGBM Features"
    )
    plt.tight_layout()
    plt.savefig("lgbm_importances01.png")
