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
import lightgbm as lgb
import re
import gc
import os
import time
import psutil
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from src.p7_file import make_dir
from src.p7_regex import sel_var

warnings.simplefilter(action="ignore", category=FutureWarning)

from src.p7_constantes import (
    DATA_CLEAN_DIR,
    NUM_THREADS,
    DATA_BASE,
    DATA_INTERIM,
    MODEL_DIR,
    RANDOM_SEED,
    # GENERATE_SUBMISSION_FILES,
    # STRATIFIED_KFOLD,
    # NUM_FOLDS,
    # EARLY_STOPPING,
)

from src.p7_util import timer


SUBMISSION_SUFIX = "_simple_debug"

CONFIG_SIMPLE = {
    "debug": False,
    "nan_as_cat": True,
    "data_output_dir": DATA_INTERIM,
    "data_filename": "all_data_simple_kernel_ohe.csv",
    "generate_submission_files": True,
    "model_dir": MODEL_DIR,
    "model_subdir": "light_simple/",
    "importance_filename": "feature_importance.csv",  # Average over NUM_FOLDS
    "submission_filename": "lightgbm_simple_submission.csv",
    "num_threads": NUM_THREADS,
    "stratified_kfold": True,
    "num_folds": 10,
    "early_stopping": 100,
    "random_seed": 1001,
}

# LightGBM parameters found by Bayesian optimization for kernel simple_features
LIGHTGBM_PARAMS_SIMPLE = {
    "n_estimators": 10000,
    "learning_rate": 0.02,
    "num_leaves": 34,
    "colsample_bytree": 0.9497036,
    "subsample": 0.8715623,
    "max_depth": 8,
    "reg_alpha": 0.041545473,
    "reg_lambda": 0.0735294,
    "min_split_gain": 0.0222415,
    "min_child_weight": 39.3259775,
    "silent": -1,
    "verbose": -1,
}


def make_data_dir(config=CONFIG_SIMPLE):
    make_dir(directories_to_make=[config["data_output_dir"]])
    return


def make_model_dir(config=CONFIG_SIMPLE):
    to_make = [
        config["model_dir"],
        os.path.join(config["model_dir"], config["model_subdir"]),
    ]
    make_dir(to_make)
    return


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv
def application(num_rows=None, nan_as_category=True):
    # Read data, on ne lit pas application_test. Nous recrérons un jeu de test avec target
    df = pd.read_csv(
        os.path.join(DATA_BASE, "application_train.csv"),
        nrows=num_rows,
        na_values=["XNA", "Unknown"],
        keep_default_na=True,
    )
    print("Data samples: {}".format(len(df)))

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    # Les valeurs manquantes sont toutes des Non defaut
    # df = df[df["CODE_GENDER"] != "XNA"]

    # Categorical features with Binary encode (0 or 1; two categories)
    # FLAG_OW_CAR et FLAG_OWN_REALTY n'ont aucune valeur manquante
    # Nous n'aurons donc pas de valeur négative avec pd.factorize
    for bin_feature in [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "EMERGENCYSTATE_MODE",
    ]:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # CODE_GENDER et EMERGENCYSTATE_MODE ont des NaN
    df["CODE_GENDER"].replace(-1, np.nan, inplace=True)
    df["EMERGENCYSTATE_MODE"].replace(-1, np.nan, inplace=True)

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365_243 -> nan (équivaut à 1_000 ans)
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    # NaN values for DAYS_LAST_PHONE_CHANGE: 0 -> nan
    df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan, inplace=True)

    # Some simple new features (percentages)
    # [TODO] Renommer ces RATIOS calculés
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv(os.path.join(DATA_BASE, "bureau.csv"), nrows=num_rows)
    bb = pd.read_csv(os.path.join(DATA_BASE, "bureau_balance.csv"), nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for col in bb_cat:
        bb_aggregations[col] = ["mean"]
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
    bb_agg.columns = pd.Index(
        [e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()]
    )
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
    bureau.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

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
    bureau_agg.columns = pd.Index(
        ["BURO_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()]
    )
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
    active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
    active_agg.columns = pd.Index(
        ["ACTIVE_" + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()]
    )
    bureau_agg = bureau_agg.join(active_agg, how="left", on="SK_ID_CURR")
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]
    closed_agg = closed.groupby("SK_ID_CURR").agg(num_aggregations)
    closed_agg.columns = pd.Index(
        ["CLOSED_" + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()]
    )
    bureau_agg = bureau_agg.join(closed_agg, how="left", on="SK_ID_CURR")
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv(
        os.path.join(DATA_BASE, "previous_application.csv"), nrows=num_rows
    )
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev["DAYS_FIRST_DRAWING"].replace(365243, np.nan, inplace=True)
    prev["DAYS_FIRST_DUE"].replace(365243, np.nan, inplace=True)
    prev["DAYS_LAST_DUE_1ST_VERSION"].replace(365243, np.nan, inplace=True)
    prev["DAYS_LAST_DUE"].replace(365243, np.nan, inplace=True)
    prev["DAYS_TERMINATION"].replace(365243, np.nan, inplace=True)
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

    prev_agg = prev.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(
        ["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()]
    )
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
    approved_agg = approved.groupby("SK_ID_CURR").agg(num_aggregations)
    approved_agg.columns = pd.Index(
        ["APPROVED_" + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()]
    )
    prev_agg = prev_agg.join(approved_agg, how="left", on="SK_ID_CURR")
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1]
    refused_agg = refused.groupby("SK_ID_CURR").agg(num_aggregations)
    refused_agg.columns = pd.Index(
        ["REFUSED_" + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()]
    )
    prev_agg = prev_agg.join(refused_agg, how="left", on="SK_ID_CURR")
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv(os.path.join(DATA_BASE, "POS_CASH_balance.csv"), nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    for cat in cat_cols:
        aggregations[cat] = ["mean"]

    pos_agg = pos.groupby("SK_ID_CURR").agg(aggregations)
    pos_agg.columns = pd.Index(
        ["POS_" + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()]
    )
    # Count pos cash accounts
    pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv(
        os.path.join(DATA_BASE, "installments_payments.csv"), nrows=num_rows
    )
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    # Days past due and days before due (no negative values)
    ins["DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DBD"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]
    ins["DPD"] = ins["DPD"].apply(lambda x: x if x > 0 else 0)
    ins["DBD"] = ins["DBD"].apply(lambda x: x if x > 0 else 0)
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
    ins_agg.columns = pd.Index(
        ["INSTAL_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()]
    )
    # Count installments accounts
    ins_agg["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv(os.path.join(DATA_BASE, "credit_card_balance.csv"), nrows=num_rows)

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
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=nan_as_category)
    cc_agg_cat = (
        cc[["SK_ID_CURR"] + cat_cols].groupby("SK_ID_CURR").agg(["mean", "sum", "var"])
    )
    cc_agg = cc_agg_num.merge(cc_agg_cat, how="left", on="SK_ID_CURR")
    cc_agg.columns = pd.Index(
        ["CC_" + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()]
    )

    del cc_agg_num
    del cc_agg_cat
    gc.collect()

    # Count credit card lines
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    del cc
    gc.collect()

    # [DEBUG]
    """object_features = cc_agg.select_dtypes(include="object").columns
    print()
    print("credit_card_balance - object_features", object_features)
    print()"""
    return cc_agg


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm_simple(df=None, config=CONFIG_SIMPLE):
    make_model_dir(config)
    if df is None:
        data_filepath = os.path.join(config["data_output_dir"], config["data_filename"])
        df = pd.read_csv(data_filepath)
    """if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)"""
    # Ajout pour error : [LightGBM] [Fatal] Do not support special JSON characters in feature name.
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

    # Divide in training/validation and test data
    train_df = df[df["TARGET"].notnull()]
    # test_df = df[df["TARGET"].isnull()]
    if config["debug"]:
        train_df = train_df.head(10_000)
        # test_df = test_df.head(1000)
    print("Starting LightGBM. Train shape: {}".format(train_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if config["stratified_kfold"]:
        folds = StratifiedKFold(
            n_splits=config["num_folds"],
            shuffle=True,
            random_state=config["random_seed"],
        )
    else:
        folds = KFold(
            n_splits=config["num_folds"],
            shuffle=True,
            random_state=config["random_seed"],
        )
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    # sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [
        f
        for f in train_df.columns
        if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]

    for n_fold, (train_idx, valid_idx) in enumerate(
        folds.split(train_df[feats], train_df["TARGET"])
    ):
        train_x, train_y = (
            train_df[feats].iloc[train_idx],
            train_df["TARGET"].iloc[train_idx],
        )
        valid_x, valid_y = (
            train_df[feats].iloc[valid_idx],
            train_df["TARGET"].iloc[valid_idx],
        )

        # LightGBM parameters found by Bayesian optimization
        params = {
            "random_state": config["random_seed"],
            "nthread": config["num_threads"],
        }
        clf = LGBMClassifier(**{**params, **LIGHTGBM_PARAMS_SIMPLE})

        clf.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric="auc",
            # verbose=200,
            callbacks=[lgb.early_stopping(stopping_rounds=config["early_stopping"])],
        )

        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_
        )[:, 1]
        """sub_preds += (
            clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1]
            / folds.n_splits
        )"""

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0
        )
        print(
            "Fold %2d AUC : %.6f"
            % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]))
        )
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
    print("Full AUC score %.6f" % roc_auc_score(train_df["TARGET"], oof_preds))

    # Compute average and std importance importance
    mean_importance = (
        feature_importance_df.drop("fold", axis=1)
        .groupby("feature")
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
        )
        .sort_values(by="importance_mean", ascending=False)
    )
    # write importance
    path_model = os.path.join(config["model_dir"], config["model_subdir"])
    importance_filepath = os.path.join(path_model, config["importance_filename"])
    mean_importance.to_csv(importance_filepath, index=True)
    print("Importance saved in", importance_filepath)

    # Write submission file
    if config["generate_submission_files"]:
        submission_filepath = os.path.join(path_model, config["submission_filename"])
        # test_df["TARGET"] = sub_preds
        # test_df[["SK_ID_CURR", "TARGET"]].to_csv(submission_filepath, index=False)
        print("Submission saved in", submission_filepath)
    # display_importances(feature_importance_df)
    return mean_importance


# Display/plot feature importance
def display_importances(
    mean_importance, top=40, sort_by_name=False, save_img=True, config=CONFIG_SIMPLE
):
    best_importance = mean_importance.head(top)
    if sort_by_name:
        best_importance = best_importance.sort_values(by="feature")

    # Taille de la figure
    width = 8
    margin = 1
    bar_height = 0.3
    height = margin + bar_height * top

    # print(best_importance)
    fig = plt.figure(figsize=(width, height))
    barplot = sns.barplot(
        x="importance_mean",
        y="feature",
        data=best_importance,
    )

    # Récupérer les coordonnées des barres du barplot
    x_positions = [bar.get_width() for bar in barplot.patches]
    y_positions = [bar.get_y() + bar.get_height() / 2 for bar in barplot.patches]

    # Ajouter les barres d'erreur
    plt.errorbar(
        x=x_positions,
        y=y_positions,
        # data=best_importance,
        xerr=[best_importance["importance_std"], best_importance["importance_std"]],
        fmt="none",  # Aucun marqueur pour les points de données
        # capsize=5,  # Taille des barres à l'extrémité des lignes d'erreur
        color="black",  # Couleur des barres d'erreur
    )

    fig.suptitle(f"Top {top} Most important LightGBM Features (avg over folds)\n")
    plt.tight_layout()

    if save_img:
        path_model = os.path.join(config["model_dir"], config["model_subdir"])
        figname = os.path.splitext(config["importance_filename"])[0] + ".png"
        importance_figpath = os.path.join(path_model, figname)
        print("Importance saved in", importance_figpath)
        plt.savefig(importance_figpath)


def get_simple_data(config=CONFIG_SIMPLE):
    make_data_dir(config)
    num_rows = 10000 if config["debug"] else None
    nan_as_category = config["nan_as_cat"]

    with timer("Process Application"):
        df = application(num_rows, nan_as_category)
        print("Application df shape:", df.shape)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows, nan_as_category)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how="left", on="SK_ID_CURR")
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows, nan_as_category)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how="left", on="SK_ID_CURR")
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows, nan_as_category)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how="left", on="SK_ID_CURR")
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows, nan_as_category)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how="left", on="SK_ID_CURR")
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows, nan_as_category)
        print("Credit card balance df shape:", cc.shape)

        # df = df.join(cc, how="left", on="SK_ID_CURR")
        df = df.merge(cc, how="left", on="SK_ID_CURR")
        del cc
        gc.collect()

        to_drop = sel_var(df.columns, "Unnamed", verbose=False)
        if to_drop:
            df = df.drop(to_drop, axis=1)
        df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    with timer("Remplacement des valeurs infinies par NaN"):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # write data
    print("write data")
    data_filepath = os.path.join(config["data_output_dir"], config["data_filename"])
    # df.to_csv(data_filepath, index=False)
    # print("Data saved in", data_filepath)
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]
    del df
    gc.collect()

    # Pour la compétition, le jeu de test initial ne comporte pas de Target.
    # Nous ne pouvons donc pas l'utiliser.
    # Nous nous réservons donc nouveau un jeu de test parmi le jeu de train initial.
    with timer("Partage Train Test(25%)"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
        )
        del X, y
        gc.collect()

        test = pd.concat([X_test, y_test], axis=1)
        del X_test, y_test
        gc.collect()
        test_path = os.path.join(config["data_output_dir"], "v0_test.csv")
        test.to_csv(test_path, index=False)
        print(f"\nTaille du jeu de test : {test.shape}, sauvegardé dans {test_path}")

        train = pd.concat([X_train, y_train], axis=1)
        del X_train, y_train
        gc.collect()
        train_path = os.path.join(config["data_output_dir"], "v0_train.csv")
        train.to_csv(train_path, index=False)
        print(
            f"Taille du jeu de Train : {train.shape}, {get_memory_consumed(train, verbose=False)} Mo. Sauvegardé dans {train_path}"
        )
    return train, test


def add_ext_source_1_known(df):
    df["EXT_SOURCE_1_KNOWN"] = df["EXT_SOURCE_1"].apply(
        lambda x: False if np.isnan(x) else True
    )
    return df


def reduce_memory(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != bool:
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df.loc[:, col] = df.loc[:, col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df.loc[:, col] = df.loc[:, col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df.loc[:, col] = df.loc[:, col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df.loc[:, col] = df.loc[:, col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df.loc[:, col] = df.loc[:, col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df.loc[:, col] = df.loc[:, col].astype(np.float32)
                    else:
                        df.loc[:, col] = df.loc[:, col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


"""def get_memory_usage(df, unit="Mo", verbose=True):
    memory_usage = df.memory_usage(deep=True)
    # Somme de l'utilisation de la mémoire par colonne
    total_memory = memory_usage.sum()

    if unit.lower() == "ko" or unit.lower() == "kb":
        memory_usage = total_memory / 1024
    elif unit.lower() == "mo" or unit.lower() == "mb":
        memory_usage = total_memory / (1024 * 1024)
    elif unit.lower() == "go" or unit.lower() == "gb":
        memory_usage = total_memory / (1024 * 1024 * 1024)
    else:
        # print("L'unité doit être Ko, Mo ou Go, l'unité fournie est", unit)
        print("L'unité doit être Ko Kb, Mo, Mb ou Go Gb, got", unit)
        return
    if verbose:
        print("Taille mémoire du DataFrame :", memory_usage, unit)
    return memory_usage, unit"""


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


"""def get_batch_size(df, verbose=True):
    n_intial_rows = df.shape[0]
    memory_available_gb = get_available_memory(verbose=verbose)
    memory_consumed_gb = get_memory_consumed(df=df, verbose=verbose)
    del df
    gc.collect()

    batch_size = int(memory_available_gb * n_intial_rows / memory_consumed_gb)
    if verbose:
        print(f"Taille de batch conseillée : {batch_size}")
    return batch_size"""


"""def lgbm_feature_importance_simple(config=CONFIG_SIMPLE):
    with timer("Run LightGBM with kfold"):
        num_rows = 10000 if config["debug"] else None
        df = pd.read_csv(config["data_filepath"], nrows=num_rows)

        feat_importance = kfold_lightgbm(df, config)
    return feat_importance"""


"""def main(debug=False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how="left", on="SK_ID_CURR")
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how="left", on="SK_ID_CURR")
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how="left", on="SK_ID_CURR")
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how="left", on="SK_ID_CURR")
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how="left", on="SK_ID_CURR")
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm_simple(df, config=CONFIG_SIMPLE)
        # feat_importance = kfold_lightgbm_sklearn(df)
        print(feat_importance)


if __name__ == "__main__":
    submission_file_name = "submission_kernel_simple.csv"
    with timer("Full model run"):
        main(debug=False)
        # get_simple_data(debug=True)"""
