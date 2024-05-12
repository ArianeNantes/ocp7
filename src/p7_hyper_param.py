import numpy as np
import logging
import re
import os
import gc
import time
import joblib
import lightgbm as lgb
import optuna
import plotly
import kaleido
import mlflow

import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
)

from src.p7_constantes import DATA_INTERIM, DATA_BASE, MODEL_DIR
from src.p7_constantes import NUM_THREADS
from src.p7_constantes import LOCAL_HOST, LOCAL_PORT
from src.p7_util import timer, clean_ram
from src.p7_file import make_dir
from src.p7_regex import sel_var
from src.p7_constantes import MODEL_DIR, DATA_INTERIM
from src.p7_constantes import LOCAL_HOST, LOCAL_PORT

print("mlflow", mlflow.__version__)
print("optuna", optuna.__version__)
print("numpy", np.__version__)
print("plotly", plotly.__version__)
print("kaleido", kaleido.__version__)


CONFIG_SEARCH = {
    "model_dir": MODEL_DIR,
    "model_type": "lightgbm",
    "subdir": "light_simple/",
    "data_dir": DATA_INTERIM,
    "train_filename": "train.csv",
    "feature_importance_filename": "feature_importance.csv",
    "n_predictors": 20,
    "moo_objective": False,
    "metric": "weighted_recall",
    "n_trials": 40,
}


def get_train(config=CONFIG_SEARCH):
    # On charge le train avec toutes les features
    train = pd.read_csv(os.path.join(DATA_INTERIM, config["train_filename"]))
    to_drop = sel_var(train.columns, "Unnamed")
    if to_drop:
        train = train.drop(to_drop, axis=1)
    train = train.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

    print("Forme de train.csv :", train.shape)
    return train


def get_sorted_features_by_importance(config=CONFIG_SEARCH):
    sorted_features_by_importance = (
        pd.read_csv(
            os.path.join(
                config["model_dir"],
                config["subdir"],
                config["feature_importance_filename"],
            )
        )
        .set_index("feature")
        .index.tolist()
    )
    return sorted_features_by_importance


def build_experiment(
    X,
    experiment_name=None,
    experiment_description=None,
    experiment_tags=None,
    config=CONFIG_SEARCH,
):
    if not experiment_name:
        # Si le nom de l'expérience n'est pas fourni en param, on le crée
        data_filepath = os.path.join(config["data_dir"], config["train_filename"])
        n_rows, n_cols = X.shape

        experiment_name = f"{config['subdir'][:-1]}_{n_cols}x{n_rows}_{config['metric']}_{config['n_trials']}trials"
        print("experiment_name", experiment_name)

    if not experiment_description:
        experiment_description = (
            f"Recherche d'hyperparamètres pour le modèle {config['subdir'][:-1]}, impact du nombre de features\n"
            "Recherche Bayesienne - mono objectif - Hyperband"
        )

    if not experiment_tags:
        experiment_tags = {
            "model": "lightgbm",
            "task": "hyperparam",
            "metric": config["metric"],
            "mlflow.note.content": experiment_description,
        }

    # On vérifie si l'expérience existe déjà
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)

    if existing_experiment:
        print(f"WARNING : L'expérience '{experiment_name}' existe déjà")
        experiment_id = existing_experiment.experiment_id

    else:
        print(f"Création de l'expérience '{experiment_name}'")
        experiment_id = mlflow.create_experiment(
            experiment_name,
            # artifact_location=os.path.join(MODEL_DIR, subdir).as_uri(),
            artifact_location=os.path.join(MODEL_DIR, config["subdir"]),
            tags=experiment_tags,
        )

    return experiment_id


def build_parent_run_name(config=CONFIG_SEARCH):
    run_name = f"{config['model_type']}_single_{config['metric']}_best"
    run_description = (
        f"Single objective - métrique {config['metric']} - all data kernel simple"
    )
    return run_name, run_description


"""def lgb_single_objective(trial):
    with mlflow.start_run(nested=True):
        # On définit les hyperparamètres
        params = {
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["dart", "gbdt"]
            ),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.0001, 0.5, log=True
            ),
            "max_bin": trial.suggest_int("max_bin", 128, 512, step=32),
            "n_estimators": trial.suggest_int("n_estimators", 40, 400, step=20),
            
        }

        other_params = {}

    return"""


def sk_single_objective(X, y, optimize_boosting_type=True, config=CONFIG_SEARCH):
    # Nécessite l'installation optuna-integration
    def _objective(trial):
        parent_run_name, parent_run_description = build_parent_run_name(config=config)
        child_run_name = f"N_{trial.number}"
        with mlflow.start_run(run_name=child_run_name, nested=True):
            if optimize_boosting_type:
                boosting_type = trial.suggest_categorical(
                    "boosting_type", ["dart", "gbdt"]
                )
            else:
                boosting_type = "gbdt"
            lambda_l1 = (trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),)
            lambda_l2 = (trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),)
            num_leaves = (trial.suggest_int("num_leaves", 2, 256),)
            feature_fraction = (trial.suggest_float("feature_fraction", 0.4, 1.0),)
            bagging_fraction = (trial.suggest_float("bagging_fraction", 0.4, 1.0),)
            bagging_freq = (trial.suggest_int("bagging_freq", 1, 7),)
            min_child_samples = (trial.suggest_int("min_child_samples", 5, 100),)
            learning_rate = (
                trial.suggest_float("learning_rate", 0.0001, 0.5, log=True),
            )
            max_bin = trial.suggest_int("max_bin", 128, 512, step=32)
            n_estimators = trial.suggest_int("n_estimators", 40, 400, step=20)

            hyperparams = {
                "boosting_type": boosting_type,
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "num_leaves": num_leaves,
                "feature_fraction": feature_fraction,
                "bagging_fraction": bagging_fraction,
                "bagging_freq": bagging_freq,
                "min_child_samples": min_child_samples,
                "learning_rate": learning_rate,
                "max_bin": max_bin,
                "n_estimators": n_estimators,
            }

            # 'binary' est la métrique d'erreur
            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial, "binary"
            )

            # Pour intégration optuna mlflow avec nested runs voir :
            # https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html?highlight=run%20description

            # On n'utilise pas cross_val_score pour des problèmes de RAM
            # scores = cross_val_score(model, X, y, scoring="f1_macro", cv=5)
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            auc_scores = []
            accuracy_scores = []
            balanced_accuracy_scores = []
            f1_scores = []
            recall_scores = []
            weighted_recall_scores = []
            precision_scores = []
            balanced_precision_scores = []
            fit_durations = []

            for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]

                model = lgb.LGBMClassifier(
                    # force_row_wise=True,
                    force_col_wise=True,
                    objective="binary",
                    is_unbalanced=False,
                    boosting_type=boosting_type,
                    n_estimators=n_estimators,
                    lambda_l1=lambda_l1,
                    lambda_l2=lambda_l2,
                    num_leaves=num_leaves,
                    feature_fraction=feature_fraction,
                    bagging_fraction=bagging_fraction,
                    bagging_freq=bagging_freq,
                    min_child_samples=min_child_samples,
                    learning_rate=learning_rate,
                    max_bin=max_bin,
                    callbacks=[pruning_callback],
                    verbose=-1,
                )
                t0_fit = time.time()
                model.fit(
                    X_train,
                    y_train,
                    # eval_set=[(X_train, y_train), (X_val, y_val)],
                    eval_set=(X_val, y_val),
                    eval_metric=config["metric"],
                )
                fit_durations.append(time.time() - t0_fit)

                pred_y = model.predict_proba(X_val)[:, 1]
                auc_scores.append(roc_auc_score(y_val, y_score=pred_y))

                pred_y = model.predict(X_val)
                f1_scores.append(f1_score(y_val, pred_y))
                recall_scores.append(recall_score(y_val, pred_y))
                weighted_recall_scores.append(
                    recall_score(y_val, pred_y, average="weighted")
                )
                accuracy_scores.append(accuracy_score(y_val, pred_y))
                balanced_accuracy_scores.append(balanced_accuracy_score(y_val, pred_y))
                precision_scores.append(precision_score(y_val, pred_y, zero_division=0))
                balanced_precision_scores.append(balanced_accuracy_score(y_val, pred_y))
                del X_train
                del y_train
                del X_val
                del y_val
                del model
                gc.collect()

            # Fin d'un trial
            del folds
            gc.collect()
            dic_metrics = {
                "auc": np.mean(auc_scores),
                "f1": np.mean(f1_scores),
                "weighted_recall": np.mean(weighted_recall_scores),
                "recall": np.mean(recall_scores),
                "accuracy": np.mean(accuracy_scores),
                "balanced_accuracy": np.mean(balanced_accuracy_scores),
                "precision": np.mean(precision_scores),
                "balanced_precision": np.mean(balanced_precision_scores),
                "fit_duration": np.mean(fit_durations),
            }
            for k, v in dic_metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_params(hyperparams)

        return dic_metrics[config["metric"]]

    return _objective


def sk_single_search(X, y, experiment_name=None, config=CONFIG_SEARCH):
    # Use the fluent API to set the tracking uri and the active experiment
    mlflow.set_tracking_uri(f"{LOCAL_HOST}:{LOCAL_PORT}")

    with timer("Optimize hyperparameters"):
        # Utilise l'algorithme d'optimisation TPE (Tree-structured Parzen Estimator) comme méthode d'échantillonnage
        # Il s'agit de l'algo qui génère les valeurs des hyperparams lors de chaque essai d'optimisation
        # Ici il est utilisé en conjonction avec le pruning Hyperband
        # => Le sampler choisit les params à essayer, le pruner les arrête prématurément si non performants
        sampler = optuna.samplers.TPESampler()

        # Optuna a réalisé plusieurs études empiriques avec différents algorithmes de pruning.
        # Empiriquement, l'algorithme Hyperband a donné les meilleurs résultats
        # Voir : https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako
        # reduction_factor contrôle combien de trials sont proposés dans chaque Halving Round
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=10, max_resource=300, reduction_factor=3
        )

        # On active l'expérience
        experiment_id = build_experiment(
            X=X, experiment_name=experiment_name, config=config
        )
        experiment_metadata = mlflow.set_experiment(experiment_id=experiment_id)
        print(f"Experience '{experiment_metadata.name}' activée")

        # On crèe une run MLFlow
        parent_run_name, parent_run_description = build_parent_run_name(config=config)

        with mlflow.start_run(
            experiment_id=experiment_id, run_name=parent_run_name, nested=True
        ) as run:
            # description du run
            mlflow.set_tag("mlflow.note.content", parent_run_description)

            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                study_name=f"study_{experiment_metadata.name}",
                # storage=os.path.join(MODEL_DIR, subdir)
            )

            # gc appelle le garbage collector après chaque trial
            study.optimize(
                sk_single_objective(
                    X=X, y=y, optimize_boosting_type=True, config=config
                ),
                n_trials=config["n_trials"],
                gc_after_trial=True,
                n_jobs=NUM_THREADS,
            )

            best_params = study.best_trial.params
            best_score = study.best_trial.value
            mlflow.log_params(best_params)

            fig = optuna.visualization.plot_parallel_coordinate(
                study,
                params=["boosting_type", "num_leaves", "learning_rate", "n_estimators"],
            )
            im_dir = os.path.join(config["model_dir"], config["subdir"])
            # fig.write_image(file=os.path.join(im_dir, "single_parallel_coordinates.png"), format="png", scale=6)
            fig.write_html(os.path.join(im_dir, "single_parallel_coordinates.html"))
            fig = optuna.visualization.plot_param_importances(study)
            # fig.write_image(file=os.path.join(im_dir, "single_hyperparam_importance.png"), format="png", scale=1)
            fig.write_html(os.path.join(im_dir, "single_hyperparam_importance.html"))

            mlflow.log_params(best_params)
            mlflow.log_metric(config["metric"], best_score)
            mlflow.log_artifact(
                os.path.join(im_dir, "single_parallel_coordinates.html")
            )
            mlflow.log_artifact(
                os.path.join(im_dir, "single_hyperparam_importance.html")
            )
            # mlflow.log_artifact(os.path.join(im_dir, "single_hyperparam_importance.png"))
        # Force mlflow à terminer le run même s'il y a une erreur dedans
        mlflow.end_run()
    return study


# Pénalise les Faux Negatifs dans le F1 Score
def weight_f1(y_true, y_pred, weight_fn=10):
    _, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # fn = Nombre de faux négatifs (oubli de prédire un défaut)
    # fp = Nombre de faux positifs (défaut prédit à tort)

    # f1 standard = 2 * tp / (2 * tp + fp + fn)

    # weighted_f1 = 2 * tp / (2 * tp + weight_fn * fn  + (1 - weight_fn) * fp)
    weighted_f1 = 2 * tp / (2 * tp + (weight_fn * fn + fp) / weight_fn)

    return weighted_f1


# Créer une métrique personnalisée à partir de la fonction de perte
# custom_f1 = make_scorer(weight_f1, greater_is_better=True)

"""
La fonction _objectif est appelée une fois pour chaque essai (trial).
Ici on entraîne un LGBMClassfier et on calcule la métrique : ? replacer le f1-score par une autre
Optuna passe un objet trial à la fonction _objectif, qu'on peut utiliser pour en définir les paramètres.
log=True : applique une log scale aux valeurs à tester dans l'étendue spécifiée (pour les valeurs num), 
Effet : plus de valeurs sont testées à proximité de la borne basse et moins (logarithmiquement) vers
la borne haute
Convient particulièrement bien au learning rate : on veut se concentrer sur des valeurs + petites et 
augmenter exponentiellement le pas des valeurs à tester pour les plus grandes

Le pruning callbac est le mécanisme qui applique le pruning dynamique pendant l'entraînement du modèle LightGBM 
dans le cadre d'une étude Optuna. 
1 - Initialisation du callback : 
Crée le callback de pruning qui surveillera l'entraînement du modèle LightGBM 
et effectuera le pruning selon les instructions spécifiées par l'étude Optuna
2 - Évaluation périodique : 
Pendant l'entraînement de LightGBM, le callback est appelé périodiquement
pour évaluer les performances du modèle en fonction des critères de pruning définis par Optuna.
3 - Décision de pruning : 
Le callback utilise les informations fournies par Optuna, 
telles que les valeurs d'objectif de l'essai actuel et les valeurs d'objectif des essais précédents, 
pour décider si l'essai actuel doit être pruned en fonction de son efficacité par rapport à d'autres essais.
4 - Pruning : 
Pruning appliqué si décidé (arrête l'entraînement de ce modèle LightGBM)
5 - Réévaluation et ajustement :
Après le pruning d'un essai, l'entraînement peut continuer avec les essais restants, 
et le processus de pruning peut être répété à intervalles réguliers jusqu'à ce que l'étude soit terminée.


"""


"""
Pruning (="Elagage") = technique pour arrêter prématurément l'exécution (train + eval) si 
il est peu probable que l'essai conduise à une amélioreration significative des performances
(l'entraînement est stoppé si l'algo sous-performe)

**** HYPERBAND

Hyperband est une technique de pruning qui combine le pruning de type "Successive halving" et
d'autres techniques (random search, multi-bracket resource allocation strategy).

1 - Successive Halving (Demi-seuccessifs):
Au début de l'optimisation, Hyperband crée un grand nombre de configurations d'hyperparamètres
et entraîne chaque modèle pendant un petit nombre d'itérations (eochs). 
Ensuite, il évalue les performances de ces modèles et conserve uniquement les meilleurs. 
Il répète ce processus plusieurs fois, en doublant à chaque fois le nombre d'itérations, 
mais en réduisant de moitié le nombre de configurations conservées.

Ainsi les modèles prometteurs bénéficient de plus d'itérations pour converger vers de bonnes performances,
tandis que les modèles moins prometteurs sont élagués à chaque étape.

2 - Pruning basé sur la performance:
À chaque étape du Successive Halving, Hyperband évalue périodiquement les performances des modèles en formation
et décide de "pruner" les modèles moins prometteurs. 
Les modèles qui n'atteignent pas un certain seuil de performance sont arrêtés prématurément.

*** MULTI_BRACKET RESSOURCE ALLOCATION strategy (Allocation de ressource à plusieurs 'brackets' ou niveaux / échelon)

divise les ressources disponibles (telles que le nombre d'itérations, d'évaluations de modèles, etc.)
en plusieurs "brackets" ou "niveaux" et alloue ces ressources de manière dynamique en fonction de la performance
des modèles à chaque niveau.

1 - Multi-brackets
L'algorithme divise le budget total de ressources (par exemple, le nombre maximal d'itérations ou le temps total de calcul) en plusieurs "brackets" ou "échelons".
Chaque bracket représente une étape distincte de l'algorithme où différents ensembles d'hyperparamètres sont évalués.
Chaque bracket est caractérisé par un budget de ressources spécifique, qui peut être soit le nombre d'itérations, soit le temps de calcul. 
Les brackets initiaux ont un budget élevé tandis que les brackets ultérieurs ont des budgets plus bas.

2 - Resource allocation
Alloue dynamiquement les ressources aux config d'hyperparams
Dans chaque bracket, Hyperband commence par évaluer un grand nombre de configurations d'hyperparamètres pour un nombre restreint d'itérations. 
Ensuite, il élimine les configurations sous-performantes et alloue davantage de ressources aux configurations les plus prometteuses.

Within each bracket, successive halving is applied to iteratively eliminate underperforming configurations and
allocate more resources to the remaining promising ones.
At the begining of each bracket, a new set of hyperparam configurations is sampled using random search.

=> reduce risk of missing good config
more efficient and effective hyperparam tuning.
"""


"""
SAMPLER TPE :
Tree-structured Parzen Estimator
TPESampler utilise une approche bayésienne pour estimer les distributions de probabilité des valeurs d'hyperparamètres,
en se basant sur les performances des essais précédents, 
afin de guider efficacement la recherche vers des régions prometteuses de l'espace des hyperparamètres.

1 - Initialisation des distributions :
L'algorithme TPE commence par définir des distributions de probabilité pour chaque hyperparamètre à optimiser.
Ces distributions peuvent être continues (comme des distributions gaussiennes) ou discrètes (comme des distributions uniformes).
Pour chaque hyperparamètre, deux distributions sont définies : 
une pour les valeurs considérées comme "bonnes" (positive), et une pour les valeurs considérées comme "mauvaises" (négative).

2 - Évaluation des essais :
À chaque étape de l'optimisation, 
TPE utilise les essais précédents pour estimer les distributions de probabilité des valeurs d'hyperparamètres.
Les essais sont divisés en deux groupes : 
ceux qui ont produit de bonnes performances et ceux qui ont produit de mauvaises performances, 
en fonction du critère d'objectif (par exemple, la précision d'un modèle).
Les distributions de probabilité sont mises à jour en utilisant les essais des deux groupes,
en ajustant les paramètres des distributions pour mieux modéliser les valeurs d'hyperparamètres qui ont conduit à de bonnes performances.

3 - Échantillonnage des essais :
Une fois que les distributions de probabilité ont été mises à jour, 
TPE échantillonne de nouvelles valeurs d'hyperparamètres à partir de ces distributions. 
Les valeurs échantillonnées sont généralement celles qui maximisent l'espérance d'une fonction d'acquisition 
(par exemple, l'espérance de l'amélioration de l'objectif) 
ou qui minimisent une fonction de coût (par exemple, l'espérance de la perte de l'objectif).

4 - Évaluation des performances :
Les nouvelles valeurs d'hyperparamètres échantillonnées sont utilisées pour effectuer de nouveaux essais.
Les performances de ces essais sont évaluées à l'aide de la fonction d'objectif, 
et les résultats sont utilisés pour mettre à jour les distributions de probabilité et répéter le processus.
"""

"""
Métriques pour notre cas :

Le Recall mesure la proportion de vrais positifs (défauts correctement prédits)
parmi tous les vrais positifs et faux négatifs (défauts réels).
En choisissant le Recall comme métrique, on se concentre sur la capacité du modèle à capturer la majorité des cas de défauts, 
minimisant ainsi le risque de faux négatifs (oublier de prédire un défaut lorsqu'il existe réellement).

Le F1-score est la moyenne harmonique de la précision et du recall.
Le F1-score est particulièrement utile lorsque les classes sont déséquilibrées car il donne plus de poids aux classes minoritaires. 
Il peut être une bonne métrique lorsqu'on veut trouver un compromis entre la précision et le recall.

AUC-ROC (Area Under the Receiver Operating Characteristic Curve) : L'AUC-ROC est une métrique qui mesure la capacité du modèle à classer
correctement les exemples positifs et négatifs. 
Elle serait robuste aux déséquilibres de classe et donne une indication de la capacité du modèle à discriminer entre les classes.
"""

"""
balanced accuracy
The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. 
It is defined as the average of recall obtained on each class.
The best value is 1 and the worst value is 0 when adjusted=False.
"""
