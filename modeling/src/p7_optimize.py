from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import optuna
import io
import json
from copy import deepcopy

from src.p7_constantes import (
    DATA_CLEAN_DIR,
    DATA_INTERIM,
    MODEL_DIR,
    VAL_SEED,
)

from src.p7_evaluate import EvaluatorCV
from src.p7_tracking import start_mlflow_ui, create_or_load_experiment
from src.p7_metric import business_gain_score

from src.p7_tracking import log_features
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE


class OptimBase:
    def __init__(
        self,
        X,
        y,
        fixed_params=None,
        to_suggest=None,
        mlflow_experiment_name="",
        threshold_prob=0.5,
        param_bg={
            "loss_fn": -10,
            "loss_fp": -1,
            "gain_tn": 1,
            "gain_tp": 1,
        },
    ):
        self.X = X
        self.y = y
        self.mlflow_experiment_name = mlflow_experiment_name
        self.exp_id = None
        # self.sampler = optuna.samplers.RandomSampler(seed=VAL_SEED)
        self.sampler = optuna.samplers.TPESampler(n_startup_trials=40, seed=VAL_SEED)
        self.n_folds = 5
        self.n_jobs = 14
        self.study = None
        # Contiendra toutes les infos sur le meilleur essai après optimisation
        self.best_trial = None
        if fixed_params is None:
            fixed_params = {
                "class_weight": "balanced",
                "n_jobs": 14,
                "random_state": VAL_SEED,
            }
        # Si on a oublié de préciser la graine ou le n_jobs on les rajoute
        self.fixed_params = fixed_params
        if "random_state" not in fixed_params.keys():
            self.fixed_params["random_state"] = VAL_SEED
        if "n_jobs" not in fixed_params.keys():
            self.fixed_params["n_jobs"] = 14
        if to_suggest is None:
            to_suggest = []

        self.threshold_prob = threshold_prob
        if "threshold_prob" in self.fixed_params.keys():
            self.threshold_prob = self.fixed_params["threshold_prob"]
        self.param_bg = param_bg
        self.to_suggest = to_suggest

    # A surcharger dans les classes enfants
    def objective(self, trial):
        return

    def optimize(self, n_trials=5):
        if not self.study:
            # On crée l'étude en précisant que l'on maximise la fonction objective (business_gain)
            self.study = optuna.create_study(direction="maximize", sampler=self.sampler)
            if self.mlflow_experiment_name:
                # Démarre l'interface mlflow si ce n'est pas déjà le cas
                _ = start_mlflow_ui()
                # Crée l'expérience si elle n'existe pas ou la charge si elle existe
                self.exp_id = create_or_load_experiment(
                    name=self.mlflow_experiment_name,
                    description=f"Optimisation d'hyperparamètres",
                )

        # On optimise en maximisant le business_gain
        self.study.optimize(
            self.objective, n_trials=n_trials, callbacks=[self.store_best_trial]
        )

        # Afficher les meilleurs paramètres
        print("Meilleur Business Gain : {:.4f}".format(self.study.best_value))
        print("Meilleurs hyperparamètres :")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")

        # Si une expérience mlflow a été créée, on crée un run pour le meilleur trial
        if self.exp_id:
            self.create_best_run()
        # S'il n'y a pas de traçage mlflow, on montre juste le plot des importances des paramètres
        else:
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.update_layout(title=f"Importance des hyperparamètres")
            fig.show()
        return

    def create_best_run(self):
        run_name = f"T_{self.study.best_trial.number}_best_of_{len(self.study.get_trials())}_trials"
        with mlflow.start_run(
            experiment_id=self.exp_id,
            run_name=run_name,
            # tags=run_tags,
            # description=run_description,
        ) as run:
            run_id = run.info.run_id
            mlflow.log_params(self.study.best_params)
            # metrics = {"business_gain": self.study.best_value}
            metrics = self.study.best_trial.user_attrs.get("val_scores")
            # mlflow.log_metrics(self.study.best_value)

            mlflow.log_metrics(metrics)
            if self.best_trial:
                json_buffer = io.StringIO()
                json.dump(self.best_trial, json_buffer, indent=2)
                json_buffer.seek(0)
                mlflow.log_text(
                    json_buffer.read(), artifact_file=f"best_trial/all_params.json"
                )

            log_features(self.X)

            # On logue les graphiques
            # plot importance des paramètres du trial
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.update_layout(
                title=f"Importance des hyperparamètres, Exp={self.mlflow_experiment_name}"
            )
            fig.show()
            mlflow.log_figure(fig, "hyperparam_importance.html")
            # plot
            fig = optuna.visualization.plot_parallel_coordinate(
                self.study,
                params=self.study.best_params,
                target_name="business_gain",
            )
            fig.update_layout(title=f"Parallel Plot, Exp={self.mlflow_experiment_name}")
            fig.show()
            mlflow.log_figure(fig, "parallel_coordinate.html")
            fig = optuna.visualization.plot_optimization_history(
                self.study, target_name="business_gain"
            )
            fig.update_layout(
                title=f"Historique d'optimisation, Exp={self.mlflow_experiment_name}"
            )
            fig.show()
            # fig.write_html(os.path.join(self.output_dir, "optimization_history.html"))
            mlflow.log_figure(fig, "optimization_history.html")
        return

    # Si le trial est le best, alors on stocke les paramètres (y compris ceux qui sont fixés) et le threshold dans la classe
    # On le fait dans un callback car dans la fonction objective, optuna ne sait pas si c'est le best trial.
    # La signature du callback réclame la study, c'est pourquoi on la passe en paramètre au lieu de self.study
    def store_best_trial(self, study, trial):
        if study.best_trial.number == trial.number:
            self.best_trial = {
                "model_params": trial.user_attrs.get("model_params", {}),
                "bg_params": trial.user_attrs.get("bg_params", {}),
                "threshold_prob": trial.user_attrs.get("threshold_prob"),
                "val_scores": trial.user_attrs.get("val_scores"),
                "trial_number": trial.number,
            }


class OptimLogReg(OptimBase):
    def __init__(
        self,
        X,
        y,
        fixed_params=None,
        to_suggest=None,
        mlflow_experiment_name="",
        threshold_prob=0.5,
        param_bg={
            "loss_fn": -10,
            "loss_fp": -1,
            "gain_tn": 1,
            "gain_tp": 1,
        },
    ):
        if fixed_params is None:
            fixed_params = {
                "class_weight": "balanced",
                "objective": "binary",
                "n_jobs": 14,
                "random_state": VAL_SEED,
                "verbosity": -1,
                "force_col_wise": True,
            }
        if to_suggest is None:
            to_suggest = [
                "solver",
                "C",
                "penalty",
                "l1_ratio",
            ]

        super().__init__(
            X=X,
            y=y,
            fixed_params=fixed_params,
            to_suggest=to_suggest,
            mlflow_experiment_name=mlflow_experiment_name,
            threshold_prob=threshold_prob,
            param_bg=param_bg,
        )
        # self.threshold_prob = threshold_prob
        # self.param_bg = param_bg

    def objective(self, trial):
        # Si l'expérience mlflow existe, on crée un run correspondant au trial Optuna
        run_id = None
        if self.exp_id:
            run_name = f"{trial.number}"
            with mlflow.start_run(
                experiment_id=self.exp_id,
                run_name=run_name,
                # tags=run_tags,
                # description=run_description,
            ) as run:
                run_id = run.info.run_id

        # On construit le dictionnaire avec les paramètres par défaut et on y ajoute les paramètres fixés
        model_params = LogisticRegression().get_params()
        for k in self.fixed_params.keys():
            if k != "threshold_prob":
                model_params[k] = self.fixed_params[k]

        # class_weight = trial.suggest_categorical("class_weight", ["balanced"])
        if "penalty" in self.to_suggest:
            model_params["penalty"] = trial.suggest_categorical(
                "penalty", ["l1", "l2", "elasticnet"]
            )
        if "solver" in self.to_suggest:
            model_params["solver"] = trial.suggest_categorical(
                "solver", ["liblinear", "saga"]
            )

        # Gestion des combinaisons invalides entre solver et pénalités
        if model_params["penalty"] == "elasticnet" and model_params["solver"] != "saga":
            raise optuna.exceptions.TrialPruned()
        if model_params["penalty"] == "l1" and model_params["solver"] not in [
            "liblinear",
            "saga",
        ]:
            raise optuna.exceptions.TrialPruned()

        if "C" in self.to_suggest:
            model_params["C"] = trial.suggest_loguniform("C", 1e-6, 1e2)

        if "max_iter" in self.to_suggest:
            model_params["max_iter"] = trial.suggest_int("max_iter", 100, 1000)

        if "l1_ratio" in self.to_suggest:
            # l1_ratio = None
            # Si elasticnet, on propose l1_ratio
            if model_params["penalty"] == "elasticnet":
                model_params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        if "threshold_prob" in self.to_suggest:
            threshold_prob = trial.suggest_float("threshold_prob", 0.1, 0.9)
        else:
            threshold_prob = self.threshold_prob

        model = LogisticRegression(**model_params)

        pipe = ImblearnPipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(k_neighbors=5, n_jobs=14, random_state=VAL_SEED)),
                (
                    "clf",
                    model,
                ),
            ]
        )
        """pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    model,
                ),
            ]
        )"""
        evaluator = EvaluatorCV(
            pipe=pipe,
            cv=5,
            device="CPU",
            score_train_set=False,
            param_bg=self.param_bg,
            random_state=VAL_SEED,
            verbose=False,
            threshold_prob=threshold_prob,
        )
        evaluator.evaluate(self.X, self.y)
        val_scores = evaluator.get_mean_val_scores(prefix="")
        business_gain = val_scores["business_gain"]

        # Dans optuna, il ne sera loggé par défaut que la métrique à optimiser, pour loguer toutes les métriques,
        # on personnalise pour enregistrer toutes les métriques
        trial.set_user_attr("val_scores", val_scores)

        # Idem pour tous les paramètres
        model_params = evaluator.get_model_params()
        bg_params = evaluator.get_bg_params()
        # threshold_prob = evaluator.threshold_prob
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr("bg_params", bg_params)
        trial.set_user_attr("threshold_prob", threshold_prob)

        # Si le run mlflow a été créé, on logue dans le run les paramètres et les scores de validation
        # Les artifacts (dont les paramètres de business_gain) ne seront logués que dans le best_run
        # Mais comme le seuil de décision est optimisé, on le logue dans les params
        if run_id:
            with mlflow.start_run(run_id=run_id):
                params = evaluator.get_model_params()
                # params["threshold_prob"] = threshold_prob
                params = trial.params
                mlflow.log_params(trial.params)
                mlflow.log_metrics(val_scores)
        return business_gain

    def get_best_model(self):
        if self.best_trial is not None:
            model_params = self.best_trial["model_params"]
            return LogisticRegression(**model_params)
        else:
            print("Le modèle n'a pas été optimisé")
            return


class OptimLgbm(OptimBase):
    def __init__(
        self,
        X,
        y,
        fixed_params=None,
        to_suggest=None,
        mlflow_experiment_name="",
        threshold_prob=0.5,
        param_bg={
            "loss_fn": -10,
            "loss_fp": -1,
            "gain_tn": 1,
            "gain_tp": 1,
        },
    ):
        if fixed_params is None:
            fixed_params = {
                "class_weight": "balanced",
                "objective": "binary",
                "n_jobs": 14,
                "random_state": VAL_SEED,
            }
        if to_suggest is None:
            to_suggest = [
                # "boosting_type",
                "learning_rate",
                "n_estimators",
                "num_leaves",
                "min_child_samples",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                # "reg_lambda",
                "max_depth",
                "min_split_gain",
            ]

        super().__init__(
            X=X,
            y=y,
            fixed_params=fixed_params,
            to_suggest=to_suggest,
            mlflow_experiment_name=mlflow_experiment_name,
        )
        self.sampler = optuna.samplers.TPESampler(n_startup_trials=40, seed=VAL_SEED)
        self.threshold_start = 0.1
        self.threshold_end = 0.9

    def objective(self, trial):
        # Si l'expérience mlflow existe, on crée un run correspondant au trial Optuna
        run_id = None
        if self.exp_id:
            run_name = f"{trial.number}"
            with mlflow.start_run(
                experiment_id=self.exp_id,
                run_name=run_name,
                # tags=run_tags,
                # description=run_description,
            ) as run:
                run_id = run.info.run_id

        # On construit le dictionnaire avec les paramètres par défaut et on y ajoute les paramètres fixés
        model_params = LGBMClassifier().get_params()
        for k in self.fixed_params.keys():
            if k != "threshold_prob":
                model_params[k] = self.fixed_params[k]

        # Manière dont les arbres sont entraînés et utilisés pour la prédiction
        # chaque arbre est entraîné pour corrigé les erreurs du précédent
        # gbdt : pas de dropout. Avantage : rapide, inconvénient : risque de surapprentissage
        # dart : droput. A chaque itération certains arbres sont masqués aléatoirement pendant l'entraînement du nouvel arbre
        #   Avantage : réduit le surfit, Inconvénient : plus lent
        if "boosting_type" in self.to_suggest:
            model_params["boosting_type"] = trial.suggest_categorical(
                "boosting_type", ["gbdt", "dart"]
            )
        #
        if "learning_rate" in self.to_suggest:
            model_params["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.0001, 0.5, log=True
            )

        if "n_estimators" in self.to_suggest:
            model_params["n_estimators"] = trial.suggest_int(
                "n_estimators", 100, 1000, step=20
            )

        # Nombre maximal de feuilles (plus grand = plus complexe)
        if "num_leaves" in self.to_suggest:
            model_params["num_leaves"] = trial.suggest_int(
                "num_leaves", 8, 255, log=True
            )

        # Nombre minimum d’observations dans une feuille
        if "min_child_samples" in self.to_suggest:
            model_params["min_child_samples"] = trial.suggest_int(
                "min_child_samples", 10, 100
            )

        # Regul L1
        if "reg_alpha" in self.to_suggest:
            model_params["reg_alpha"] = trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            )

        # regul L2
        if "reg_lambda" in self.to_suggest:
            model_params["reg_lambda"] = trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True
            )

        if "max_depth" in self.to_suggest:
            # On veut un int entre 3 et 16 ou la valeur -1 (profondeur sans limite)
            model_params["max_depth"] = trial.suggest_categorical(
                "max_depth", [-1] + list(range(3, 17))
            )

        # Gain minimal pour faire un split
        if "min_split_gain" in self.to_suggest:
            model_params["min_split_gain"] = trial.suggest_float(
                "min_split_gain", 0.0, 1.0
            )

        # Ratio de colonnes échantillonées à chaque arbre
        if "colsample_bytree" in self.to_suggest:
            model_params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            )

        # Ratio de lignes échantillonées (bagging)
        if "subsample" in self.to_suggest:
            model_params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        # Ratio pondération des classes
        if "scale_pos_weight" in self.to_suggest:
            model_params["scale_pos_weight"] = trial.suggest_float(
                "scale_pos_weight", 0.5, 10.0
            )

        if "threshold_prob" in self.to_suggest:
            threshold_prob = trial.suggest_float(
                "threshold_prob", self.threshold_start, self.threshold_end
            )
        else:
            threshold_prob = self.threshold_prob

        model = LGBMClassifier(**model_params)
        pipe = model

        evaluator = EvaluatorCV(
            pipe=pipe,
            cv=5,
            device="CPU",
            score_train_set=False,
            param_bg=self.param_bg,
            random_state=VAL_SEED,
            verbose=False,
            threshold_prob=threshold_prob,
        )
        evaluator.evaluate(self.X, self.y)
        val_scores = evaluator.get_mean_val_scores(prefix="")
        business_gain = val_scores["business_gain"]

        # Dans optuna, il ne sera loggé par défaut que la métrique à optimiser, pour loguer toutes les métriques,
        # on personnalise pour enregistrer toutes les métriques
        trial.set_user_attr("val_scores", val_scores)

        # Idem pour tous les paramètres
        model_params = evaluator.get_model_params()
        bg_params = evaluator.get_bg_params()
        # threshold_prob = evaluator.threshold_prob
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr("bg_params", bg_params)
        trial.set_user_attr("threshold_prob", threshold_prob)

        # Si le run mlflow a été créé, on logue dans le run les paramètres et les scores de validation
        # Les artifacts (dont les paramètres de business_gain) ne seront logués que dans le best_run
        # Mais comme le seuil de décision est optimisé, on le logue dans les params
        if run_id:
            with mlflow.start_run(run_id=run_id):
                params = evaluator.get_model_params()
                # params["threshold_prob"] = threshold_prob
                params = trial.params
                mlflow.log_params(trial.params)
                mlflow.log_metrics(val_scores)

        return business_gain

    def get_best_model(self):
        if self.best_trial is not None:
            model_params = self.best_trial["model_params"]
            return LGBMClassifier(**model_params)
        else:
            print("Le modèle n'a pas été optimisé")
            return


class OptimThreshold:
    def __init__(
        self,
        model,
        n_thresholds=99,
        start=0.01,
        stop=0.99,
        random_state=42,
        mlflow_experiment_name="",
    ):
        self.model = deepcopy(model)
        if isinstance(model, LogisticRegression):
            self.pipe = ImblearnPipeline(
                [
                    ("imputer", SimpleImputer()),
                    ("scaler", StandardScaler()),
                    ("smote", SMOTE(k_neighbors=5, n_jobs=14, random_state=VAL_SEED)),
                    (
                        "clf",
                        model,
                    ),
                ]
            )
        else:
            self.pipe = self.model
        self.n_thresholds = n_thresholds
        self.thresholds = np.linspace(start, stop, self.n_thresholds)
        self.mean_scores = None
        self.std_scores = None
        self.best_business_gain = None
        self.best_threshold = None
        self.best_results = None
        self.mlflow_experiment_name = mlflow_experiment_name
        self.exp_id = None
        self.subtitle = f"{self.model.__class__.__name__}"

    def optimize(self, X, y, param_bg, cv=5, verbose=True):
        # Si un nom d'expérience est donné, on crée une expérience mlflow
        if self.mlflow_experiment_name:
            # Démarre l'interface mlflow si ce n'est pas déjà le cas
            _ = start_mlflow_ui()
            # Crée l'expérience si elle n'existe pas ou la charge si elle existe
            self.exp_id = create_or_load_experiment(
                name=self.mlflow_experiment_name,
                description=f"Optimisation du seuil de probabilité pour modèle {self.model.__class__.__name__}",
            )
            # On crée le run à l'intérieur de l'expérience
            run_name = f"Optim_threshold_probability_{self.model.__class__.__name__}"
            with mlflow.start_run(
                experiment_id=self.exp_id,
                run_name=run_name,
            ) as run:
                run_id = run.info.run_id

                # On effectue l'optimisation
                best_threshold, best_business_gain = self.optimize_without_mlflow(
                    X=X, y=y, param_bg=param_bg, cv=cv, verbose=verbose
                )
                # On logue en param du run : le threshold, en métrique : le business_gain et dans les artifacts : une récap
                mlflow.log_params({"treshold_prob": best_threshold})
                mlflow.log_metrics({"business_gain": best_business_gain})

                # On construit la récapitulation, on la stocke et on la logue :
                self.best_results = {
                    "model_params": self.model.get_params(),
                    "bg_params": param_bg,
                    "threshold_prob": best_threshold,
                    "business_gain": best_business_gain,
                }
                if self.best_results:
                    json_buffer = io.StringIO()
                    json.dump(self.best_results, json_buffer, indent=2)
                    json_buffer.seek(0)
                    mlflow.log_text(
                        json_buffer.read(),
                        artifact_file=f"best_results/all_params.json",
                    )

                # On logue les features
                log_features(X)

                # On trace le graphique et on le logue
                fig = self.plot_bg()
                # img = pil_image_from_fig(fig)
                # mlflow.log_image(img, artifact_file="plot_optim_threshold.png")
                mlflow.log_figure(fig, "plot_optim_threshold.png")
        return

    def optimize_without_mlflow(self, X, y, param_bg, cv=5, verbose=True):

        if verbose:
            print("Optimisation du seuil de probabilité")
            print(
                f"\tparam Business Gain : loss_fn={param_bg['loss_fn']}, loss_fp={param_bg['loss_fp']}, gain_tn={param_bg['gain_tn']}, gain_tp={param_bg['gain_tp']}"
            )
            print(f"\tNombre de folds : {cv}")
            print(f"\tNombre de seuils à chaque fold : {self.n_thresholds}\n")
        thresholds = self.thresholds

        all_scores = []

        self.subtitle = f"{self.model.__class__.__name__} optimisé\nshape X : {X.shape[0]} observations {X.shape[1]} features"

        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=VAL_SEED)
        for fold_num, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            if verbose:
                print(f"Fold {fold_num + 1}/{cv}...")
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]

            scores_fold = []
            for threshold in thresholds:
                self.pipe.fit(X_train, y_train)
                y_prob = self.pipe.predict_proba(X_val)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                score = business_gain_score(
                    tn=tn,
                    fp=fp,
                    fn=fn,
                    tp=tp,
                    loss_fn=param_bg["loss_fn"],
                    loss_fp=param_bg["loss_fp"],
                    gain_tn=param_bg["gain_tn"],
                    gain_tp=param_bg["gain_tp"],
                )
                scores_fold.append(score)

            # Chaque ligne de all_score représente un fold, chaque colonne représente un seuil de probabilité
            all_scores.append(scores_fold)
        # On calcule la moyenne et les écarts types au travers des folds (donc en ligne)
        mean_scores = np.mean(all_scores, axis=0)
        std_scores = np.std(all_scores, axis=0)
        # Le meilleur seuil de probabilité est celui qui maximise le business gain moyen
        best_threshold = thresholds[np.argmax(mean_scores)]
        best_business_gain = np.max(mean_scores)

        # On stocke les résultats dans l'instance de l'objet
        self.best_threshold = best_threshold
        self.mean_scores = mean_scores
        self.std_scores = std_scores
        self.best_business_gain = best_business_gain

        if verbose:
            print(f"Meilleur seuil de probabilité :{best_threshold:.4f}")
            print(f"Meilleur Business Gain : {best_business_gain:.4f}")
        return best_threshold, best_business_gain

    def plot_bg(self, title="", subtitle=""):
        upper_scores = self.mean_scores + self.std_scores
        lower_scores = self.mean_scores - self.std_scores
        figsize = (10, 6)
        fig, ax = plt.subplots(figsize=figsize)

        if not title:
            title = "Business Gain moyen en fonction du seuil de probabilité"
        if not subtitle:
            subtitle = self.subtitle
        plt.plot(
            self.thresholds,
            self.mean_scores,
            label=f"Business Gain (max = {self.best_business_gain:.2f})",
        )
        ax.fill_between(
            self.thresholds,
            lower_scores,
            upper_scores,
            color="blue",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        plt.axvline(
            x=self.best_threshold,
            linestyle="--",
            color="black",
            label=f"Meilleur seuil ({self.best_threshold:.2f})",
        )

        plt.xlabel("Seuil de probabilité")
        plt.ylabel("Business Gain")
        fig.suptitle(title)
        ax.set_title(subtitle, ha="left", x=0, fontsize=ax.xaxis.label.get_fontsize())
        plt.legend()
        plt.grid(True)
        # plt.show()
        return fig


class OptimThreshold_old:
    def __init__(
        self,
        model,
        n_thresholds=99,
        random_state=42,
        # mlflow_experiment_name="",
    ):
        self.model = deepcopy(model)
        if isinstance(model, LogisticRegression):
            self.pipe = Pipeline(
                [
                    ("imputer", SimpleImputer()),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        self.model,
                    ),
                ]
            )
        else:
            self.pipe = self.model
        self.n_thresholds = n_thresholds
        self.thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        self.mean_scores = None
        self.std_scores = None
        self.best_business_gain = None
        self.best_threshold = None
        self.subtitle = f"{self.model.__class__.__name__}"

    """def optimize_and_track_mlflow(self, X, y, param_bg, exp_id, cv=5, verbose=True):
        run_name = f"Optim_threshold_probability"
        with mlflow.start_run(
            experiment_id=exp_id,
            run_name=run_name,
            # tags=run_tags,
            # description=run_description,
        ) as run:
            run_id = run.info.run_id
            best_threshold, best_business_gain = self.optimize(X=X, y=y, param_bg=param_bg, cv=cv, verbose=verbose)
            

        return"""

    def optimize(self, X, y, param_bg, cv=5, verbose=True, experiment_id=None):

        if verbose:
            print("Optimisation du seuil de probabilité")
            print(
                f"\tparam Business Gain : loss_fn={param_bg['loss_fn']}, loss_fp={param_bg['loss_fn']}, gain_tn={param_bg['gain_tn']}, gain_tp={param_bg['gain_tp']}"
            )
            print(f"\tNombre de folds : {cv}")
            print(f"\tNombre de seuils à chaque fold :{self.n_thresholds + 1}\n")
        thresholds = self.thresholds

        all_scores = []

        self.subtitle = f"{self.model.__class__.__name__} optimisé\nshape X : {X.shape[0]} observations {X.shape[1]} features"

        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=VAL_SEED)
        for fold_num, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            if verbose:
                print(f"Fold {fold_num + 1}/{cv}...")
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]

            scores_fold = []
            for threshold in thresholds:
                self.pipe.fit(X_train, y_train)
                y_prob = self.pipe.predict_proba(X_val)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                score = business_gain_score(
                    tn=tn,
                    fp=fp,
                    fn=fn,
                    tp=fp,
                    loss_fn=param_bg["loss_fn"],
                    loss_fp=param_bg["loss_fp"],
                    gain_tn=param_bg["gain_tn"],
                    gain_tp=param_bg["gain_tp"],
                )
                scores_fold.append(score)

            # Chaque ligne de all_score représente un fold, chaque colonne représente un seuil de probabilité
            all_scores.append(scores_fold)
        # On calcule la moyenne et les écarts types au travers des folds (donc en ligne)
        mean_scores = np.mean(all_scores, axis=0)
        std_scores = np.std(all_scores, axis=0)
        # Le meilleur seuil de probabilité est celui qui maximise le business gain moyen
        best_threshold = thresholds[np.argmax(mean_scores)]
        best_business_gain = np.max(mean_scores)

        # On stocke les résultats dans l'instance de l'objet
        self.best_threshold = best_threshold
        self.mean_scores = mean_scores
        self.std_scores = std_scores
        self.best_business_gain = best_business_gain

        if verbose:
            print(f"Meilleur seuil de probabilité :{best_threshold:.4f}")
            print(f"Meilleur Business Gain : {best_business_gain:.4f}")
        return best_threshold, best_business_gain

    def plot_bg(self, title="", subtitle=""):
        upper_scores = self.mean_scores + self.std_scores
        lower_scores = self.mean_scores - self.std_scores
        figsize = (10, 6)
        fig, ax = plt.subplots(figsize=figsize)

        if not title:
            title = "Business Gain moyen en fonction du seuil de probabilité"
        if not subtitle:
            subtitle = self.subtitle
        plt.plot(
            self.thresholds,
            self.mean_scores,
            label=f"Business Gain (max = {self.best_business_gain:.2f})",
        )
        ax.fill_between(
            self.thresholds,
            lower_scores,
            upper_scores,
            color="blue",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        plt.axvline(
            x=self.best_threshold,
            linestyle="--",
            color="black",
            label=f"Meilleur seuil ({self.best_threshold:.2f})",
        )

        plt.xlabel("Seuil de probabilité")
        plt.ylabel("Business Gain")
        fig.suptitle(title)
        ax.set_title(subtitle, ha="left", x=0, fontsize=ax.xaxis.label.get_fontsize())
        plt.legend()
        plt.grid(True)
        # plt.show()
        return fig
