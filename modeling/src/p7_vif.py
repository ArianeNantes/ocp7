import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from cuml.linear_model import LinearRegression as CuLinearRegression
from cuml.metrics import r2_score

import gc
import cupy as cp
import time

from src.p7_util import format_time


# Elimine les features uune par une récursivement à l'aide du VIF sans validation croisée
# Pour les datasets ne comportant pas de NaN uniquement.
class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, vif_threshold=10.0, max_iter_reg=1000, verbose=True):
        # self.pipe_prepare = pipe_prepare
        self.vif_threshold = vif_threshold
        self.max_iter_reg = max_iter_reg
        # self.epsilon = epsilon
        self.verbose = verbose

    def fit(self, X, y=None, max_iter=100, epsilon=1e-6):
        self.vif_ = {}
        self.selected_features_ = []
        self.removed_features_ = []

        t0 = time.time()
        if self.verbose:
            print(
                f"Elimination récursive de features avec un VIF > {self.vif_threshold:.0f}"
            )
            print(f"\tDevice : GPU")
            print(f"\tshape X  : {X.shape}")
            print(f"\tSans validation croisée\n")
            if max_iter > 0:
                print(f"Itérations ({max_iter} itérations maximum)...")
            else:
                print(f"Itérations...")

        current_features = [
            f for f in X.columns.tolist() if f not in ["TARGET", "SK_ID_CURR"]
        ]

        iteration = 1

        while True:
            X_current = X[current_features].copy()
            # X_prepared = self.pipe_prepare.fit_transform(X_current, y)
            # X_prepared = X_prepared.astype(np.float64)

            vif_scores = {}
            for col in X_current.columns:
                col_target = X_current[col]
                X_others = X_current.drop(columns=[col])

                try:
                    # model = CuLinearRegression(max_iter=self.max_iter_reg, copy_X=False)
                    model = CuLinearRegression(copy_X=False)
                    model.fit(X_others, col_target)
                    y_pred = model.predict(X_others)
                    r2 = r2_score(col_target, y_pred)
                    vif = np.inf if (1.0 - r2) < epsilon else 1.0 / (1.0 - r2)
                    vif_scores[col] = float(vif)
                except Exception as e:
                    if self.verbose:
                        print(f"VIF non calculé pour {col} : {e}")

            self.vif_ = vif_scores

            worst_feature = max(vif_scores, key=vif_scores.get)
            worst_vif = vif_scores[worst_feature]

            if self.verbose:
                print(
                    f"[Itération {iteration}] (Elapsed time : {format_time((time.time()-t0))}) Max VIF: {worst_feature} = {worst_vif:.2f}"
                )

            if worst_vif < self.vif_threshold:
                break

            self.removed_features_.append(worst_feature)
            current_features.remove(worst_feature)

            if iteration == max_iter:
                print(
                    f"Arrêt du processus car le maximum d'itérations ({max_iter}) a été atteint"
                )
                break

            iteration += 1

            # Libération mémoire
            gc.collect()
            cp._default_memory_pool.free_all_blocks()

        self.selected_features_ = current_features
        duration = format_time(time.time() - t0)
        if self.verbose:
            print(f"Durée du Fit (hh:mm:ss) : {duration}")
            print(f"{len(self.removed_features_)} features ont un VIF trop élevé :")
            print(self.removed_features_)
            print(f"{len(self.selected_features_)} features restantes")
        return self

    def transform(self, X):
        return X.drop(columns=self.removed_features_, errors="ignore")

    def get_vif(self):
        return self.vif_
