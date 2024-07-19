<<<<<<< HEAD
import h2o
from h2o.estimators import (
    H2OIsolationForestEstimator,
    H2OKMeansEstimator,
    H2ODeepLearningEstimator,
)
from h2o.grid.grid_search import H2OGridSearch
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def train_isolation_forest(
    df: pd.DataFrame, model_parameters: dict
) -> H2OIsolationForestEstimator:

    h2o_frame = h2o.H2OFrame(df)

    isoforest = H2OIsolationForestEstimator(
        ntrees=model_parameters["ntrees"],
        sample_size=model_parameters["sample_size"],
        seed=model_parameters["seed"],
    )

    isoforest.train(
        training_frame=h2o_frame,
        x=model_parameters.get("feature_names"),
    )

    return isoforest


def train_kmeans(df: pd.DataFrame, model_parameters: dict) -> H2OKMeansEstimator:
    h2o_frame = h2o.H2OFrame(df)

    kmeans = H2OKMeansEstimator(
        k=model_parameters["k"],
        estimate_k=model_parameters.get("esitmate_k", True),
        standardize=model_parameters.get("standardize", False),
        seed=model_parameters["seed"],
    )

    kmeans.train(
        training_frame=h2o_frame,
        x=model_parameters.get("feature_names"),
    )

    return kmeans


def train_local_outlier_factor(
    df: pd.DataFrame, model_parameters: dict
) -> LocalOutlierFactor:

    lof = LocalOutlierFactor(
        n_neighbors=model_parameters.get("n_neighbors", 20), novelty=True, n_jobs=10
    )

    np_array = df[model_parameters["feature_names"]].to_numpy()

    lof = lof.fit(np_array)

    return lof


def train_svm(df: pd.DataFrame, model_parameters: dict):
    svm_model = OneClassSVM(
        gamma=model_parameters.get("gamma", "scale"),
        tol=model_parameters.get("tol", 1e-3),
        nu=model_parameters.get("nu", 0.5),
        max_iter=model_parameters.get("max_iterations", -1),
    )

    np_array = df[model_parameters["feature_names"]].to_numpy()

    svm_model.fit(X=np_array)

    return svm_model


def train_autoencoder(
    df: pd.DataFrame, model_parameters: dict, grid_search: bool = False
) -> H2ODeepLearningEstimator:

    h2o_frame = h2o.H2OFrame(df)
    # train, test = h2o_frame.split_frame(ratios = [0.75], destination_frames=["train", "test"], seed = model_parameters.get("seed"))

    if grid_search:
        # create hyperameter and search criteria lists (ranges are inclusive..exclusive))
        dl_hyper_params_tune = {
            "activation": ["Tanh", "RectifierWithDropout", "TanhWithDropout"],
            "hidden": [[2], [5], [10], [2, 2]],
            "input_dropout_ratio": [0, 0.05],
            "l1": [
                0,
                1e-05,
                2e-05,
                3e-05,
                4e-05,
                5e-05,
                6e-05,
                7e-05,
                8e-05,
                9e-05,
                1e-04,
            ],
            "l2": [
                0,
                1e-05,
                2e-05,
                3e-05,
                4e-05,
                5e-05,
                6e-05,
                7e-05,
                8e-05,
                9e-05,
                1e-04,
            ],
        }

        random_search_criteria_tune = {
            "strategy": "RandomDiscrete",
            "max_runtime_secs": 60,
            "seed": model_parameters.get("seed"),
            "stopping_rounds": 5,  # stop grid search once the best models have similar AUC
            "stopping_metric": "MSE",
            "stopping_tolerance": 1e-3,
        }

        dl_model = H2ODeepLearningEstimator(
            seed=model_parameters.get("seed"),
            epochs=model_parameters.get("epochs", 10.0),
            stopping_rounds=model_parameters.get("stopping_rounds", 5),
            stopping_metric=model_parameters.get("stopping_metric", "auto"),
            stopping_tolerance=model_parameters.get("stopping_tolerance", 0.0),
            autoencoder=True,
            reproducible=True,
        )

        ## Run Deep Learning Grid Search
        dl_grid = H2OGridSearch(
            dl_model,
            hyper_params=dl_hyper_params_tune,
            grid_id="ae_grid",
            search_criteria=random_search_criteria_tune,
        )

        dl_grid.train(
            x=model_parameters.get("feature_names"),
            max_runtime_secs=3600,
            training_frame=h2o_frame,
        )

        # Get Top best model
        best_model = dl_grid.get_grid("mse").models[0]

        hyperparams_dict = dl_grid.get_hyperparams_dict(best_model.key)
        print(f"Hyper parameters: {hyperparams_dict}")

        model_parameters.update(hyperparams_dict)

        return best_model, dl_grid
    else:
        dl_model = H2ODeepLearningEstimator(
            seed=model_parameters.get("seed"),
            epochs=model_parameters.get("epochs", 10.0),
            stopping_rounds=model_parameters.get("stopping_rounds", 5),
            stopping_metric=model_parameters.get("stopping_metric", "auto"),
            stopping_tolerance=model_parameters.get("stopping_tolerance", 0.0),
            activation=model_parameters.get["activation"],
            hidden=model_parameters.get["hidden"],
            input_dropout_ratio=model_parameters["input_dropout_ratio"],
            l1=model_parameters["l1"],
            l2=model_parameters["l2"],
            autoencoder=True,
            reproducible=True,
        )

        dl_model.train(
            x=model_parameters.get("feature_names"),
            max_runtime_secs=3600,
            training_frame=h2o_frame,
        )

        return dl_model


def train_ar(df: pd.DataFrame, model_parameters: dict):
    model = AutoReg(df[model_parameters["target"]], lags=model_parameters["lags"])
    model_fit = model.fit()

    return model_fit


def train_arima(df: pd.DataFrame, model_parameters: dict):
    model = ARIMA(
        df[model_parameters["target"]],
        order=(model_parameters["p"], model_parameters["d"], model_parameters["q"]),
    )
    model_fit = model.fit()

    return model_fit


def train_xgboost(df: pd.DataFrame, model_parameters: dict):
    model = xgb.XGBRegressor(random_state=model_parameters["seed"])
    model.fit(df[model_parameters["feature_names"]], df[model_parameters["target"]])

=======
import h2o
from h2o.estimators import (
    H2OIsolationForestEstimator,
    H2OKMeansEstimator,
    H2ODeepLearningEstimator,
)
from h2o.grid.grid_search import H2OGridSearch
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def train_isolation_forest(
    df: pd.DataFrame, model_parameters: dict
) -> H2OIsolationForestEstimator:

    h2o_frame = h2o.H2OFrame(df)

    isoforest = H2OIsolationForestEstimator(
        ntrees=model_parameters["ntrees"],
        sample_size=model_parameters["sample_size"],
        seed=model_parameters["seed"],
    )

    isoforest.train(
        training_frame=h2o_frame,
        x=model_parameters.get("feature_names"),
    )

    return isoforest


def train_kmeans(df: pd.DataFrame, model_parameters: dict) -> H2OKMeansEstimator:
    h2o_frame = h2o.H2OFrame(df)

    kmeans = H2OKMeansEstimator(
        k=model_parameters["k"],
        estimate_k=model_parameters.get("esitmate_k", True),
        standardize=model_parameters.get("standardize", False),
        seed=model_parameters["seed"],
    )

    kmeans.train(
        training_frame=h2o_frame,
        x=model_parameters.get("feature_names"),
    )

    return kmeans


def train_local_outlier_factor(
    df: pd.DataFrame, model_parameters: dict
) -> LocalOutlierFactor:

    lof = LocalOutlierFactor(
        n_neighbors=model_parameters.get("n_neighbors", 20), novelty=True, n_jobs=10
    )

    np_array = df[model_parameters["feature_names"]].to_numpy()

    lof = lof.fit(np_array)

    return lof


def train_svm(df: pd.DataFrame, model_parameters: dict):
    svm_model = OneClassSVM(
        gamma=model_parameters.get("gamma", "scale"),
        tol=model_parameters.get("tol", 1e-3),
        nu=model_parameters.get("nu", 0.5),
        max_iter=model_parameters.get("max_iterations", -1),
    )

    np_array = df[model_parameters["feature_names"]].to_numpy()

    svm_model.fit(X=np_array)

    return svm_model


def train_autoencoder(
    df: pd.DataFrame, model_parameters: dict, grid_search: bool = False
) -> H2ODeepLearningEstimator:

    h2o_frame = h2o.H2OFrame(df)
    # train, test = h2o_frame.split_frame(ratios = [0.75], destination_frames=["train", "test"], seed = model_parameters.get("seed"))

    if grid_search:
        # create hyperameter and search criteria lists (ranges are inclusive..exclusive))
        dl_hyper_params_tune = {
            "activation": ["Tanh", "RectifierWithDropout", "TanhWithDropout"],
            "hidden": [[2], [5], [10], [2, 2]],
            "input_dropout_ratio": [0, 0.05],
            "l1": [
                0,
                1e-05,
                2e-05,
                3e-05,
                4e-05,
                5e-05,
                6e-05,
                7e-05,
                8e-05,
                9e-05,
                1e-04,
            ],
            "l2": [
                0,
                1e-05,
                2e-05,
                3e-05,
                4e-05,
                5e-05,
                6e-05,
                7e-05,
                8e-05,
                9e-05,
                1e-04,
            ],
        }

        random_search_criteria_tune = {
            "strategy": "RandomDiscrete",
            "max_runtime_secs": 60,
            "seed": model_parameters.get("seed"),
            "stopping_rounds": 5,  # stop grid search once the best models have similar AUC
            "stopping_metric": "MSE",
            "stopping_tolerance": 1e-3,
        }

        dl_model = H2ODeepLearningEstimator(
            seed=model_parameters.get("seed"),
            epochs=model_parameters.get("epochs", 10.0),
            stopping_rounds=model_parameters.get("stopping_rounds", 5),
            stopping_metric=model_parameters.get("stopping_metric", "auto"),
            stopping_tolerance=model_parameters.get("stopping_tolerance", 0.0),
            autoencoder=True,
            reproducible=True,
        )

        ## Run Deep Learning Grid Search
        dl_grid = H2OGridSearch(
            dl_model,
            hyper_params=dl_hyper_params_tune,
            grid_id="ae_grid",
            search_criteria=random_search_criteria_tune,
        )

        dl_grid.train(
            x=model_parameters.get("feature_names"),
            max_runtime_secs=3600,
            training_frame=h2o_frame,
        )

        # Get Top best model
        best_model = dl_grid.get_grid("mse").models[0]

        hyperparams_dict = dl_grid.get_hyperparams_dict(best_model.key)
        print(f"Hyper parameters: {hyperparams_dict}")

        model_parameters.update(hyperparams_dict)

        return best_model, dl_grid
    else:
        dl_model = H2ODeepLearningEstimator(
            seed=model_parameters.get("seed"),
            epochs=model_parameters.get("epochs", 10.0),
            stopping_rounds=model_parameters.get("stopping_rounds", 5),
            stopping_metric=model_parameters.get("stopping_metric", "auto"),
            stopping_tolerance=model_parameters.get("stopping_tolerance", 0.0),
            activation=model_parameters.get["activation"],
            hidden=model_parameters.get["hidden"],
            input_dropout_ratio=model_parameters["input_dropout_ratio"],
            l1=model_parameters["l1"],
            l2=model_parameters["l2"],
            autoencoder=True,
            reproducible=True,
        )

        dl_model.train(
            x=model_parameters.get("feature_names"),
            max_runtime_secs=3600,
            training_frame=h2o_frame,
        )

        return dl_model


def train_ar(df: pd.DataFrame, model_parameters: dict):
    model = AutoReg(df[model_parameters["target"]], lags=model_parameters["lags"])
    model_fit = model.fit()

    return model_fit


def train_arima(df: pd.DataFrame, model_parameters: dict):
    model = ARIMA(
        df[model_parameters["target"]],
        order=(model_parameters["p"], model_parameters["d"], model_parameters["q"]),
    )
    model_fit = model.fit()

    return model_fit


def train_xgboost(df: pd.DataFrame, model_parameters: dict):
    model = xgb.XGBRegressor(random_state=model_parameters["seed"])
    model.fit(df[model_parameters["feature_names"]], df[model_parameters["target"]])

>>>>>>> 0e34c14 (Initial commit)
    return model