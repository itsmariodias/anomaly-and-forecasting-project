import h2o
from h2o.estimators import (
    H2OIsolationForestEstimator,
    H2OKMeansEstimator,
    H2ODeepLearningEstimator,
)
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class H2OProbWrapper:
    def __init__(self, h2o_model, feature_names, output_name):
        self.h2o_model = h2o_model
        self.feature_names = feature_names
        self.output_name = output_name

    def predict_anomaly_score(self, X):
        self.dataframe = pd.DataFrame(X, columns=self.feature_names)
        self.predictions = get_isoforest_predictions(
            self.h2o_model, self.dataframe, self.feature_names
        )
        return self.predictions[self.output_name]

    def predict_kmeans_distance(self, X):
        self.dataframe = pd.DataFrame(X, columns=self.feature_names)
        self.predictions = get_kmeans_predictions(
            self.h2o_model, self.dataframe, self.feature_names
        )
        return self.predictions[self.output_name]

    def predict_lof_score(self, X):
        self.dataframe = pd.DataFrame(X, columns=self.feature_names)
        self.predictions = get_lof_predictions(
            self.h2o_model, self.dataframe, self.feature_names
        )
        return self.predictions[self.output_name]

    def predict_reconstruction_error(self, X):
        self.dataframe = pd.DataFrame(X, columns=self.feature_names)
        self.predictions = get_autoencoder_predictions(
            self.h2o_model, self.dataframe, self.feature_names
        )
        return self.predictions[self.output_name]

    def predict_svm_class(self, X):
        self.dataframe = pd.DataFrame(X, columns=self.feature_names)
        self.predictions = get_svm_predictions(
            self.h2o_model, self.dataframe, self.feature_names
        )
        return self.predictions[self.output_name]


def get_isoforest_predictions(
    isoforest: H2OIsolationForestEstimator, data: pd.DataFrame, feature_names: list
) -> pd.DataFrame:
    data = data[feature_names]
    h2o_frame = h2o.H2OFrame(data)
    predictions = isoforest.predict(h2o_frame)

    return predictions.as_data_frame(use_multi_thread=True)


def get_kmeans_predictions(
    kmeans: H2OKMeansEstimator, data: pd.DataFrame, feature_names: list
) -> pd.DataFrame:
    data = data[feature_names]
    data.reset_index(drop=True, inplace=True)

    h2o_frame = h2o.H2OFrame(data)

    # Predict cluster labels
    cluster_labels = kmeans.predict(h2o_frame).as_data_frame(use_multi_thread=True)
    # Find the cluster centers
    cluster_centers = kmeans.centers()
    # Calculate the distance from each point to its assigned cluster center
    predictions = data.apply(
        lambda x: np.linalg.norm(
            x - cluster_centers[cluster_labels.iloc[x.name]["predict"]]
        ),
        axis=1,
    ).to_frame("distance")

    return predictions


def get_lof_predictions(
    lof: LocalOutlierFactor,
    data: pd.DataFrame,
    feature_names: list,
    is_train: bool = False,
) -> pd.DataFrame:
    data = data[feature_names]

    if is_train:
        scores = (
            lof.negative_outlier_factor_
        )  # We are not supposed to use predict() for train data when using novelty=True mode
    else:
        scores = lof.score_samples(data.to_numpy())
    scores = -scores

    predictions = pd.DataFrame(scores, columns=["lof_scores"])

    return predictions


def get_svm_predictions(
    svm_model: OneClassSVM,
    data: pd.DataFrame,
    feature_names: list,
) -> pd.DataFrame:
    data = data[feature_names]

    scores = svm_model.score_samples(data.to_numpy())
    scores = -scores

    predictions = pd.DataFrame(scores, columns=["scores"])

    return predictions


def get_autoencoder_predictions(
    ae_model: H2ODeepLearningEstimator, data: pd.DataFrame, feature_names: list
) -> pd.DataFrame:
    data = data[feature_names]
    h2o_frame = h2o.H2OFrame(data)

    reconstruction_error = ae_model.anomaly(h2o_frame)

    predictions = reconstruction_error.as_data_frame(use_multi_thread=True)

    return predictions


def get_forecast(model, start, end):
    yhat = model.predict(start=start, end=end)
    return yhat


def calculate_thresholds(predictions: pd.DataFrame, column: str):

    quantile_frame = predictions.quantile([0.95, 0.99])
    soft_threshold = quantile_frame.at[0.95, column]
    hard_threshold = quantile_frame.at[0.99, column]

    return soft_threshold, hard_threshold


def classify_anomaly(
    predictions: pd.Series, soft_threshold: float, hard_threshold: float
):
    return predictions > soft_threshold, predictions > hard_threshold


def generate_commentary(
    shap_values: np.ndarray, feature_names: list, top: int = 5
) -> str:
    data = pd.Series(shap_values, index=feature_names)

    comment = f"Top {top} features affecting the anomaly score are:\n"

    data = data.sort_values(ascending=False)

    index = 1
    for feature, shap_value in list(data.items())[:top]:
        contribution = "increases" if shap_value > 0 else "decreases"
        comment += (
            f"{index}. {feature} {contribution} the score by {abs(shap_value):.2f}.\n"
        )
        index += 1

    return comment


def get_metrics(Y, Y_pred, model_parameters, prefix):
    soft_anomaly, hard_anomaly = classify_anomaly(
        Y_pred,
        model_parameters["soft_threshold"],
        model_parameters["hard_threshold"],
    )

    metrics = {
        f"{prefix}_average_precision": average_precision_score(Y, Y_pred),
        f"{prefix}_roc_auc": roc_auc_score(Y, Y_pred),
    }

    metrics.update(get_scores(Y, soft_anomaly, prefix=prefix+"_soft"))

    metrics.update(get_scores(Y, hard_anomaly, prefix=prefix+"_hard"))

    print(f"Metrics: {metrics}")

    return metrics


def get_scores(Y, Y_pred, prefix):
    return {
        f"{prefix}_accuracy": accuracy_score(Y, Y_pred),
        f"{prefix}_precision": precision_score(Y, Y_pred),
        f"{prefix}_recall": recall_score(Y, Y_pred),
        f"{prefix}_f1": f1_score(Y, Y_pred),
        f"{prefix}_f2": fbeta_score(Y, Y_pred, beta=2),
    }

def get_mse(Y, Y_pred):
    error = mean_squared_error(Y, Y_pred)
    print(f"MSE: {error} and RMSE: {error ** 0.5}")

    return error
