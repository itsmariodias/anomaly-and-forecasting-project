import numpy as np
import shap
import pandas as pd


def get_kernel_shap(
    model, background_dataset, model_parameters: dict
) -> shap.KernelExplainer:

    # TODO Is is possible to save the explainer?
    explainer = shap.KernelExplainer(
        model, background_dataset, seed=model_parameters.get("seed"), link="identity" #TODO figure link out + svm not working with link logit
    )

    return explainer


def generate_background_dataset(df: pd.DataFrame, model_parameters: dict):
    background_df = shap.kmeans( # TODO Get a more representative sample, e.g. Get 7 days consecutive, same day every month, samples of from different trends
        df[model_parameters["feature_names"]],
        model_parameters.get("background_dataset_clusters", 10),
    )

    return background_df


def explain_shap(
    data: pd.DataFrame,
    row_index: int,
    model,
    background_data,
    model_parameters: dict,
    plot_bar: bool = True,
) -> np.ndarray:
    row = data.iloc[[row_index]][model_parameters["feature_names"]]

    explainer = get_kernel_shap(model, background_data, model_parameters)

    shap_values = explainer(row, silent=True)

    if plot_bar:
        shap.plots.bar(shap_values[0])

    return shap_values.values[0]
