import os
import pickle
import h2o
import matplotlib.pyplot as plt
import pandas as pd


# Load a set of pickle files, put them together in a single DataFrame, and order them by time
# It takes as input the folder DIR_INPUT where the files are stored, and the BEGIN_DATE and END_DATE
def read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE):

    files = [
        os.path.join(DIR_INPUT, f)
        for f in os.listdir(DIR_INPUT)
        if f >= BEGIN_DATE + ".pkl" and f <= END_DATE + ".pkl"
    ]

    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)

    df_final = df_final.sort_values("TRANSACTION_ID")
    df_final.reset_index(drop=True, inplace=True)
    #  Note: -1 are missing values for real world data
    df_final = df_final.replace([-1], 0).infer_objects(copy=False)

    return df_final


def load_data(path: str) -> pd.DataFrame:
    df = h2o.import_file(path)
    return df.as_data_frame(use_multi_thread=True)


def load_model(model_name: str, model_dir: str, is_h2o: bool = True):
    # Load the Pickle file
    model_path = get_versioned_filename(model_name, model_dir, new_file=False)
    with open(model_path, "rb") as pklf:
        data = pickle.load(pklf)

        # Retrieve the model data from the Pickle file
        model_data = data["model"]

        if is_h2o:
            # Save the model data to a temporary file
            model_temp_path = os.path.join(model_dir, "model_temp.bin")
            with open(model_temp_path, "wb") as model_file:
                model_file.write(model_data)

            # Load the model into H2O from the temporary file
            model = h2o.load_model(model_temp_path)

            # Clean up temporary files if necessary
            os.remove(model_temp_path)
        else:
            model = model_data

        print(f"Loaded model {model_name}")

        return model, data["background_data"], data["parameters"]


def save_model(
    model,
    background_data,
    parameters,
    save_dir: str,
    model_name: str,
    is_h2o: bool = True,
):

    if is_h2o:
        # Save the H2O model
        model_path = h2o.save_model(model=model, path=save_dir, force=True)

        # Load and store the model file
        with open(model_path, "rb") as model_file:
            model_data = model_file.read()

        # Clean up temporary files
        os.remove(model_path)
    else:
        model_data = model

    # Save all data into a Pickle file
    pickle_path = get_versioned_filename(model_name, save_dir, new_file=True)
    with open(pickle_path, "wb") as pklf:

        # Store data
        data = {
            "model": model_data,
            "parameters": parameters,
        }

        if background_data:
            data["background_data"] = background_data

        pickle.dump(data, pklf)

    print(f"Saved model and data at {pickle_path}")
    return pickle_path.split("/")[-1]


def plot_line_chart(
    df: pd.DataFrame, x: str, y: str | list, xlabel: str, ylabel: str, title: str
):
    plt.figure(figsize=(20, 10))

    if type(y) == list:
        for y_ in y:
            plt.plot(df[x], df[y_], label=y_)
            plt.plot(df[x], df[y_], label=y_)
    else:
        plt.plot(df[x], df[y], label=y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)


def plot_anomaly_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    anomaly: str,
    xlabel: str,
    ylabel: str,
    title: str,
):
    plt.figure(figsize=(20, 10))

    plt.plot(df[x], df[y], label=y)

    anomaly_points = df[df[anomaly] == 1]

    plt.scatter(x=anomaly_points[x], y=anomaly_points[y], s=10, c="red")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)


def get_versioned_filename(filename: str, save_dir: str, new_file: bool = False) -> str:
    os.makedirs(save_dir, exist_ok=True)

    dir_list = os.listdir(save_dir)

    file_dir_list = list(filter(lambda x: filename in x, dir_list))
    if len(file_dir_list) > 0:
        file_dir_list = sorted(file_dir_list, reverse=True)

        latest_file = file_dir_list[0]
        version_number = int(latest_file.split("_")[-1][1:])
    else:
        version_number = 0

    if new_file:
        return os.path.join(save_dir, filename + "_v" + str(version_number + 1))
    else:
        if version_number == 0:
            raise FileNotFoundError(f"No version exists for file {filename}")
        return os.path.join(save_dir, filename + "_v" + str(version_number))


def plot_forecast(train, actuals, preds, x, y):
    plt.figure(figsize=(20, 5))
    plt.plot(
        train[x],
        train[y],
        color="cornflowerblue",
        label="Train data",
    )
    plt.plot(
        actuals[x],
        preds,
        color="orange",
        label="Predictions",
    )
    plt.plot(
        actuals[x],
        actuals[y],
        color="green",
        label="Actuals",
    )
    plt.grid()
    plt.legend(loc="best")
    plt.title(y)
    plt.show(block=False)


def log_results(log_path, model_filename, model_parameters, train_metrics, val_metrics):

    data = [
        [
            model_filename,
            str(model_parameters),
            *train_metrics.values(),
            *val_metrics.values(),
        ]
    ]
    columns = [
        "model_filename",
        "model_parameters",
        *train_metrics.keys(),
        *val_metrics.keys(),
    ]

    df = pd.DataFrame(data=data, columns=columns)

    write_header = True
    if os.path.exists(log_path):
        write_header = False

    with open(log_path, "a", newline="") as f:
        df.to_csv(f, header=write_header, index=False)
