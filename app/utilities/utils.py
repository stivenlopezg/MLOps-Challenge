import mlflow
import cloudpickle
import pandas as pd
from typing import Union
from keras import Sequential
from xgboost import XGBClassifier
from app.config.config import path_root
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


def print_dataset_info(data: pd.DataFrame, column: str):
    """
    Imprime la información del dataset para la columna especificada.

    Parameters
    ----------
    data: pd.DataFrame
        El DataFrame que contiene los datos.
    column: str
        La columna para la cual se mostrará la información.
    """
    values = data[column].value_counts()
    for v, n in zip(values.index, values.values):
        print(f"{v}: {n}")


def save_dataset(dataframe: pd.DataFrame, filepath: str, file_type: str = None, **kwargs):
    """
    Guarda el DataFrame en un archivo especificado.

    Parameters
    ----------
    dataframe: pd.DataFrame
        El DataFrame que se guardará.
    filepath: str
        La ruta donde se guardará el archivo.
    file_type: str, opcional
        El tipo de archivo (por defecto es 'csv').
    **kwargs
        Argumentos adicionales pasados a los métodos de escritura específicos (por ejemplo, `to_csv` o `to_parquet`).

    Returns
    -------
    None
    """
    if file_type is None:
        file_type = 'csv'

    if file_type == 'csv':
        return dataframe.to_csv(filepath, **kwargs)
    elif file_type == 'parquet':
        return dataframe.to_parquet(filepath, **kwargs)
    else:
        raise ValueError(f"File type '{file_type}' not supported.")


def load_dataset(filepath: str, file_type: str = None, **kwargs) -> pd.DataFrame:
    """
    Carga un DataFrame desde un archivo especificado.

    Parameters
    ----------
    filepath: str
        La ruta del archivo a cargar.
    file_type: str, opcional
        El tipo de archivo (por defecto es 'csv').
    **kwargs
        Argumentos adicionales pasados a los métodos de lectura específicos (por ejemplo, `read_csv` o `read_parquet`).

    Returns
    -------
    pd.DataFrame
        El DataFrame cargado desde el archivo.
    """
    if file_type is None:
        file_type = 'csv'

    if file_type == 'csv':
        return pd.read_csv(filepath, **kwargs)
    elif file_type == 'parquet':
        return pd.read_parquet(filepath, **kwargs)
    else:
        raise ValueError(f"File type '{file_type}' not supported.")


def extract_score_by_trial(tuner) -> Union[float, None]:
    """
    Extrae la puntuación del mejor ensayo de un sintonizador.

    Parameters
    ----------
    tuner
        El sintonizador utilizado para la búsqueda.

    Returns
    -------
    Union[float, None]
        La puntuación del mejor ensayo o None si no hay ensayos disponibles.
    """
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    return best_trial.score


def get_best_model(tuner):
    """
    Obtiene el mejor modelo de un sintonizador.

    Parameters
    ----------
    tuner
        El sintonizador utilizado para la búsqueda.

    Returns
    -------
    El mejor modelo.
    """
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model


def save_artifact_locally(obj, local_path: str):
    """
    Guarda un objeto localmente utilizando Cloudpickle.

    Parameters
    ----------
    obj
        El objeto que se guardará.
    local_path: str
        La ruta local donde se guardará el objeto.

    Returns
    -------
    None
    """
    with open(local_path, "wb") as file:
        cloudpickle.dump(obj=obj, file=file)


def load_artifact_locally(local_path: str):
    """
    Carga un objeto desde una ruta local utilizando Cloudpickle.

    Parameters
    ----------
    local_path: str
        La ruta local desde donde se cargará el objeto.

    Returns
    -------
    obj
        El objeto cargado desde la ruta local.
    """
    with open(local_path, "rb") as file:
        artifact = cloudpickle.load(file=file)
    return artifact


def save_experiment(run_name: str, description: str, tuner, model_path: str, mlflow_model_path: str, signature,
                    evaluation, class_names: dict = None, metric: str = None, artifacts_path: str = None):
    """
    Guarda los resultados de un experimento en MLflow.

    Parameters
    ----------
    run_name: str
        Nombre del run en MLflow.
    description: str
        Descripción del run en MLflow.
    tuner
        Sintonizador utilizado para la búsqueda.
    model_path: str
        Ruta donde se guardará el modelo.
    mlflow_model_path: str
        Ruta donde se guardara el modelo en MLflow
    signature
        Firma del modelo para MLflow.
    evaluation
        Resultados de la evaluación del modelo.
    class_names: dict, opcional
        Nombres de las clases para la matriz de confusión.
    metric: str, opcional
        Métrica principal a registrar (por defecto es 'accuracy').
    artifacts_path: str, opcional
        Ruta local de artefactos a loggear en MLflow.

    Returns
    -------
    None
    """
    if metric is None:
        metric = "accuracy"

    model = get_best_model(tuner=tuner)

    if class_names is None:
        cm = evaluation.confusion_matrix()
    else:
        cm = evaluation.confusion_matrix(class_names=class_names)

    with mlflow.start_run(run_name=run_name, description=description):
        if artifacts_path is None:
            pass
        else:
            mlflow.log_artifacts(local_dir=f"{path_root}/{artifacts_path}", artifact_path=artifacts_path)
        mlflow.log_metrics(metrics=evaluation.calculate_metrics())
        mlflow.log_params(params=tuner.get_best_hyperparameters(num_trials=1)[0].values)
        mlflow.log_metric(key=f"{metric} validation", value=extract_score_by_trial(tuner=tuner))
        mlflow.log_table(data=cm, artifact_file="metrics/confusion_matrix.json")
        if isinstance(model, (SGDClassifier, HistGradientBoostingClassifier)):
            mlflow.sklearn.save_model(sk_model=model, path=model_path)
            mlflow.sklearn.log_model(sk_model=model, artifact_path=mlflow_model_path, signature=signature)
        elif isinstance(model, XGBClassifier):
            mlflow.xgboost.save_model(xgb_model=model, path=model_path)
            mlflow.xgboost.log_model(xgb_model=model, artifact_path=mlflow_model_path, signature=signature)
        elif isinstance(model, Sequential):
            mlflow.tensorflow.save_model(model=model, path=model_path)
            mlflow.tensorflow.log_model(model=model, artifact_path=mlflow_model_path, signature=signature)
