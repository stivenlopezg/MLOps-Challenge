import os
import time
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from app.config.config import logger, path_root
from app.preprocessing.text import prepare_text_data
from app.utilities.utils import load_dataset, load_artifact_locally, save_dataset


def execute_inference(tracking_uri: str, experiment_name: str):
    logger.info("El proceso de inferencia ha comenzado correctamente.")
    client = MlflowClient()
    logger.info("Se ha cargado el cliente de MlFlow")
    mlflow.set_tracking_uri(uri=tracking_uri)

    text_inference = load_dataset(filepath="app/data/inference.csv")["item"]
    logger.info(f"Se ha cargado el dataset correctamente. El dataset tiene {text_inference.shape[0]} observaciones.")

    runs = mlflow.search_runs(experiment_names=[experiment_name],
                              order_by=["metrics.f1 DESC"])
    logger.info("Ha empezado el proceso de descarga de artefactos.")
    local_dir = "inference/"
    full_path = os.path.join(path_root, local_dir)
    os.makedirs(name=full_path, exist_ok=True)

    metrics_path = "models/"
    run_id = runs["run_id"].iloc[0]
    local_path = client.download_artifacts(run_id=run_id, path=metrics_path, dst_path=full_path)
    logger.info(f"Se han descargado los artefactos en la carpeta {full_path}.")

    tokenizer = load_artifact_locally(local_path=f"{local_path}/preprocessing/tokenizer.pkl")
    encoder = load_artifact_locally(local_path=f"{local_path}/preprocessing/label_encoder.pkl")
    logger.info("Se ha cargado el tokenizer para hacer el preprocesamiento de texto.")

    logger.info("Ha iniciado el proceso de preprocesamiento de texto.")
    prep_data = prepare_text_data(texts=text_inference, tokenizer=tokenizer)
    logger.info(f"Ha finalizado correctamente el preprocesamiento de texto. La dimensión de la matriz es {prep_data.shape[0]}x{prep_data.shape[1]}")

    model_type = runs["tags.mlflow.runName"].iloc[0].split("-")[0]
    logger.info(f"Se ha instanciado el modelo {model_type.upper()}.")
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/models/{model_type}")

    logger.info("Ha comenzado el proceso de predicción ...")
    start_time = time.time()
    predictions = pd.DataFrame(data={"prediction": model.predict(prep_data)})
    predictions["prediction"] = predictions["prediction"].map({i: v for i, v in enumerate(encoder.classes_)})
    save_dataset(dataframe=predictions, filepath="app/data/predictions.csv", index=False)
    logger.info(f"El proceso de predicción ha finalizado correctamente. Se ha demorado {time.time() - start_time} sgs.")
    return None
