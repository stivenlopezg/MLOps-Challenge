import os
import time
import mlflow
import warnings
import numpy as np
from keras.utils import to_categorical
from app.modeling.tuning import tune_model
from keras.callbacks import EarlyStopping
from mlflow.models import infer_signature
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from urllib3.exceptions import NotOpenSSLWarning
from app.preprocessing.text import prepare_text_data
from app.modeling.evaluation import ClassificationEvaluator
from app.config.config import logger, label_col, text_col, path_root
from app.utilities.utils import load_dataset, save_artifact_locally, save_experiment, get_best_model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def execute_train(tracking_uri: str, experiment_name: str):
    logger.info("El proceso de entrenamiento ha comenzado correctamente.")
    start_execution = time.time()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)
    logger.info("Se ha definido la uri de tracking en MlFlow.")

    logger.info("Ha comenzado la importación del set de datos.")
    train_data = load_dataset(filepath=f"app/data/train.csv")
    logger.info(f"El set de train ha cargado correctamente. Este tiene {train_data.shape[0]} items.")
    validation_data = load_dataset(filepath=f"app/data/validation.csv")
    logger.info(f"El set de validation ha cargado correctamente. Este tiene {validation_data.shape[0]} items.")
    test_data = load_dataset(filepath=f"app/data/test.csv")
    logger.info(f"El set de test ha cargado correctamente. Este tiene {validation_data.shape[0]} items.")

    logger.info("Se selecciona solo la columna objetivo.")
    train_label = train_data.pop(label_col)
    validation_label = validation_data.pop(label_col)
    test_label = test_data.pop(label_col)

    logger.info("Se selecciona la columna que contiene el texto de los items.")
    texts_train, texts_validation, texts_test = (train_data[text_col].copy(),
                                                 validation_data[text_col].copy(),
                                                 test_data[text_col].copy())

    logger.info("Se instancia el tokenizer para tokenizar y normalizar el texto.")
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts=texts_train)

    logger.info("Se codifica la variable objetivo para que quede de manera numerica.")
    encoder = LabelEncoder()
    train_label = encoder.fit_transform(train_label)
    test_label = encoder.transform(test_label)
    validation_label = encoder.transform(validation_label)

    logger.info("Empieza la creación del directorio para guardar los modelos y los objetos para el preprocesamiento.")
    artifacts_path = f"models/preprocessing"
    full_path = os.path.join(path_root, artifacts_path)
    os.makedirs(name=full_path, exist_ok=True)

    logger.info("Ha comenzado el proceso para guardar los objetos de preprocesamiento de texto.")
    save_artifact_locally(obj=tokenizer, local_path=f"{full_path}/tokenizer.pkl")
    save_artifact_locally(obj=encoder, local_path=f"{full_path}/label_encoder.pkl")
    logger.info("Se ha guardado correctamente los objetos para el preprocesamiento de texto.")

    logger.info("Iniciamos el preprocesamiento de texto sobre los sets de train, validation y test.")
    texts_train = prepare_text_data(texts=texts_train, tokenizer=tokenizer, max_len=30)
    texts_validation = prepare_text_data(texts=texts_validation, tokenizer=tokenizer, max_len=30)
    texts_test = prepare_text_data(texts=texts_test, tokenizer=tokenizer, max_len=30)
    logger.info("Ha finalizado correctamente el preprocesamiento de texto. Ahora podemos entrenar los algoritmos.")

    class_names = {i: v for i, v in enumerate(encoder.classes_)}

    logger.info("Empezamos el proceso de entrenamiento.")
    start_time = time.time()
    logger.info("Empieza el entrenamiento y optimización de hiperparametros del algoritmo SGD.")
    sgd_tuner = tune_model(model_type="sgd")
    sgd_tuner.search(texts_train, train_label)
    logger.info(f"Ha finalizado el entrenamiento del algoritmo SGD. Ha tardado {time.time() - start_time} sgs.")
    model = get_best_model(tuner=sgd_tuner)
    logger.info("Empieza el proceso de guardar las métricas, parámetros y modelo en MlFlow.")
    start_time = time.time()
    save_experiment(run_name="sgd-model",
                    description="Entrenamiento SGD para clasificación de dominio.",
                    tuner=sgd_tuner, model_path=f"{path_root}/models/sgd", mlflow_model_path="models/sgd",
                    signature=infer_signature(model_input=texts_test, model_output=model.predict(texts_test)),
                    evaluation=ClassificationEvaluator(observed=test_label, predicted=model.predict(texts_test)),
                    class_names=class_names, artifacts_path=artifacts_path)
    logger.info(
        f"Se han guardado las métricas, parámetros y modelo localmente y en el server. El proceso duró {time.time() - start_time}")

    logger.info("Empieza el entrenamiento y optimización de hiperparametros del algoritmo HistGradientBoosting.")
    start_time = time.time()
    hgbm_tuner = tune_model(model_type="hist_gbm")
    hgbm_tuner.search(texts_train, train_label)
    logger.info(
        f"Ha finalizado el entrenamiento del algoritmo HistGradientBoosting. Ha tardado {time.time() - start_time} sgs.")
    model = get_best_model(tuner=hgbm_tuner)
    logger.info("Empieza el proceso de guardar las métricas, parámetros y modelo en MlFlow.")
    save_experiment(run_name="hgbm-model",
                    description="Entrenamiento HistGBM para clasificación de dominio.",
                    tuner=sgd_tuner, model_path=f"{path_root}/models/hgbm", mlflow_model_path="models/hgbm",
                    signature=infer_signature(model_input=texts_test, model_output=model.predict(texts_test)),
                    evaluation=ClassificationEvaluator(observed=test_label, predicted=model.predict(texts_test)),
                    class_names=class_names, artifacts_path=artifacts_path)
    logger.info(
        f"Se han guardado las métricas, parámetros y modelo localmente y en el server. El proceso duró {time.time() - start_time}")

    logger.info("Empieza el entrenamiento y optimización de hiperparametros del algoritmo XGBoosting.")
    start_time = time.time()
    xgb_tuner = tune_model(model_type="xgboost")
    xgb_tuner.search(texts_train, train_label)
    logger.info(
        f"Ha finalizado el entrenamiento del algoritmo XGBoost. Ha tardado {time.time() - start_time} sgs.")
    model = get_best_model(tuner=xgb_tuner)
    logger.info("Empieza el proceso de guardar las métricas, parámetros y modelo en MlFlow.")
    save_experiment(run_name="xgb-model",
                    description="Entrenamiento XGBoost para clasificación de dominio.",
                    tuner=sgd_tuner, model_path=f"{path_root}/models/xgb", mlflow_model_path="models/xgb",
                    signature=infer_signature(model_input=texts_test, model_output=model.predict(texts_test)),
                    evaluation=ClassificationEvaluator(observed=test_label, predicted=model.predict(texts_test)),
                    class_names=class_names, artifacts_path=artifacts_path)
    logger.info(
        f"Se han guardado las métricas, parámetros y modelo localmente y en el server. El proceso duró {time.time() - start_time}")

    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 30
    embedding_dim = 50
    logger.info("Empieza el entrenamiento y optimización de hiperparametros de la arquitectura RNN.")
    start_time = time.time()
    rnn_tuner = tune_model(model_type="rnn", input_dim=vocab_size, output_dim=embedding_dim, max_length=max_length)
    rnn_tuner.search(texts_train, to_categorical(train_label),
                     epochs=100, batch_size=256, validation_data=(texts_validation, to_categorical(validation_label)),
                     callbacks=[early_stopping])
    logger.info(
        f"Ha finalizado el entrenamiento de la arquitectura RNN. Ha tardado {time.time() - start_time} sgs.")
    model = get_best_model(tuner=rnn_tuner)
    logger.info("Empieza el proceso de guardar las métricas, parámetros y modelo en MlFlow.")
    save_experiment(run_name="rnn-model",
                    description="Entrenamiento Simple RNN para clasificación de dominio.",
                    tuner=sgd_tuner, model_path=f"{path_root}/models/rnn", mlflow_model_path="models/rnn",
                    signature=infer_signature(model_input=texts_test, model_output=model.predict(texts_test)),
                    evaluation=ClassificationEvaluator(observed=test_label, predicted=np.argmax(model.predict(texts_test), axis=1)),
                    class_names=class_names, artifacts_path=artifacts_path)
    logger.info(
        f"Se han guardado las métricas, parámetros y modelo localmente y en el server. El proceso duró {time.time() - start_time}")

    logger.info("Empieza el entrenamiento y optimización de hiperparametros de la arquitectura LSTM.")
    start_time = time.time()
    lstm_tuner = tune_model(model_type="lstm", input_dim=vocab_size, output_dim=embedding_dim, max_length=max_length)
    lstm_tuner.search(texts_train, to_categorical(train_label), epochs=100,
                      batch_size=256, validation_data=(texts_validation, to_categorical(validation_label)),
                      callbacks=[early_stopping])
    logger.info(
        f"Ha finalizado el entrenamiento de la arquitectura LSTM. Ha tardado {time.time() - start_time} sgs.")
    model = get_best_model(tuner=lstm_tuner)
    logger.info("Empieza el proceso de guardar las métricas, parámetros y modelo en MlFlow.")
    save_experiment(run_name="lstm-model",
                    description="Entrenamiento LSTM para clasificación de dominio.",
                    tuner=sgd_tuner, model_path=f"{path_root}/models/lstm", mlflow_model_path="models/lstm",
                    signature=infer_signature(model_input=texts_test, model_output=model.predict(texts_test)),
                    evaluation=ClassificationEvaluator(observed=test_label, predicted=np.argmax(model.predict(texts_test), axis=1)),
                    class_names=class_names, artifacts_path=artifacts_path)
    logger.info(
        f"Se han guardado las métricas, parámetros y modelo localmente y en el server. El proceso duró {time.time() - start_time}")
    logger.info(
        f"Ha finalizado correctamente el proceso de entrenamiento. Ha durado {time.time() - start_execution} sgs.")
    return None
