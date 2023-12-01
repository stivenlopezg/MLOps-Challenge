from app.builders.model_builders.keras_builder import LstmModel, RNNModel
from app.builders.model_builders.xgboost_builder import build_xgboost_model
from app.builders.model_builders.sklearn_builder import build_sgd_model, build_hist_gbm_model


def build_model(hp, model_type, input_dim: int = None, output_dim: int = None, max_length: int = None):
    """
    Construye un modelo de acuerdo al tipo especificado.

    Parameters
    ----------
    hp: HyperParameter
        Objeto HyperParameter de Keras Tuner que contiene información sobre
        los hiperparámetros para sintonizar.
    model_type: str
        Tipo de modelo a construir. Puede ser 'lstm', 'rnn', 'sgd', "hist_gbm", y 'xgboost'.
    input_dim: int, opcional
        Entero que define el tamaño del vocabulario del embedding. Requerido para 'lstm' y `rnn'.
    output_dim: int, opcional
        Dimensión del espacio de embeddings. Requerido para 'lstm' y `rnn'.
    max_length: int, opcional
        Longitud máxima de la secuencia. Requerido para 'lstm' y `rnn'.

    Returns
    -------
    model
        Modelo construido con los hiperparámetros especificados.
    """
    if model_type == 'lstm':
        if input_dim is None or output_dim is None or max_length is None:
            raise ValueError("Para LSTM, input_dim, output_dim y max_length son obligatorios.")
        return LstmModel(input_dim=input_dim, output_dim=output_dim, max_length=max_length).build(hp)

    elif model_type == 'rnn':
        if input_dim is None or output_dim is None or max_length is None:
            raise ValueError("Para RNN, input_dim, output_dim y max_length son obligatorios.")
        return RNNModel(input_dim=input_dim, output_dim=output_dim, max_length=max_length).build(hp)

    elif model_type == 'sgd':
        return build_sgd_model(hp)

    elif model_type == "hist_gbm":
        return build_hist_gbm_model(hp)

    elif model_type == 'xgboost':
        return build_xgboost_model(hp)

    else:
        raise ValueError(f"Model type '{model_type}' not supported.")
