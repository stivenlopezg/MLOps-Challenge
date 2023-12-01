from keras_tuner import Objective
from sklearn.model_selection import KFold
from app.builders.build_models import build_model
from keras_tuner.oracles import BayesianOptimizationOracle
from keras_tuner.tuners import SklearnTuner, BayesianOptimization
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score

cv = KFold(n_splits=5)


def tune_model(model_type, input_dim: int = None, output_dim: int = None,
               max_length: int = None, objective: str = None, max_trials: int = 5, **kwargs):
    """
    Construye un modelo de acuerdo al tipo especificado.

    Parameters
    ----------
    model_type: str
        Tipo de modelo a construir. Puede ser 'lstm', 'rnn', 'sklearn', o 'xgboost'.
    input_dim: int, opcional
        Entero que define el tamaño del vocabulario del embedding. Requerido para 'lstm' y `rnn'.
    output_dim: int, opcional
        Dimensión del espacio de embeddings. Requerido para 'lstm' y `rnn'.
    max_length: int, opcional
        Longitud máxima de la secuencia. Requerido para 'lstm' y `rnn'.
    objective: str, opcional

    max_trials: int, default = 5

    Returns
    -------
    tuner
        Modelo construido con los hiperparámetros especificados.
    """
    global scoring

    if objective is None:
        objective = "accuracy"

    if objective == "accuracy":
        scoring = make_scorer(score_func=accuracy_score)
    elif objective == "precision":
        scoring = make_scorer(score_func=precision_score)
    elif objective == "recall":
        scoring = make_scorer(score_func=recall_score)

    def build_model_wrapper(hp):
        return build_model(hp, model_type, input_dim, output_dim, max_length)

    if model_type in ["sgd", "hist_gbm", "xgboost"]:
        tuner = SklearnTuner(oracle=BayesianOptimizationOracle(objective=Objective(name='score', direction='max'),
                                                               max_trials=max_trials),
                             hypermodel=build_model_wrapper,
                             scoring=scoring,
                             cv=cv,
                             directory="keras_tuner_logs",
                             project_name=f"{model_type}_tuning", **kwargs)
    else:
        tuner = BayesianOptimization(hypermodel=build_model_wrapper,
                                     max_trials=max_trials,
                                     objective=Objective(name=f"val_{objective}", direction="max"),
                                     directory="keras_tuner_logs",
                                     project_name=f"{model_type}_tuning", **kwargs)

    return tuner
