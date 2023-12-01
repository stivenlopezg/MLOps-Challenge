from xgboost import XGBClassifier
from keras_tuner.engine.hyperparameters import HyperParameter


def build_xgboost_model(hp):
    """
    Construye un modelo de clasificación xgboost.

    Parameters
    ----------
    hp: HyperParameter
        Objeto HyperParameter de Keras Tuner que contiene información sobre
        los hiperparámetros para sintonizar.

    Returns
    -------
    model
        Modelo de clasificación xgboost construido con los hiperparámetros especificados.
    """
    params = {'learning_rate': hp.Float('learning_rate', min_value=0.01, max_value=0.3, sampling='log'),
              'max_depth': hp.Int('max_depth', min_value=3, max_value=10, step=1),
              'min_child_weight': hp.Float('min_child_weight', min_value=1.0, max_value=10.0, step=1),
              'subsample': hp.Float('subsample', min_value=0.5, max_value=1.0, step=0.1),
              'gamma': hp.Float('gamma', min_value=0.0, max_value=1.0, step=0.1),
              'colsample_bytree': hp.Float('colsample_bytree', min_value=0.5, max_value=1.0, step=0.1),
              'tree_method': 'hist',
              'seed': 42}

    model = XGBClassifier(**params)
    return model
