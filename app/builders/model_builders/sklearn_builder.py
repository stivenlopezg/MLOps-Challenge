from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from keras_tuner.engine.hyperparameters import HyperParameter


def build_sgd_model(hp):
    """
    Construye un modelo de clasificación SGDClassifier.

    Parameters
    ----------
    hp: HyperParameter
        Objeto HyperParameter de Keras Tuner que contiene información sobre
        los hiperparámetros para sintonizar.

    Returns
    -------
    model
        Modelo de clasificación SGDClassifier construido con los hiperparámetros especificados.
    """
    params = {'penalty': hp.Choice(name="penalty_sgd", values=["l2", "l1", "elasticnet"]),
              'alpha': hp.Float(name='alpha_sgd', min_value=1e-2, max_value=10, step=10, sampling='log'),
              'l1_ratio': hp.Float(name="l1_ratio_sgd", min_value=0, max_value=1, step=0.1),
              'validation_fraction': 0.3,
              'max_iter': 10000,
              'early_stopping': True,
              'random_state': 42}

    model = SGDClassifier(**params)

    return model


def build_hist_gbm_model(hp):
    """
    Construye un modelo de clasificación HistGradientBoostingClassifier.

    Parameters
    ----------
    hp: HyperParameter
        Objeto HyperParameter de Keras Tuner que contiene información sobre
        los hiperparámetros para sintonizar.

    Returns
    -------
    model
        Modelo de clasificación HistGradientBoostingClassifier construido con los hiperparámetros especificados.
    """
    params = {'learning_rate': hp.Float(name="learning_rate_hgb", min_value=0.0, max_value=1.0, step=0.1),
              'max_iter': hp.Int(name='max_iter_hgb', min_value=50, max_value=200, step=10),
              'max_depth': hp.Int(name='max_depth_hgb', min_value=3, max_value=10, step=1),
              'early_stopping': True,
              'validation_fraction': 0.3,
              'random_state': 42}
    model = HistGradientBoostingClassifier(**params)

    return model
