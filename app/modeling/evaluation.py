import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


class ClassificationEvaluator(object):
    """
    Clase para calcular las principales métricas de un problema de clasificación.
    """

    def __init__(self, observed: pd.Series or list, predicted: pd.Series or list):
        """
        Inicializa la instancia de la clase.

        Parameters
        ----------
        observed: pd.Series or list
            La serie o lista que contiene las etiquetas observadas.
        predicted: pd.Series or list
            La serie o lista que contiene las etiquetas predichas.
        """
        self.observed = observed
        self.predicted = predicted
        self.metrics = None

    def generate_report(self):
        """
        Genera un DataFrame con las métricas más usadas de clasificación.

        Returns
        -------
        pd.DataFrame
            El DataFrame con el reporte de clasificación.
        """
        report = np.round(pd.DataFrame(classification_report(y_true=self.observed,
                                                             y_pred=self.predicted, output_dict=True)), 2).T
        return report

    def confusion_matrix(self, class_names: dict = None, **kwargs):
        """
        Devuelve la matriz de confusión como un DataFrame.

        Parameters
        ----------
        class_names: dict, opcional
            Nombres de las clases para la matriz de confusión.

        Returns
        -------
        pd.DataFrame
            La matriz de confusión.
        """
        table = np.round(pd.crosstab(index=self.observed, columns=self.predicted,
                                     rownames=['Observed'], colnames=['Predicted'], **kwargs), 2)
        if class_names is None:
            return table
        else:
            mapper = class_names
            table = table.rename(columns=mapper, index=mapper)
            return table

    def calculate_metrics(self):
        """
        Calcula las métricas más usadas.

        Returns
        -------
        dict
            Un diccionario con las métricas calculadas.
        """
        metrics = {'precision': np.round(precision_score(self.observed,
                                                         self.predicted,
                                                         average="macro"),
                                         decimals=4),
                   'recall': np.round(recall_score(self.observed,
                                                   self.predicted,
                                                   average="macro"),
                                      decimals=4),
                   'f1': np.round(f1_score(self.observed,
                                           self.predicted,
                                           average="macro"),
                                  decimals=4)}
        self.metrics = metrics
        return metrics

    def print_metrics(self):
        """
        Imprime un resumen de las métricas calculadas.

        Returns
        -------
        string
            El resumen de las métricas calculadas.
        """
        global metrics
        if self.metrics is None:
            metrics = self.calculate_metrics()
        for m, v in metrics.items():
            print(f"{m}: {v}")
