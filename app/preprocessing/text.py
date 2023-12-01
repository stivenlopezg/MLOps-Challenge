import pandas as pd
from keras.preprocessing.sequence import pad_sequences


def prepare_text_data(texts: pd.Series, tokenizer, max_len: int = 30):
    """
    Prepara datos de texto para su procesamiento en modelos de aprendizaje automático.

    Parameters
    ----------
    texts: pd.Series
        Serie que contiene los textos a procesar.
    tokenizer
        Tokenizador utilizado para convertir los textos en secuencias numéricas.
    max_len: int, opcional
        Longitud máxima deseada de las secuencias (por defecto es 30).

    Returns
    -------
    numpy.ndarray
        Datos de texto preparados.
    """
    prep_data = tokenizer.texts_to_sequences(texts=texts)
    prep_data = pad_sequences(sequences=prep_data, maxlen=max_len)
    return prep_data
