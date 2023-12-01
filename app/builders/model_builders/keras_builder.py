from keras import Sequential
from keras_tuner import HyperModel
from keras.layers import Dense, Embedding, LSTM, SimpleRNN
from keras_tuner.engine.hyperparameters import HyperParameter


class LstmModel(HyperModel):
    def __init__(self, input_dim: int, output_dim: int, max_length: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length = max_length

    def build(self, hp):
        model = Sequential()
        model.add(layer=Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.max_length))
        model.add(layer=LSTM(units=hp.Int(name="first_input_unit", min_value=64, max_value=128, step=32),
                             activation=hp.Choice(name="first_activation", values=["relu", "selu", "sigmoid"]),
                             dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
        model.add(layer=LSTM(units=hp.Int(name="second_input_unit", min_value=64, max_value=128, step=32),
                             activation=hp.Choice(name="second_activation", values=["relu", "selu", "sigmoid"]),
                             dropout=0.25, recurrent_dropout=0.25, return_sequences=False))
        model.add(layer=Dense(units=3, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model


class RNNModel(HyperModel):
    def __init__(self, input_dim: int, output_dim: int, max_length: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length = max_length

    def build(self, hp):
        model = Sequential()
        model.add(layer=Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.max_length))
        model.add(layer=SimpleRNN(units=hp.Int(name="first_input_unit", min_value=64, max_value=128, step=32),
                                  activation=hp.Choice(name="first_activation", values=["relu", "selu", "sigmoid"]),
                                  dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
        model.add(layer=SimpleRNN(units=hp.Int(name="second_input_unit", min_value=64, max_value=128, step=32),
                                  activation=hp.Choice(name="second_activation", values=["relu", "selu", "sigmoid"]),
                                  dropout=0.25, recurrent_dropout=0.25, return_sequences=False))
        model.add(layer=Dense(units=3, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model
