from cnn import CNN
from deconv import Deconv
from fully_connected import FullyConnected


def get_model(model_name, input_size=9395):
    if model_name == "fully_connected":
        return FullyConnected(input_size)
    elif model_name == "cnn":
        return CNN(input_size)
    elif model_name == "deconv":
        return Deconv(input_size)
