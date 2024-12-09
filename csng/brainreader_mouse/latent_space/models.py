from cnn import CNN2D
from deconv import Deconv
from deconv_separate import DeconvSeparate
from fully_connected import FullyConnected
from ridge_regression import RidgeRegression


def get_model(model_name, activation="relu"):
    if model_name == "ridge_regression":
        return RidgeRegression()
    elif model_name == "fully_connected":
        return FullyConnected()
    elif model_name == "cnn":
        return CNN2D()
    elif model_name == "deconv":
        return Deconv(activation)
    elif model_name == "deconv_separate":
        return DeconvSeparate()
