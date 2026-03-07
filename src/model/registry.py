from model.model_xgboost import train_xgboost
from model.model_lightgbm import train_lightgbm
from model.model_lstm import train_lstm
from model.model_tcn import train_tcn
from model.model_transformer import train_transformer


MODEL_REGISTRY = {

    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,

    "lstm": train_lstm,
    "tcn": train_tcn,
    "transformer": train_transformer
}