from src.model.model_xgboost import train_xgboost
from src.model.model_lightgbm import train_lightgbm
from src.model.model_lstm import train_lstm
from src.model.model_tcn import train_tcn
from src.model.model_transformer import train_transformer


MODEL_REGISTRY = {

    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,

    "lstm": train_lstm,
    "tcn": train_tcn,
    "transformer": train_transformer
}