from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_xgboost(X_train, X_val, y_train, y_val):
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val,y_val)],
        verbose=True
    )

    return xgb_model