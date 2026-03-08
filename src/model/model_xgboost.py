from xgboost import XGBRegressor


def train_xgboost(config, data):

    params = config["models"]["xgboost"]

    model = XGBRegressor(**params)

    model.fit(
        data["ml"]["X_train"],
        data["ml"]["y_train"],
        eval_set=[(data["ml"]["X_val"],data["ml"]["y_val"])],
        verbose=False
    )

    return model