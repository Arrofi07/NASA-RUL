from lightgbm import LGBMRegressor

def train_lightgbm(config, data):

    params = config["models"]["lightgbm"]

    model = LGBMRegressor(**params)

    model.fit(
        data["ml"]["X_train"], 
        data["ml"]["y_train"],
        eval_set=[(data["ml"]["X_val"],data["ml"]["y_val"])]
        )

    return model