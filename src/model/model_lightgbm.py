from lightgbm import LGBMRegressor

def train_lightgbm(X_train, y_train):

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1
    )

    model.fit(X_train, y_train)

    return model