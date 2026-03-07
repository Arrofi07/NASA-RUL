from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

def tune_xgboost(X_train, X_val, y_train, y_val):
    param_grid = {

        "max_depth":[4,5,6,7,8],
        "min_child_weight":[1,3,5,7],

        "subsample":[0.6,0.8,1],
        "colsample_bytree":[0.6,0.8,1],

        "learning_rate":[0.01,0.03,0.05,0.1],

        "n_estimators":[300,500,700,900],

        "gamma":[0,0.5,1,2]

    }

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    search = RandomizedSearchCV(

        estimator=xgb,
        param_distributions=param_grid,
        n_iter=50,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    search.fit(X_train,y_train)

    best_model = search.best_estimator_
    print(search.best_params_)

    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val,y_val)],
        early_stopping_rounds=50,
        verbose=True
    )

    return best_model