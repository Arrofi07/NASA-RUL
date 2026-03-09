from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

def tune_lightgbm(X_train, X_val, y_train, y_val):
    param_grid = {

        "num_leaves":[31,63,127,255],
        "max_depth":[4,6,8,10],

        "learning_rate":[0.01,0.03,0.05,0.1],

        "n_estimators":[300,500,800,1200],

        "min_child_samples":[10,20,40,80],

        "subsample":[0.6,0.8,1.0],
        "colsample_bytree":[0.6,0.8,1.0],

        "reg_alpha":[0,0.1,0.5,1],
        "reg_lambda":[0,0.1,0.5,1]

    }

    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        random_state=42
    )

    search = RandomizedSearchCV(

        estimator=lgb_model,
        param_distributions=param_grid,
        n_iter=60,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    search.fit(X_train,y_train)

    best_lgb = search.best_estimator_
    print(search.best_params_)

    best_lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val,y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=100
    )

    return best_lgb