from sklearn.metrics import mean_squared_error
import numpy as np
import torch


def evaluate_torch_model(model, loader, device):

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)

            pred = model(X).cpu().numpy()

            preds.extend(pred)
            targets.extend(y.numpy())

    rmse = np.sqrt(mean_squared_error(targets, preds))

    return rmse

def evaluate_model(name, model, data, device):

    if name in ["xgboost", "lightgbm"]:

        preds = model.predict(data["ml"]["X_val"])

        rmse = np.sqrt(mean_squared_error(data["ml"]["y_val"], preds))

    else:

        model.eval()

        preds = []
        targets = []

        with torch.no_grad():

            for X, y in data["dl"]["val_loader"]:

                X = X.to(device)

                pred = model(X).cpu().numpy()

                preds.extend(pred)
                targets.extend(y.numpy())

        rmse = np.sqrt(mean_squared_error(targets, preds))

    return rmse