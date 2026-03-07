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

def evaluate_model(model_name, model, data, device):

    if model_name in ["xgboost", "lightgbm"]:

        preds = model.predict(data["X_val"])

        rmse = np.sqrt(mean_squared_error(data["y_val"], preds))

    else:

        rmse = evaluate_torch_model(
            model,
            data["val_loader"],
            device
        )

    return rmse