import torch


def predict(model_name, model, data, device="cpu"):

    if model_name in ["xgboost", "lightgbm"]:

        preds = model.predict(data["ml"]["X_test"])

    else:

        X_tensor = torch.tensor(
            data["dl"]["X_test_seq"],
            dtype=torch.float32
        ).to(device)

        model.eval()

        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy().flatten()

    return preds