from src.model.model_lstm import LSTMModel
import torch
from src.model.train import train_torch_model

import mlflow
import mlflow.pytorch

experiment = mlflow.get_experiment_by_name("RUL_LSTM_Tuning")

def tune_lstm(config, data):

    device = config["training"]["device"]
    params = config["models"]["lstm"]

    with mlflow.start_run(experiment_id=experiment.experiment_id):

        # log parameters
        mlflow.log_params({
            "hidden_size": params["hidden_size"],
            "num_layers": params["num_layers"],
            "dropout": params["dropout"],
            "lr": config["training"]["lr"],
            "epochs": config["training"]["epochs"]
        })

        model = LSTMModel(
            input_size=config["data"]["input_size"],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"]
        )

        criterion = torch.nn.MSELoss()

        val_rmse = train_torch_model(
            model,
            data["dl"]["train_loader"],
            data["dl"]["val_loader"],
            optimizer,
            criterion,
            device,
            epochs=config["training"]["epochs"]
        )

        # log metric
        mlflow.log_metric("val_rmse", val_rmse)

        # log model
        mlflow.pytorch.log_model(model, "model")

    return model, val_rmse

import random
import copy

def random_search_lstm(config, data, n_trials=20):

    search_space = {
        "hidden_size": [64, 128],
        "num_layers": [2, 3],
        "dropout": [0.2, 0.3],
        "lr": [1e-3, 5e-4]
    }

    best_rmse = float("inf")
    best_model = None
    best_params = None

    for trial in range(n_trials):

        print(f"\nTrial {trial+1}/{n_trials}")

        hidden_size = random.choice(search_space["hidden_size"])
        num_layers = random.choice(search_space["num_layers"])
        dropout = random.choice(search_space["dropout"])
        lr = random.choice(search_space["lr"])

        trial_config = copy.deepcopy(config)

        trial_config["models"]["lstm"]["hidden_size"] = hidden_size
        trial_config["models"]["lstm"]["num_layers"] = num_layers
        trial_config["models"]["lstm"]["dropout"] = dropout
        trial_config["training"]["lr"] = lr

        model, val_rmse = tune_lstm(trial_config, data)

        print("Validation RMSE:", val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = model
            best_params = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "lr": lr
            }

    print("\nBest RMSE:", best_rmse)
    print("Best Params:", best_params)

    return best_model, best_params