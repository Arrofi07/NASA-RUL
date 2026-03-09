import torch


def train_torch_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=20):

    for epoch in range(epochs):

        model.train()

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            preds = model(X).squeeze()

            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print(f"Epoch {epoch}, Loss {loss.item()}")