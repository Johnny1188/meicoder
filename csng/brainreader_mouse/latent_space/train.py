import torch


class Trainer:
    def __init__(
        self, data, model, optimizer, criterion, scheduler, device
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data = data
        self.valid_loader = data.valid_data()
        self.device = device
        self.scheduler = scheduler

    def validation(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.valid_loader):
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device
                )
                preds = self.model(inputs)
                loss = self.criterion(preds, targets)
                total_loss += loss.item()
        return total_loss / len(self.valid_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loader = self.data.train_data()
            self.model.train()
            train_loss = 0
            for _, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device
                )
                # Forward pass
                self.optimizer.zero_grad()

                predictions = self.model(inputs)
                mse_loss = self.criterion(predictions, targets)

                # Total loss
                loss = mse_loss

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            valid_loss = self.validation()
            print(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss: .4f}"
            )
            if self.scheduler is not None:
                self.scheduler.step(valid_loss)
