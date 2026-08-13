import torch


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving.

    If restore_best_weights=True, the model weights are restored to the best epoch.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self._best_state_dict = None

    def step(self, model: torch.nn.Module, val_loss: float) -> bool:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = float(val_loss)
            self.bad_epochs = 0
            if self.restore_best_weights:
                # Deepcopy state dict to avoid in-place mutation side effects
                self._best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.restore_best_weights and self._best_state_dict is not None:
            model.load_state_dict(self._best_state_dict)
