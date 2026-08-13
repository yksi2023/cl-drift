import torch
from typing import Optional, Dict
from src.methods import get_method


def incremental_learning(
    model: torch.nn.Module,
    experiment_dataset,
    epochs: int,
    device: torch.device,
    num_classes: int,
    increment: int,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_config: Optional[Dict] = None,
    batch_size: int = 64,
    method: str = "normal",
    memory_per_class: Optional[int] = None,
    save_dir: str = "experiments/ckpts",
    early_stopping_patience: int = 5,
    channels_last: bool = False,
    ewc_lambda: float = 1000.0,
    ewc_protect: str = "all",
    lwf_lambda: float = 1.0,
    lwf_temperature: float = 2.0,
    anchor_lambda: float = 0.0,
    anchor_layers: str = "layer3,layer4",
    anchor_loss: str = "mse",
):
    """Train the model incrementally on new tasks (paper TIL setting)."""
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    if channels_last:
        try:
            model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    common_kwargs = {
        "model": model,
        "experiment_dataset": experiment_dataset,
        "epochs": epochs,
        "device": device,
        "num_classes": num_classes,
        "increment": increment,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler_config": scheduler_config,
        "batch_size": batch_size,
        "save_dir": save_dir,
        "early_stopping_patience": early_stopping_patience,
    }

    method_lower = method.lower()
    method_kwargs = {}
    if method_lower in ("replay", "anchored_replay"):
        if memory_per_class is None:
            raise ValueError(f"memory_per_class is required for method={method}")
        method_kwargs = {"memory_per_class": memory_per_class}
        if method_lower == "anchored_replay":
            method_kwargs.update({
                "anchor_lambda": anchor_lambda,
                "anchor_layers": anchor_layers,
                "anchor_loss": anchor_loss,
            })
    elif method_lower == "ewc":
        method_kwargs = {
            "ewc_lambda": ewc_lambda,
            "ewc_protect": ewc_protect,
        }
    elif method_lower == "lwf":
        method_kwargs = {
            "lwf_lambda": lwf_lambda,
            "lwf_temperature": lwf_temperature,
        }

    learner = get_method(method)(**common_kwargs, **method_kwargs)
    learner.run()
