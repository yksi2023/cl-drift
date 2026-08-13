from src.methods import get_method


def sequential_learning(
    model,
    tasks,
    config,
    num_iterations=5000,
    batch_size=1024,
    lr=0.001,
    device="cpu",
    method="normal",
    save_dir="./experiments/rnn_drift",
    train_pool_size=30,
    train_seed=12345,
    early_stop_patience=500,
    early_stop_delta=1e-3,
    memory_per_task=300,
    replay_num_tasks=5,
):
    """Train the model sequentially on cognitive tasks (paper setting)."""
    common_kwargs = {
        "model": model,
        "tasks": tasks,
        "config": config,
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "lr": lr,
        "device": device,
        "save_dir": save_dir,
        "train_pool_size": train_pool_size,
        "train_seed": train_seed,
        "early_stop_patience": early_stop_patience,
        "early_stop_delta": early_stop_delta,
    }

    method_kwargs = {}
    if method.lower() == "replay":
        method_kwargs = {
            "memory_per_task": memory_per_task,
            "replay_num_tasks": replay_num_tasks,
        }

    learner = get_method(method)(**common_kwargs, **method_kwargs)
    return learner.run()
