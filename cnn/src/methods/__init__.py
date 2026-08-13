from src.methods.base import BaseContinualMethod
from src.methods.normal import NormalMethod
from src.methods.replay import ReplayMethod
from src.methods.anchored_replay import AnchoredReplayMethod
from src.methods.ewc import EWCMethod
from src.methods.lwf import LwFMethod

__all__ = [
    'BaseContinualMethod',
    'NormalMethod',
    'ReplayMethod',
    'AnchoredReplayMethod',
    'EWCMethod',
    'LwFMethod',
]

METHOD_REGISTRY = {
    'normal': NormalMethod,
    'replay': ReplayMethod,
    'anchored_replay': AnchoredReplayMethod,
    'ewc': EWCMethod,
    'lwf': LwFMethod,
}

def get_method(name: str):
    """Get continual learning method class by name."""
    name = name.lower()
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]
