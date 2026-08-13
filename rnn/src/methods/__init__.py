from src.methods.normal import NormalMethod
from src.methods.replay import ReplayMethod

_METHOD_MAP = {
    "normal": NormalMethod,
    "replay": ReplayMethod,
}


def get_method(name: str):
    """Factory function to get a continual learning method class by name."""
    key = name.lower()
    if key not in _METHOD_MAP:
        raise ValueError(f"Unknown method '{name}'. Available: {list(_METHOD_MAP.keys())}")
    return _METHOD_MAP[key]
