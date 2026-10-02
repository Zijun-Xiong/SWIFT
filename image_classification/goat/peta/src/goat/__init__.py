"""
MoLE Initialization
"""
from .config import GOATConfig
from .layer import GOATLayer, LinearGOATLayer
from .model import GOATModel
from .io import load_swift_adapter, register_swift_with_peft, save_swift_adapter

__all__ = [
    "GOATConfig",
    "GOATLayer",
    "LinearGOATLayer",
    "GOATModel",
    "load_swift_adapter",
    "register_swift_with_peft",
    "save_swift_adapter",
]


def __getattr__(name):
    raise AttributeError(f"Module {__name__} has no attribute {name}.")
