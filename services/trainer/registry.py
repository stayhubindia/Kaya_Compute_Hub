from typing import Dict, Type
from services.trainer.backends.base import BaseTrainerBackend
from services.trainer.backends.demo import DemoTrainerBackend
from services.trainer.backends.pytorch import PyTorchTrainerBackend

BACKEND_REGISTRY: Dict[str, Type[BaseTrainerBackend]] = {
    "demo": DemoTrainerBackend,
    "pytorch": PyTorchTrainerBackend,
}

def get_trainer_backend(backend_name: str) -> BaseTrainerBackend:
    name = backend_name.lower()
    backend_cls = BACKEND_REGISTRY.get(name)
    if not backend_cls:
        raise ValueError(f"Trainer backend '{backend_name}' is not supported or registered.")
    return backend_cls()
