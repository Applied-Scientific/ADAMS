"""
Docking backends for the unified pipeline.

Backend classes are registered for use by DockingPipeline(backend=...).
"""

from .base_gpu import BaseGPUBackend
from .unidock import UniDockBackend
from .vina import VinaBackend
from .vina_gpu import VinaGPUBackend

BACKEND_REGISTRY = {
    "vina": VinaBackend,
    "vina_gpu": VinaGPUBackend,
    "unidock": UniDockBackend,
}


def get_backend_class(name):
    """Return the backend class for the given name, or None if unknown."""
    return BACKEND_REGISTRY.get(name)


__all__ = [
    "BACKEND_REGISTRY",
    "BaseGPUBackend",
    "UniDockBackend",
    "VinaBackend",
    "VinaGPUBackend",
    "get_backend_class",
]
