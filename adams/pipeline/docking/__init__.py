"""Search and docking functionality."""

__all__ = ["DockingPipeline"]


def __getattr__(name):
    if name == "DockingPipeline":
        from .docking import DockingPipeline
        return DockingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
