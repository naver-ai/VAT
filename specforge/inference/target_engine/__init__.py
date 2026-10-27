# coding=utf-8
# Derived from SpecForge (https://github.com/sgl-project/SpecForge)
# MIT License

"""Target engines: the backend-agnostic capture surface (TargetEngine)."""

from .base import KNOWN_BACKENDS, TargetEngine

__all__ = [
    "TargetEngine",
    "KNOWN_BACKENDS",
]

# NOTE: the DFlash engines are not re-exported here; import them from the
# dflash_target_model submodule.
