# Derived from SpecForge (https://github.com/sgl-project/SpecForge)
# MIT License

from specforge.inference.target_engine.base import KNOWN_BACKENDS, TargetEngine

__all__ = [
    # Generic (Phase B) surface
    "TargetEngine",
    "KNOWN_BACKENDS",
]

# NOTE: the DFlash engines are not re-exported here; import them from
# specforge.inference.target_engine.dflash_target_model.
