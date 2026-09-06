"""Algorithm-neutral DSSAT infrastructure for AWM irrigation experiments."""

from .backend import DSSATWorkerBackend, DSSATWorkerPaths
from .management import DSSATExperimentRenderer, IRRIGATION_MARKER, IrrigationEvent
from .output_reader import CachedDSSATOutputReader
from .runner import DSSATRunner
from .runtime_assets import (
    CUSTOM_DSSAT_BASE_VERSION,
    CUSTOM_DSSAT_BUILD_LABEL,
    CUSTOM_DSSAT_EXECUTABLE,
    prepare_worker_from_template,
    render_dssatpro,
    validate_versioned_template,
)
from .workspace import managed_workspace_name, validate_dssatpro_record_width

__all__ = [
    "CUSTOM_DSSAT_BASE_VERSION",
    "CUSTOM_DSSAT_BUILD_LABEL",
    "CUSTOM_DSSAT_EXECUTABLE",
    "CachedDSSATOutputReader",
    "DSSATExperimentRenderer",
    "DSSATRunner",
    "DSSATWorkerBackend",
    "DSSATWorkerPaths",
    "IRRIGATION_MARKER",
    "IrrigationEvent",
    "managed_workspace_name",
    "prepare_worker_from_template",
    "render_dssatpro",
    "validate_dssatpro_record_width",
    "validate_versioned_template",
]
