"""Algorithm-neutral DSSAT infrastructure for AWM irrigation experiments."""

from .backend import DSSATWorkerBackend, DSSATWorkerPaths
from .management import DSSATExperimentRenderer, IRRIGATION_MARKER, IrrigationEvent
from .output_reader import CachedDSSATOutputReader
from .runner import DSSATRunner
from .runtime_assets import (
    CUSTOM_DSSAT_BASE_VERSION,
    CUSTOM_DSSAT_BUILD_LABEL,
    CUSTOM_DSSAT_EXECUTABLE,
    prepare_project_worker,
    prepare_worker_from_template,
    render_dssatpro,
    validate_versioned_template,
)
from .runtime_paths import (
    DEFAULT_AWM_RUNTIME_BASE,
    RUNTIME_NAMESPACE_HEX_LENGTH,
    WorkspaceRootLock,
    archive_root_for_project,
    ensure_runtime_roots,
    evaluation_root_for_project,
    register_dssat_runtime,
    runtime_namespace_for_project,
    runtime_root_for_project,
    worker_root_for_project,
    worker_workspace_for_project,
)
from .workspace import managed_workspace_name, validate_dssatpro_record_width

__all__ = [
    "CUSTOM_DSSAT_BASE_VERSION",
    "CUSTOM_DSSAT_BUILD_LABEL",
    "CUSTOM_DSSAT_EXECUTABLE",
    "CachedDSSATOutputReader",
    "DEFAULT_AWM_RUNTIME_BASE",
    "DSSATExperimentRenderer",
    "DSSATRunner",
    "DSSATWorkerBackend",
    "DSSATWorkerPaths",
    "IRRIGATION_MARKER",
    "IrrigationEvent",
    "RUNTIME_NAMESPACE_HEX_LENGTH",
    "WorkspaceRootLock",
    "archive_root_for_project",
    "ensure_runtime_roots",
    "evaluation_root_for_project",
    "managed_workspace_name",
    "prepare_project_worker",
    "prepare_worker_from_template",
    "register_dssat_runtime",
    "render_dssatpro",
    "runtime_namespace_for_project",
    "runtime_root_for_project",
    "validate_dssatpro_record_width",
    "validate_versioned_template",
    "worker_root_for_project",
    "worker_workspace_for_project",
]
