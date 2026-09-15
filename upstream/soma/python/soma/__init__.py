"""SOMA's Python front-end.

Spec section 6: in-process Python and ML surrogates are peer components,
subject to the same ``Constrain`` and ``Diagnose`` contracts as physics.
Spec section 3.2: the Python front-end builds the same graph object as C++ and
YAML, and emits the same flat ``soma_graph_v1`` descriptor.

Two entry points, for two different jobs:

``soma.Context``
    The stable C ABI of ``include/soma/soma_c.h``, one method per entry point.
    Use it when your Python code *is* the component and drives its own loop
    with ``Context.begin_step()`` / ``Context.end_step()``, or when you want to
    hand a descriptor to another process.

``soma.Session``
    The in-process runtime. Use it when SOMA should drive: add
    ``PyComponent`` subclasses, call ``Session.initialize()``, then
    ``Session.run()``. This is the path the C ABI deliberately does not offer
    -- ``soma_run_period()`` returns ``SOMA_ERR_UNSUPPORTED`` because the ABI
    has no way to register a callback into a component.

Errors are exceptions, never return codes. Everything derives from
``SomaError``, and each subclass also derives from the closest builtin, so
``except ValueError:`` still catches a malformed argument.

NumPy arrays bound as field storage must be C-contiguous ``float64`` and must
outlive the registry or context that holds them. The binding holds a strong
reference on your behalf, so a temporary expression is safe, but a
non-contiguous slice, a transpose, a wrong dtype or a read-only array is
refused rather than silently copied -- a copy is not the component's storage.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Tuple, Union

from _soma import (  # noqa: F401
    ERR_INVALID,
    ERR_NOTFOUND,
    ERR_RUNTIME,
    ERR_STATE,
    ERR_UNSUPPORTED,
    MEMSPACE_DEVICE,
    MEMSPACE_HOST,
    OK,
    AccumOp,
    Channel,
    ChannelBuilder,
    ChannelId,
    ComponentDecl,
    ComponentId,
    Conservation,
    Context,
    CouplingGraph,
    Diagnostic,
    Diagnostics,
    Discretization,
    ExecutionPlan,
    ExpandMode,
    FieldId,
    FieldRegistry,
    FieldSpec,
    Intent,
    InvalidArgument,
    IterationRegionDecl,
    IterationRegionIdx,
    LoweringResult,
    MemorySpace,
    MeshDecl,
    MeshId,
    NotFound,
    OperatorDAG,
    PyComponent,
    RankSet,
    Rational,
    Region,
    RegionId,
    RemapFlags,
    RunnerConfig,
    RunStats,
    ScheduleResult,
    SerialMeshBackend,
    Session,
    Severity,
    SomaError,
    SomaRuntimeError,
    StateError,
    StepContext,
    TemporalBasis,
    Units,
    Unsupported,
    ValidationError,
    Window,
    abi_version,
    dump_graphviz,
    dump_ir,
    dump_plan,
    from_descriptor,
    from_yaml,
    load_yaml,
    lower,
    schedule,
    structurally_equal,
    units,
    validate,
    version,
)

__all__ = [
    "AccumOp",
    "Channel",
    "ChannelBuilder",
    "ChannelId",
    "ComponentDecl",
    "ComponentId",
    "Conservation",
    "Context",
    "CouplingGraph",
    "Diagnostic",
    "Diagnostics",
    "Discretization",
    "ERR_INVALID",
    "ERR_NOTFOUND",
    "ERR_RUNTIME",
    "ERR_STATE",
    "ERR_UNSUPPORTED",
    "ExecutionPlan",
    "ExpandMode",
    "FieldId",
    "FieldRegistry",
    "FieldSpec",
    "Intent",
    "IntervalLike",
    "InvalidArgument",
    "IterationRegionDecl",
    "IterationRegionIdx",
    "LoweringResult",
    "MEMSPACE_DEVICE",
    "MEMSPACE_HOST",
    "MemorySpace",
    "MeshDecl",
    "MeshId",
    "NotFound",
    "OK",
    "OperatorDAG",
    "PyComponent",
    "RankSet",
    "Rational",
    "Region",
    "RegionId",
    "RemapFlags",
    "RunStats",
    "RunnerConfig",
    "ScheduleResult",
    "SerialMeshBackend",
    "Session",
    "Severity",
    "SomaError",
    "SomaRuntimeError",
    "StateError",
    "StepContext",
    "TemporalBasis",
    "Units",
    "Unsupported",
    "ValidationError",
    "Window",
    "abi_version",
    "build_plan",
    "dump_graphviz",
    "dump_ir",
    "dump_plan",
    "from_descriptor",
    "from_yaml",
    "load_yaml",
    "lower",
    "rational",
    "schedule",
    "structurally_equal",
    "units",
    "validate",
    "version",
]

#: Anything ``rational()`` and the ``interval=`` keyword accept.
IntervalLike = Union[Rational, Fraction, int, str]


def rational(value: IntervalLike) -> Rational:
    """Coerce ``value`` to an exact ``Rational``.

    Accepts a ``Rational``, an ``int``, a ``"1/3"`` string, or a
    ``fractions.Fraction``.

    A ``float`` is deliberately rejected. Spec section 4.3 requires exact
    coupling times: 1/3 and 1/7 subcycling must not drift over 10**6 steps, and
    ``0.1`` is not one tenth. Say ``Fraction(1, 10)`` or ``"1/10"``.
    """
    return Rational.coerce(value)


def _fail_validation(what, diags):
    err = ValidationError("{}:\n{}".format(what, diags.to_string()))
    err.code = None
    err.diagnostics = diags
    raise err


def build_plan(
    graph: CouplingGraph,
) -> Tuple[OperatorDAG, ExecutionPlan, SerialMeshBackend]:
    """Validate, lower and schedule ``graph`` on the serial mesh backend.

    Returns ``(dag, plan, backend)``. Raises ``ValidationError`` carrying the
    whole diagnostic list in ``.diagnostics`` if the graph does not validate,
    lower or schedule -- every problem at once, per the spec's Global
    Constraints, not just the first.

    The backend is returned, not discarded: it owns the meshes the plan's
    region sizes were derived from, and dropping it would leave the caller
    unable to build a second plan against the same sizes.
    """
    diags = validate(graph)
    if diags.has_errors:
        _fail_validation("build_plan: the graph does not validate", diags)

    backend = SerialMeshBackend()
    backend.load_graph_meshes(graph)

    lowered = lower(graph)
    if not lowered.ok:
        _fail_validation(
            "build_plan: the graph does not lower", lowered.diagnostics
        )

    scheduled = schedule(graph, lowered.dag, backend)
    if not scheduled.ok:
        _fail_validation(
            "build_plan: the graph does not schedule", scheduled.diagnostics
        )
    return lowered.dag, scheduled.plan, backend
