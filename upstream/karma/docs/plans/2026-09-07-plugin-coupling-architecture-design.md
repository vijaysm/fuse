# Plugin Coupling Architecture Design

**Status:** Architecture contract under revision. The fixed-point and static-multiscale path is approved for staged implementation planning; monolithic Newton, dynamic graph adaptation, and external-framework adapters remain deferred capabilities.

## Goals

KARMA coupling must support interchangeable prognostic, data-backed, and emulated models without changing downstream field semantics or coupling topology. Components may execute as statically linked C++, dynamically loaded native plugins, embedded managed runtimes, or out-of-process services. The same graph must support black-box explicit coupling and heterogeneous implicit coupling across different meshes, bases, ownership layouts, and model implementations.

The architecture separates immutable semantic specifications, runtime field bindings, component execution contracts, graph execution, and plugin transports. PETSc is the default nonlinear/algebraic backend but is not embedded in semantic field or component identity.

## Motivation And Build Versus Adopt

The design target is not merely another partitioned coupler. KARMA requires one language-neutral model and port contract that can describe prognostic solvers, data-backed models, learned emulators, and multiscale closures while preserving two distinct execution paths:

1. Black-box explicit or iterated-partitioned coupling, where each component owns its time integration and internal state.
2. Intrusive distributed residual and derivative composition, where capable components participate in one monolithic nonlinear or algebraic solve.

The same semantic graph must remain independent of coupling algorithm, spatial-transfer implementation, linear algebra and mesh infrastructure, programming language, memory location, and in-process versus out-of-process deployment. No evaluated framework supplies that conjunction as one native contract. The architectural claim is the conjunction and its demonstrated implementation; it is not that any individual mechanism is novel.

Existing systems remain important prior art and potential integration targets:

- **preCICE** provides mature distributed participant coupling, mesh/data exchange, mappings, fixed-point and quasi-Newton acceleration, and multilingual participant APIs. Its participant contract is intentionally black-box: it does not expose a shared distributed unknown vector, residual, Jacobian blocks, or physics-aware preconditioner across participants. Cross-participant monolithic Newton would require a different intrusive algebraic contract rather than another participant adapter.
- **ESMF/NUOPC** provides mature gridded-field, virtual-machine, component, connector, mediator, and run-sequence abstractions, particularly for operational Earth-system composition. Its standard connector/mediator contract does not expose shared nonlinear unknown ownership or component residual and Jacobian blocks. Arbitrary physics can be placed inside a component, but a monolithic solve spanning components requires a nonstandard super-component that absorbs their algebra.
- **MCT and OASIS** provide proven distributed field exchange and remapping for independently executing models. Their field-exchange contracts do not define intrusive residual/Jacobian composition.
- **FMI 3.0 and SSP** provide the strongest broadly adopted portable black-box model packaging and composition conventions. FMI arrays, clocks, scheduled execution, and directional or adjoint derivatives do not define partitioned mesh ownership, ghost regions, communicators, collective failure semantics, or distributed sparse operator assembly. An MPI-aware FMU may define private extensions, but they are outside portable FMI/SSP composition.
- **MUSCLE3** provides strong heterogeneous and multiscale orchestration through message-connected components, including dynamic execution patterns. Message ports do not define shared DoF ownership or monolithic operator blocks.
- **MOOSE and similar intrusive frameworks** provide genuine monolithic multiphysics when models are rewritten against their internal variable, discretization, and assembly contracts. Their looser multi-application paths do not make arbitrary external or multilingual models interchangeable with those intrusive kernels under one port contract.
- **PETSc and Trilinos** provide the distributed algebra required for monolithic composition but do not provide the model-port, field-semantic, transfer, lifecycle, or language-neutral substitution contract.

preCICE is the closest system to the black-box, multilingual, and replaceable-transfer portion of the target. KARMA builds a separate core because the shared black-box/intrusive contract, semantic field identity, MOAB/PETSc integration without semantic leakage, and model/emulator substitution cannot be added as a shallow preCICE participant extension. This decision does not imply rebuilding mature solvers, mesh infrastructure, array protocols, or remapping kernels: KARMA reuses PETSc, MOAB, LAPACK, DLPack-compatible protocols, and verified external transfer implementations behind explicit adapters.

This design does **not** claim to be the first general-purpose coupler, multilingual coupler, model/emulator interface, or framework with explicit and implicit algorithms. In particular, partitioned implicit iteration and a shared monolithic Newton system are different capabilities and must never be reported as equivalent.

Building a new coupling core carries substantial risk: distributed transaction and communicator correctness, numerical stability of partitioned algorithms, conservative-remap validation, ABI maintenance, packaging, observability, performance portability, and long-term conformance testing. The decision must be revisited if any of the following occurs:

- An existing framework or documented extension satisfies the mandatory contract with less complexity.
- Ordinary component or semantic interfaces begin to depend on PETSc, MOAB, MPI, or one transport's concrete types.
- The monolithic path cannot reproduce a direct PETSc formulation's convergence behavior within defined overhead budgets.
- The partitioned path cannot reproduce accepted fixed-point and acceleration results at the required MPI scale.
- A new language requires changes to the core semantic protocol rather than an adapter and capability record.
- Emulator substitution remains signature-compatible but cannot validate applicability, state, physical, derivative, and reproducibility contracts.
- No representative application demonstrates both partitioned and monolithic use of the same component contracts.

Future adapters may expose or consume preCICE participants, FMI 3.0 FMUs, NUOPC caps, and other ecosystem interfaces where their semantics are sufficient. They are extension points, not current delivery commitments, and must not be described as preserving monolithic semantics when the external standard does not expose them.

## Agnosticism Boundaries

The architecture is agnostic only where an implementation choice can vary without weakening scientific or distributed correctness.

| Axis | Architectural mechanism | Common failure | Falsifiable conformance test |
| --- | --- | --- | --- |
| Infrastructure | Semantic specifications and component contracts are separate from mesh, algebra, communicator, storage, and transport adapters. | Public contracts leak `Vec`, MOAB handles, backend ownership, file formats, or one MPI topology. | Replace a mesh, algebra, or transport adapter without changing model source or port schemas and reproduce declared results within tolerance. |
| Coupling algorithm | Component evaluation capabilities are separate from explicit, fixed-point, accelerated, and monolithic drivers. | The port contract encodes one stepping order, relaxation scheme, or universal assumption about rollback or derivatives. | Run unchanged model adapters under an explicit schedule and an iterated-partitioned schedule; when derivative capabilities exist, also run a monolithic driver. |
| Spatial transfer | Transfer operators declare source/target spaces, geometry, ownership, setup/apply lifecycle, and verified numerical properties. | Mapping is embedded in a connector, assumes matching meshes, or silently binds a field to one library. | Replace a consistent interpolation with conservative projection on the same ports and independently verify the properties each operator claims. |
| Execution architecture | ABI-neutral schemas combine with explicit adapters for native, managed-runtime, remote, CPU, and device execution. Ownership, synchronization, memory space, and collective participation are declared. | "Language-neutral" means only C-callable while callbacks, exceptions, communicators, device memory, or collective order still assume one process architecture. | Substitute native C++, Fortran, and remote implementations behind one component contract and pass the same lifecycle, failure, ownership, and numerical conformance tests. |

The following invariants are not negotiable:

- Stable semantic identity for fields, components, units, tensor shape, domain, temporal meaning, and discretization role.
- Explicit communicator, ownership, ghosting, empty-partition, and collective-participation rules.
- Deterministic lifecycle and state-transition semantics, including rollback where advertised.
- Declared mutability and transactional publication; failed operations expose no partial output.
- Capability negotiation for advance, residual, derivative action, transpose action, checkpointing, transfer guarantees, and reproducibility.
- Collective failure reconciliation before any rank may throw or proceed to a later collective.
- Transfer contracts that state consistency, conservation, orientation, coverage, and boundedness claims.
- One implementation-independent conformance suite for prognostic, data-backed, and emulator components.

KARMA is deliberately opinionated about semantics and correctness while remaining agnostic about implementations. Units, signs, field identity, residual conventions, derivative meaning, ownership, lifecycle, collective ordering, failure propagation, and commit/rollback behavior must be fixed and testable. Agnosticism is harmful when it permits ambiguous residuals, arbitrary memory ownership, undeclared collectives, silent unit conversion or model substitution, incompatible derivative definitions, or transfer claims without numerical evidence.

## Layering

1. **Semantic specifications:** `FieldSpec`, `PortSpec`, `ComponentSpec`, `TransformSpec`, `TransferSpec`, `ConnectionSpec`, and `CouplingGraphSpec`.
2. **Runtime bindings:** PETSc vectors/matrices, MOAB tags, host/device arrays, Python arrays, remote handles, and checkpoint state.
3. **Component execution:** black-box lifecycle plus residual/Jacobian capabilities, state checkpoints, rollback, cancellation, and deterministic retry.
4. **Graph execution:** transformations, spatial/temporal transfer, schedules, block residual/Jacobian composition, and transactional publication.
5. **Plugins and transports:** static C++ factories, a versioned dynamic C ABI, embedded Python, and replaceable out-of-process transports.

The semantic graph never depends on a concrete implementation. A plugin must prove compatibility with the graph before initialization.

## Field Contract

A field has immutable semantic identity independent of storage:

```cpp
struct FieldSpec
{
    FieldId id;
    SemanticVersion version;
    std::string display_name;
    FieldValueShape shape;
    OptionalScaleSpec scale;
    PhysicalQuantity quantity;
    UnitSpec units;
    CoordinateFrameSpec frame;
    FieldRole role;
    SpatialFieldSpec spatial;
    TemporalFieldSpec temporal;
    ValueConstraintSpec constraints;
    ConservationSpec conservation;
    CompatibilityMetadata metadata;
};
```

### Identity

```cpp
struct FieldId
{
    NamespaceId name_space;
    std::string name;
};
```

Examples include `atmosphere.temperature`, `ocean.surface_heat_flux`, and `structure.displacement`. Aliases must be registered explicitly. Matching by display name, variable position, MOAB tag name, or plugin implementation name is prohibited.

`SemanticVersion` versions field meaning rather than storage. Changes to units, sign convention, tensor frame, time semantics, or conservation semantics are compatibility-relevant.

### Shape and Components

```cpp
enum class FieldValueKind
{
    Scalar,
    Vector,
    Tensor,
    SymmetricTensor,
    FixedComponents,
    DynamicComponents
};

struct FieldValueShape
{
    FieldValueKind kind;
    unsigned rank;
    std::vector<std::size_t> extents;
    ComponentOrdering ordering;
    std::vector<FieldComponentSpec> components;
};
```

Field components have stable semantic identities, not only array positions. This supports species, energy groups, tensor conventions, and named vector components without colliding with the coupling-level `ComponentSpec` type.

### Units and Physical Quantity

```cpp
enum class QuantityKind { Interval, Point };

struct PhysicalQuantity
{
    QuantityId id;
    DimensionVector dimensions;
    IntensiveExtensiveKind kind;
    QuantityKind quantity_kind;
};

struct UnitSpec
{
    UnitId id;
    RationalScale scale;
    RationalOffset offset;
};
```

Dimensional compatibility is validated before inserting explicit unit transformations. Intensive values, densities, extensive totals, flux densities, integrated fluxes, and fractions are distinct semantic classes. Transfer selection must respect these distinctions.

A point is a value in an affine space, such as absolute temperature; an interval is a difference or increment. Interval-to-interval conversion is scale-only and offsets are prohibited. Point-to-point conversion may be affine when quantity semantics match. Point-to-interval and interval-to-point conversion require an explicit reference-point operation. Until `QuantityKind` is implemented end-to-end, automatic unit insertion is restricted to zero-offset conversions.

### Coordinate Frames

```cpp
struct CoordinateFrameSpec
{
    FrameId id;
    FrameKind kind;
    TensorTransformationRule transformation;
};
```

Frames include global Cartesian, cylindrical, spherical, mesh tangent/normal, material, and plugin-defined frames. Vector and tensor frame conversion is an explicit transformation stage.

### Roles

```cpp
enum class FieldRole
{
    PrognosticState,
    AlgebraicState,
    Diagnostic,
    Forcing,
    BoundaryCondition,
    Observation,
    Parameter,
    Control,
    Uncertainty,
    Mask
};
```

Roles determine lifecycle and mutation semantics. Prognostic state participates in checkpoint and rollback; diagnostics may be recomputed.

### Spatial Specification

```cpp
struct SpatialFieldSpec
{
    DomainId domain;
    FieldLocation location;
    unsigned topological_dimension;
    BasisDescriptor basis;
    CoefficientRepresentation coefficients;
    Continuity continuity;
    SubsetSpec subset;
    OwnershipSemantics ownership;
};
```

Locations include vertex, edge, face, element, FE coefficient, quadrature point, particle, global scalar, and plugin-defined locations. `BasisDescriptor` includes stable compatibility identity, family, order, topology support, interpolation metadata, and orientation behavior. Subsets identify regions, boundaries, interfaces, masks, or named sets.

### Temporal Specification

```cpp
struct TemporalFieldSpec
{
    TimeSemantics semantics;
    TimeLevelSemantics levels;
    InterpolationCapability interpolation;
    AccumulationSemantics accumulation;
};
```

Semantics distinguish instantaneous, interval average, interval integral, beginning/end/midpoint values, lagged states, predictors, correctors, and history levels. Conversion between them requires an explicit transformation.

### Constraints and Conservation

```cpp
struct ValueConstraintSpec
{
    FinitePolicy finite;
    OptionalBounds bounds;
    PositivityPolicy positivity;
    NormalizationPolicy normalization;
    MissingValuePolicy missing;
};

struct ConservationSpec
{
    ConservationRequirement local;
    ConservationRequirement global;
    std::vector<InvariantSpec> invariants;
};
```

Constraints describe required semantics; they do not silently enable clipping or repair. Conservation requirements may include global/local mass, species sums, charge neutrality, momentum, positivity, and boundedness. Algorithm selection fails when requested guarantees exceed verified plugin capabilities.

## Runtime Field Binding

`KARMAField` remains a non-owning binding from a semantic `FieldSpec` to runtime storage:

```cpp
class KARMAField
{
public:
    const FieldSpec& specification() const;
    const FieldStorageBinding& storage() const;
    CommunicatorView communicator() const;
    OwnershipView ownership() const;
    FieldAccessMode access_mode() const;

    FieldReadView read() const;
    FieldWriteTransaction begin_write();
};
```

```cpp
using FieldStorageBinding = std::variant<
    PetscVectorBinding,
    MoabTagBinding,
    HostArrayBinding,
    DeviceArrayBinding,
    PythonArrayBinding,
    RemoteFieldBinding,
    PluginDefinedBinding>;
```

A binding validates its storage against the semantic specification. Runtime handles, ranks, local indices, and storage names are not semantic identity.

Writes are transactional:

```cpp
class FieldWriteTransaction
{
public:
    void stage(DenseEntrySpan, FieldValueSpan, StageMode);
    void stage_keyed(std::span<const KeyedFieldValue>, StageMode);
    LocalValidationResult validate_local() noexcept;
    [[nodiscard]] LocalCommitResult commit() noexcept;
    void rollback() noexcept;
};
```

Dense, batched staging is the runtime path. Keyed staging is reserved for setup, sparse diagnostics, and format boundaries; stable keys resolve to dense runtime indices before hot execution. `StageMode` declares complete replacement, sparse insertion, or additive accumulation. The field binding declares whether owned entries only or ghost entries are accepted, whether complete coverage is required, and how duplicate indices are handled; insertion rejects duplicates unless an explicit deterministic replacement rule exists, while accumulation combines them in canonical order. A local field transaction owns only local staged storage, so rollback is infallible and noncollective. Commit can still fail while mutating PETSc or MOAB storage and therefore returns a nonthrowing outcome; a strong graph transaction accepts a binding only if successful prepare makes publication infallible or recoverable under the graph commit protocol. Distributed bindings reconcile `validate_local()` through the connection or graph transaction before any rank commits or throws. No consumer observes partial updates.

## Port Contract

```cpp
enum class PortDirection { Input, Output, InputOutput };
enum class PortRequirement { Required, Optional, Conditional };

struct PortSpec
{
    PortId id;
    PortDirection direction;
    PortRequirement requirement;
    FieldRequirement field;
    OptionalScaleSpec scale;
    AccessSemantics access;
    UpdateSemantics update;
    std::vector<CapabilityRequirement> required_capabilities;
};
```

Inputs declare when they are read and whether they may change during nonlinear iteration. Outputs declare when they become valid. Optional ports require an explicit default provider or absence policy.

## Component Contract

Every implementation satisfies one common runtime lifecycle:

```cpp
class KARMAComponent
{
public:
    virtual ~KARMAComponent() = default;

    virtual const ComponentSpec& specification() const noexcept = 0;
    virtual ComponentCapabilities capabilities() const noexcept = 0;

    virtual void bind(const ComponentBindings&) = 0;
    virtual void initialize(const InitializationContext&) = 0;
    virtual FinalizationResult finalize() noexcept = 0;

    virtual Checkpoint checkpoint(const CheckpointContext&) = 0;

    virtual void request_cancellation(const CancellationToken&) = 0;
};
```

```cpp
struct ComponentSpec
{
    ComponentId id;
    SemanticVersion version;
    std::vector<PortSpec> ports;
    StateSpec state;
    TimeIntegrationSpec time;
    ExecutionRequirements execution;
    DeterminismSpec determinism;
    std::vector<DerivativeCapability> derivatives;
    OptionalSurrogateContract surrogate;
};
```

A C++ or Fortran prognostic solver, embedded managed-runtime emulator, dataset replay component, and remote service may satisfy the same specification only when their complete observable contracts match. Port signatures alone are insufficient for substitution.

## Black-Box Capability

```cpp
class BlackBoxComponent
{
public:
    virtual StepResult advance(
        const TimeInterval& interval,
        const StepContext& context) = 0;
};
```

Capabilities declare time stepping, statefulness, deterministic replay, checkpoint/restore/rollback, cancellation granularity, retry safety, thread/MPI requirements, and failure-state validity. Black-box components may participate in explicit coupling, waveform relaxation, fixed-point iteration, and derivative-free implicit coupling.

## Implicit Operator Capability

> **Deferred capability:** the near-term implicit path is iterated partitioned coupling with Aitken or Anderson/IQN acceleration. The interfaces below constrain future monolithic work so that the fixed-point implementation does not create an incompatible component contract.

Residual evaluation is the only required intrusive implicit operation. The general contract is a DAE residual

\[
F(t,u,\dot u,q)=0.
\]

A graph-issued immutable token identifies the candidate state, derivative state, coupling inputs, graph epoch, and requested time-discrete linearization. For an implicit time discretization the requested operator is

\[
F_u + \sigma F_{\dot u},
\]

where `udot_shift` stores \(\sigma\). This is backend-neutral and maps directly to PETSc TS `IJacobian` semantics without embedding PETSc in the component contract.

```cpp
struct LinearizationToken
{
    GraphEpoch graph_epoch;
    NonlinearIteration iteration;
    StateVersion state;
    InputVersionSet inputs;
    TimePoint time;
    OptionalStateVersion time_derivative_state;
    Real udot_shift;
    OptionalLinearizationToken jacobian_based_on;
};

class ImplicitComponent
{
public:
    virtual ResidualEvaluationResult evaluate_residual(
        const LinearizationToken&,
        const ImplicitEvaluationContext&,
        const ComponentStateView& state,
        const OptionalStateView& time_derivative,
        ComponentResidualView residual) = 0;
};
```

The graph issues the token, pins or versions every referenced state, records successful residual evaluation under that token, and rejects stale derivative requests before any collective. `apply_jacobian` evaluates the general DAE action \(F_u v + F_{\dot u}\dot v + F_q v_q\); a time-discretization adapter constructs the shifted action by setting `time_derivative_direction = udot_shift * state_direction`. Providers never apply `udot_shift` a second time. A dedicated assembled provider receives the same convention. Synchronous in-process calls need not echo the token; asynchronous and remote results must echo it. Deliberately lagged or approximate Jacobians identify their residual basis through `jacobian_based_on` rather than weakening token validation.

Optional interfaces are separate capabilities:

```cpp
struct ComponentTangentView
{
    ComponentStateView state_direction;
    OptionalStateView time_derivative_direction;
    PortDirectionBundle input_directions;
};

struct ComponentAdjointView
{
    ComponentStateWriteView state_result;
    OptionalStateWriteView time_derivative_result;
    PortAdjointBundle input_results;
};

class JacobianActionProvider
{
public:
    virtual ApplyResult apply_jacobian(
        const LinearizationToken&,
        const ComponentTangentView& direction,
        ComponentResidualView result) = 0;
};

class JacobianTransposeActionProvider
{
public:
    virtual ApplyResult apply_jacobian_transpose(
        const LinearizationToken&,
        const ComponentResidualView& direction,
        ComponentAdjointView result) = 0;
};

class AssembledJacobianProvider
{
public:
    virtual ApplyResult assemble_jacobian(
        const LinearizationToken&,
        JacobianBlockWriter&) = 0;
};

class MassOperatorProvider
{
public:
    virtual ApplyResult apply_mass(
        const LinearizationToken&,
        const ComponentStateView& direction,
        ComponentResidualView result) = 0;
};

class PreconditionerProvider
{
public:
    virtual ApplyResult assemble_preconditioner(
        const LinearizationToken&,
        PreconditionerBlockWriter&) = 0;
};
```

`MassOperatorProvider` is an optional optimization for separable forms such as \(F=M(t,u)\dot u-f(t,u,q)\); it is not the fundamental DAE interface. Black-box components with internal time integrators expose step or waveform maps instead. A black-box stepper without a tangent action cannot participate in exact state-level monolithic Newton.

The primary derivative abstraction is `Jv` and `J^T v`, not one implementation technique. Components declare finite-difference, analytic, AD-generated, assembled, or unavailable actions. Automatic differentiation is preferred for new AD-friendly kernels but is not assumed for legacy C++, Fortran, MPI communication, opaque libraries, nonlinear solvers, or mutable global state. Active scalar types never cross the C ABI. Across a connection, independently implemented actions compose explicitly. For an implicitly solved closure, the derivative is that of the converged solution map, normally obtained through tangent or adjoint solves rather than differentiation through every solver iteration.

## Heterogeneous Coupling Jacobians

A connection is source-only by default, \(q_t=T(u_s)\). Exact target-state-dependent coupling, \(q_t=T(u_s,u_t)\), uses a `StateDependentConnectionOperator` with explicitly named source and target-state bindings, primal views, source and target tangent actions, and corresponding transpose outputs; it is not represented by the unary `KARMAFieldTransform` interface. The exact component blocks are

\[
\frac{\partial R_t}{\partial u_s}
=
\frac{\partial R_t}{\partial q_t}\frac{\partial T}{\partial u_s},
\qquad
\frac{\partial R_t}{\partial u_t}
=
\left.\frac{\partial R_t}{\partial u_t}\right|_{q_t}
+
\frac{\partial R_t}{\partial q_t}\frac{\partial T}{\partial u_t}.
\]

Connections therefore may contribute to diagonal as well as off-diagonal blocks. A target-dependent transform must provide the target derivative for exact Newton. `DerivativePolicy::LagTarget` may freeze that dependence as an explicitly compiled Picard/quasi-Newton approximation; it is never inferred silently. For linear source-only transfer maps, the derivative is the transfer map itself.

This composition supports different meshes, bases, component counts, and rank layouts.

## Confirmed Requirements for Remaining Design

The remaining design sections must support:

- Static C++ and dynamically loaded plugins.
- A versioned C ABI for dynamic plugins.
- Embedded Python as an adapter, not a core ABI dependency.
- Out-of-process components through replaceable transports.
- Synchronous remote residual/Jacobian calls when capabilities provide timeout, cancellation, deterministic retry, and rollback.
- Black-box and implicit components in the same graph.
- Heterogeneous field spaces with forward transfer and derivative/adjoint actions.
- PETSc as the default implicit solver backend without embedding PETSc in semantic specifications.

## Connection Graph

The coupling graph is an immutable validated specification compiled into a runtime execution plan.

```cpp
struct CouplingGraphSpec
{
    GraphId id;
    SemanticVersion version;
    std::vector<ComponentInstanceSpec> components;
    std::vector<ConnectionSpec> connections;
    std::vector<MergeNodeSpec> merges;
    std::vector<InstanceSetSpec> instance_sets;
    std::vector<ClosureSpec> closures;
    std::vector<OverlapSpec> overlaps;
    std::vector<SamplePoolSpec> sample_pools;
    CouplingScheduleSpec schedule;
    SolverPolicySpec solver;
    GraphFailurePolicy failure;
};
```

A component instance separates semantic contract from implementation choice:

```cpp
struct ComponentInstanceSpec
{
    ComponentInstanceId id;
    ComponentId contract;
    PluginSelection implementation;
    CommunicatorSpec communicator;
    ResourceSpec resources;
    ConfigurationDocument options;
};
```

This permits native prognostic, embedded Python, dataset replay, emulator, or remote service implementations without changing ports or connections.

### Connection Specification

```cpp
using ConnectionOperatorSpec = std::variant<
    TransformSpec,
    TransferSpec,
    TemporalTransferSpec,
    MultiscaleOperatorSpec,
    StateDependentConnectionOperatorSpec>;

struct ConnectionSpec
{
    ConnectionId id;
    PortEndpoint source;
    PortEndpoint target;
    std::vector<ConnectionOperatorSpec> operators;

    CouplingMode mode;
    CouplingSchedule schedule;
    CoveragePolicy coverage;
    FailurePolicy failure;
    CachePolicy cache;
    DerivativePolicy derivatives;
};

struct PortEndpoint
{
    ComponentInstanceRef component;
    PortId port;
};

using ComponentInstanceRef = std::variant<
    ComponentInstanceId,
    MappedInstanceRef>;

struct MappedInstanceRef
{
    InstanceSetId set;
    InstanceBinding binding;          // CurrentMapInstance or explicit stable key
};
```

Connections are directed. Bidirectional physics uses two directed connections unless a specialized transformation explicitly declares symmetric semantics.

Graph compilation validates:

1. Source and target component instances.
2. Stable source and target port identities.
3. Port direction and requirement.
4. Compatibility after the complete ordered operator chain.
5. Type inference through every operator stage.
6. Spatial transfer capabilities and target-space availability.
7. Temporal semantics and required histories.
8. Value constraints and conservation requirements.
9. Derivative requirements for the selected coupling mode.
10. Communicator, ownership, empty-partition, and transaction-domain compatibility.
11. Schedule dependencies and collective failure boundaries.
12. Cycles as invalid, lagged, fixed-point, split, or monolithic cycles.
13. Canonical connection identity without duplicate implicit contributions.
14. Fan-in only through an explicit merge node.

No runtime fallback may silently change algorithms, units, derivative behavior, coverage, or conservation guarantees after graph compilation.

## Transformation Pipeline

A connection compiles into

\[
q_t = T_n \circ T_{n-1} \circ \dots \circ T_1(u_s).
\]

```cpp
class KARMAFieldTransform
{
public:
    virtual ~KARMAFieldTransform() = default;

    virtual const TransformSpec& specification() const noexcept = 0;
    virtual TransformCapabilities capabilities() const noexcept = 0;

    virtual FieldSpec infer_output(
        const FieldSpec& input,
        const TransformInferenceContext& context) const = 0;
    virtual void setup(const TransformSetupContext&) = 0;
    virtual void apply(
        const FieldReadView& input,
        FieldWriteTransaction& output,
        const TransformContext&) = 0;
};
```

Operator families include unit conversion; component selection, reordering, aggregation, and expansion; scalar/vector/tensor conversion; coordinate-frame rotation; sign convention changes; derived fields; nonlinear plugin transforms; temporal interpolation/averaging/integration; spatial interpolation/projection/remapping; multiscale lift/restrict/upscale/downscale; constraint validation; and explicit clipping, normalization, or repair.

Spatial and temporal transfer are operator kinds in this one ordered chain, not fixed fields beside an unordered transform list. There is no universally correct spatial-versus-temporal order when meshes, coverage, frames, or operators vary in time. The configured order is semantic and appears in the graph signature. Graph compilation may fuse adjacent compatible linear operators only after validating their individual and composed contracts.

Each stage receives one complete `FieldSpec` and a `TransformInferenceContext` containing the target port requirement and available runtime-space descriptors, then infers another specification. A spatial stage therefore does not attempt to infer its target domain or basis from the source alone. The final specification must satisfy the target port exactly or through an explicit compatibility rule. Extensive-to-intensive conversion must identify the measure and space in which that measure is defined.

Automatic insertion is restricted to safe and unambiguous transforms. Zero-offset interval unit scaling may be automatic. Affine point conversions, sign changes, clipping, aggregation, temporal integration, and conservative versus nonconservative transfer require explicit configuration.

### Linearity And Derivatives

```cpp
enum class TransformLinearity
{
    Linear,
    Affine,
    Nonlinear,
    Nondifferentiable
};
```

Capabilities declare exact forward, derivative, transpose, assembled sparse representation, finite-difference permission, automatic differentiation, state/time dependence, collective behavior, allocation stability, determinism, conservation, monotonicity, positivity, and boundedness.

Affine behavior is represented as

\[
q = T u + f,
\]

where persisted linear operators and derivative actions exclude constant fill `f`.

```cpp
class TransformDerivativeProvider
{
public:
    virtual ApplyResult apply_derivative(
        const LinearizationToken&,
        const FieldReadView& direction,
        FieldWriteTransaction& result) = 0;

    virtual ApplyResult apply_derivative_transpose(
        const LinearizationToken&,
        const FieldReadView& direction,
        FieldWriteTransaction& result) = 0;
};
```

`ApplyResult` reports convergence, iterations, residual norm, relative residual, and a bounded diagnostic. Collective implementations reconcile it before any rank throws. Solve-defined transfer derivatives are valid: for \(T=M_t^{-1}B\), the transpose is \(T^T=B^T M_t^{-T}\), including the transpose solver and preconditioner behavior. A symmetric shortcut is allowed only when both the operator and realized solve are verified symmetric.

Forward and transpose applications share one accuracy contract, reproducible initial-guess policy, and convergence check. A residual tolerance alone does not guarantee an action tolerance because error scales with \(\kappa(M_t)\). A relative dot-product target of \(10^{-10}\) normally requires forward and transpose action errors around \(10^{-12}\), adjusted for conditioning and cancellation. An assembled operator is the simplest certification path but is not mandatory; a deterministic matrix-free operator with verified actions is valid.

Residuals, component derivatives, transform derivatives, and connection derivatives use the one `LinearizationToken` defined by the implicit component contract.

### Spatial Transfer Specification

```cpp
struct TransferSpec
{
    TransferFamily family;
    TransferScheme scheme;
    PluginSelection backend;
    TransferOptions options;
    RequiredGuarantees guarantees;
    MapPolicy map_policy;
    CoveragePolicy coverage;
    PartialCoveragePolicy partial_coverage;
    TracePolicy trace;
};
```

Guarantees may require constant or polynomial reproduction, global/local conservation, positivity, monotonicity, boundedness, adjoints, determinism, and allocation-stable apply. A backend is eligible only when verified capabilities satisfy every requirement.

### Temporal Transfer

```cpp
struct TemporalTransferSpec
{
    TemporalScheme scheme;
    ExtrapolationPolicy extrapolation;
    HistoryRequirement history;
    ConservationRequirement integral_conservation;
};
```

Schemes include hold-last, interpolation, interval averaging/integration, conservative accumulation, predictor/corrector exchange, and plugin-defined transforms. Connection runtime owns temporal history so checkpoint/rollback and component replacement preserve coupling state.

### Explicit Fan-In And Merge

Multiple connections may target one logical input only through an explicit merge node. Implicit multiple writes and last-writer-wins behavior are invalid.

```cpp
struct MergeSpec
{
    MergeKind kind;                 // Sum, WeightedSum, MaskedSelect, PrioritySelect
    CoveragePolicy coverage;
    OverlapPolicy overlap;          // Add, Normalize, Error, Priority
    NormalizationPolicy normalization;
    DeterministicOrder order;
};

struct MergeNodeSpec
{
    MergeId id;
    std::vector<ConnectionId> inputs;
    PortEndpoint target;
    MergeSpec merge;
};
```

`Overwrite` is valid only for indegree one; multi-source selection uses `PrioritySelect` with explicit priorities. Merge validation considers union coverage, overlap multiplicity, missing regions, and physical meaning. `Sum` preserves conservation for independent additive extensive fluxes; normalized weighted sums do not claim conservation unless their weights have a verified conservative interpretation.

For \(q=M(T_1(u_1),\ldots,T_n(u_n))\), each incoming block is

\[
\frac{\partial R}{\partial u_i}
=
\frac{\partial R}{\partial q}
\frac{\partial M}{\partial T_i}
T_i'(u_i).
\]

For `Sum`, \(\partial M/\partial T_i=I\). State-dependent masks and weights must declare their additional derivatives or an explicit lagging policy. Canonical edge order fixes local accumulation order; decomposition-independent bitwise reproducibility additionally requires a reproducible reduction.

## Runtime Connections

```cpp
class KARMAConnection
{
public:
    const CompiledConnectionSpec& specification() const noexcept;

    void setup(const ConnectionBindings&);
    ConnectionTransaction begin_exchange(const ExchangeContext&);

    void apply_forward(ConnectionTransaction&);
    void apply_derivative(
        const LinearizationToken&,
        const CouplingDirectionView&,
        CouplingResidualView&);
    void apply_derivative_transpose(
        const LinearizationToken&,
        const CouplingResidualView&,
        CouplingDirectionView&);

    void commit(ConnectionTransaction&);
    [[nodiscard]] CollectiveRollbackResult rollback(
        ConnectionTransaction&,
        Deadline) noexcept;
};
```

Setup builds/loads transfer maps, allocates histories, resolves stable keys, and validates capabilities. Apply uses immutable verified setup state.

An exchange stages source reads, transformation state, history access, communication, spatial transfer, constraint validation, and target output. `KARMAConnection::commit` merges the staged result into its parent `GraphTransaction`; it does not publish globally committed storage or advance an observable version. Transaction-scoped reads see the latest tentative version permitted by Jacobi or Gauss-Seidel semantics, while external readers see only the prior committed version. The graph's final commit performs publication after all stages and ranks succeed. Failure preserves target fields, history, map state, visible input versions, and graph epoch.

## Coupling Modes And Schedule Algebra

Basic modes are explicit, lagged, predictor/corrector, fixed-point, and implicit. A graph cycle must declare an iterative or split mode; accidental algebraic cycles are rejected.

Rather than encode every method in one enum, the schedule is compositional:

```cpp
using ScheduleNode = std::variant<
    AdvanceNode,
    ExchangeNode,
    SequenceNode,
    ParallelNode,
    MapNode,
    ReduceNode,
    ClosureCallNode,
    SubcycleNode,
    RendezvousNode,
    SplitNode,
    FixedPointNode,
    NonlinearSolveNode,
    ConditionalNode,
    EventNode>;
```

`ConditionalNode` and `EventNode` declare predicate scope, distributed reduction, deterministic tie-breaking, and a pre-branch reconciliation epoch. Rank-local observations never select branches independently. All transaction-domain ranks agree on the branch before any participant enters a branch-specific collective.

Named methods are policy builders that construct validated schedule trees.

### Explicit Sequential And Parallel Coupling

Directed acyclic dependencies execute topologically. Independent components may run concurrently. Exchanges occur at explicit nodes and publish transactionally.

### Subcycling And Rendezvous

```cpp
struct SubcyclingSpec
{
    TimeRatio ratio;
    RendezvousPolicy rendezvous;
    TemporalInterpolationPolicy interpolation;
    AccumulationPolicy accumulation;
};
```

Rendezvous may occur every macro step, at fixed times, on events, before/after substeps, over waveform windows, or through adaptive negotiation. Histories participate in checkpoint and rollback.

### Lie/Marchuk And Strang Splitting

First-order sequential splitting is compiled as an ordered sequence of advances and exchanges:

\[
u^{n+1}=\Phi_B(\Delta t)\Phi_A(\Delta t)u^n.
\]

Strang splitting is symmetric:

\[
u^{n+1}=\Phi_A(\Delta t/2)\Phi_B(\Delta t)\Phi_A(\Delta t/2)u^n.
\]

The scheduler checkpoints the macro-step state before composition. Strang splitting requires component support for fractional increments and restart at the intermediate stage. Higher-order compositions additionally require every component to support each fractional or negative increment used by the selected composition.

### Semi-Implicit And IMEX Coupling

```cpp
struct PartitionedOperatorSpec
{
    std::vector<ConnectionId> explicit_edges;
    std::vector<ConnectionId> implicit_edges;
    LagPolicy explicit_lag;
    NonlinearSolverSpec implicit_solver;
};
```

Explicit dependencies are evaluated and version-frozen; implicit blocks then solve transactionally. Stage-level nodes support additive Runge-Kutta and other IMEX schemes.

### Fixed-Point And Iterated Coupling

Iterated partitioned coupling is the first implicit delivery path. Let \(H(q_k)\) be one complete unrelaxed coupling sweep from the macro-step checkpoint. The fixed-point residual is

\[
r_k = H(q_k)-q_k.
\]

```cpp
struct FixedPointNode
{
    std::vector<ComponentInstanceId> participants;
    std::vector<PortEndpoint> iterates;
    ScheduleNodeRef sweep_body;
    InitialGuessPolicy initial_guess;
    SweepOrder order;                 // Jacobi or explicit Gauss-Seidel order
    unsigned max_iterations;
    ConvergenceSpec convergence;
    AccelerationSpec acceleration;
    DivergencePolicy divergence;
    CheckpointPolicy checkpoint;
};
```

`iterates` define the coupling vector and field order; `sweep_body` defines the exact advances, exchanges, merges, and closure calls whose output is \(H(q)\). `ConvergenceSpec` maps each iterate to its residual field and metric, so cycles and noniterated ports are never inferred from participant membership alone. One participant checkpoint represents the initial macro-step state. Every trial is evaluated from that checkpoint; unconverged trial state is discarded and only the converged trial is committed. A full new checkpoint per iteration is neither required nor implied. A component is eligible only if identical checkpoint state plus identical coupling input produces the same output and trial state within a numerical replay bound that is strictly below the smallest applicable convergence tolerance after field scaling. Statistical-only reproducibility is ineligible for deterministic fixed-point, Aitken, or Anderson/IQN; stochastic coupling requires a separately specified convergence algorithm. Remote and black-box components may satisfy this through checkpoint/restore, an opaque checkpoint token, deterministic replay, or process restart plus reload. Otherwise they are explicit-only.

Convergence is dimensionally meaningful and evaluated on the unrelaxed residual. Each field has absolute and relative tolerances and a declared `FieldMetric`. The default for primal FE interface fields is a mass-weighted RMS norm; concatenating raw DoF vectors with different units or mesh density is invalid. Each field produces a normalized violation and the iteration accepts only when the maximum required violation is at most one. A relative-only criterion is prohibited near zero. The metric, scaling, covered region, and ownership reduction are part of the compiled graph signature.

```cpp
struct LeastSquaresPolicy
{
    unsigned max_history;
    Real dependency_tolerance;
    unsigned reused_time_steps = 0;
    Real initial_relaxation;
    LeastSquaresSolver solver;        // PivotedQR or SVD
    ReductionMode reductions;         // Fast, fixed-layout, or cross-layout reproducible
};
```

Initial implementations support constant relaxation, Aitken, then filtered Anderson/IQN-ILS. Acceleration stores only finite accepted evaluations, filters near-dependent residual-difference columns before solving, and never forms unfiltered normal equations. It falls back to initial relaxation if no column survives. History restarts after graph, layout, transfer, topology, time-step-regime, or parameter changes; rejected steps; nonfinite evaluations; severe stagnation; or configured safeguards. Cross-time-step reuse defaults to zero until validated for a problem class.

Distributed metric products use stable field/component/entry ordering and owned entries once. A designated rank solves the small dense system with a specified LAPACK routine and broadcasts coefficients. Native reductions may promise determinism for a fixed layout; decomposition-independent bitwise results require a separately advertised reproducible reduction and are not the default.

Transfer operators used inside the iteration are frozen for the macro-step and must provide stable repeated application, finite output, deterministic ownership accumulation, and verified constant reproduction for consistent intensive transfer. Extensive transfer claims require the configured conservation property. Operator-norm stability and reproduction are verified over each backend's supported geometry range. A topology or stencil change invalidates acceleration history. Nearest-neighbor, Shepard, RBF, moving least squares, matching L2, and intersection L2 advertise only properties established by their own patch, conditioning, conservation, and refinement tests.

### Newton Variants

> **Deferred capability:** fixed-point, Aitken, Anderson, and IQN require residual vectors, metrics, differences, and linear combinations but no analytic derivatives. The monolithic stack follows only after the partitioned path and derivative conformance suite are established.

The graph residual is assembled from component residuals and connection operators. Supported future policies include assembled Newton, matrix-free Newton-Krylov, Jacobian-free Newton-Krylov, inexact Newton, modified/lagged Newton, quasi-Newton, nonlinear Schwarz, block Newton, and Schur-complement preconditioning. Components and connections may both contribute diagonal and off-diagonal blocks according to declared state dependencies.

Remote implicit components may participate only when they advertise synchronous residual calls, supported Jacobian capabilities, deterministic request IDs, idempotent retry, cancellation, timeout, checkpoint/restore, linearization consistency, and defined transport-failure behavior.

## Multiscale Coupling

The graph remains an immutable validated specification. Multiscale instance cardinality is resolved at graph compilation from symbolic selectors, meshes, partitions, and configured maximum capacities. Runtime active masks choose work from a compiled instance set but do not change graph topology. Adaptive model refinement, unbounded instance creation, and dynamic region replacement are deferred; a future implementation performs them by quiescing the current plan, creating a new graph epoch, recompiling at a safe point, and migrating state by stable logical identity.

This static model expresses:

- Scale-separated and heterogeneous multiscale methods with fixed microproblems per macro element, quadrature point, region, or request site.
- Concurrent macro/micro regions with fixed interfaces or overlap.
- Equation-free lift/evolve/restrict schedules with bounded convergence-controlled micro evolution.
- Explicit upscaling, downscaling, homogenization, and learned closure operators.
- Fixed-capacity ensembles with deterministic parameter assignment and reduction.
- Arlequin, handshake, and blended overlap methods with declared no-double-counting rules.

### Scale And Instance Specifications

Scale is semantic metadata rather than an inference from mesh spacing or time step. Two fields on one mesh may represent instantaneous microscopic values and volume-averaged macroscopic values.

```cpp
struct ScaleSpec
{
    ScaleId id;
    SpatialSupport spatial_support;
    TemporalSupport temporal_support;
    CoarseGrainingSemantics coarse_graining;
    OptionalAveragingVolume averaging_volume;
    OptionalRepresentativeVolume representative_volume;
    FidelityLevel fidelity;
    std::vector<ScaleRelation> relations;
};

struct InstanceSetSpec
{
    InstanceSetId id;
    ComponentId component_contract;
    SymbolicInstanceSelector selector;
    CapacitySpec capacity;
    PlacementPolicy placement;
    StateIsolationPolicy state_isolation;
};
```

`ScaleSpec` is part of immutable `FieldSpec` identity. A port may require an exact scale, accept a declared scale relation, or leave scale unconstrained; `PortSpec::scale` narrows rather than overrides the field specification. Compatibility fails before operator insertion when the binding and port requirements conflict. `InstanceSetSpec::selector` maps stable macro request sites to stable global instance keys. Stable logical identity is `(instance_set, global_key)` and never a graph epoch, pointer, local index, or MPI rank. A separate `(graph_epoch, plan_generation)` version rejects stale runtime handles and cache entries without changing migration identity. Placement is compiled separately from logical topology.

### Cross-Scale Operators And Closures

```cpp
enum class MultiscaleOperatorKind { Upscale, Downscale, Lift, Restrict };

struct MultiscaleOperatorSpec
{
    OperatorId id;
    MultiscaleOperatorKind kind;
    PortEndpoint input;
    PortEndpoint output;
    ScaleId source_scale;
    ScaleId target_scale;
    ConservationPolicy conservation;
    NormalizationPolicy normalization;
    DerivativePolicy derivatives;
};

struct ReductionSpec
{
    ReductionKind kind;              // Sum, Mean, WeightedMean, Variance, Quantile, UserDefined
    WeightSpec weights;
    DeterminismPolicy determinism;
    EmptySetPolicy empty_set;
};

struct ClosureSpec
{
    ClosureId id;
    PortSet request;
    PortSet response;
    InstanceSetId instances;
    OptionalOperatorId lift_or_downscale;
    OptionalOperatorId restrict_or_upscale;
    OptionalSamplePoolId sample_pool;
    OptionalOverlapId overlap;
    ReductionSpec reduction;
    CachePolicy cache;
    ClosureFailurePolicy failure;
};

struct OverlapSpec
{
    RegionSelector overlap;
    WeightFieldSpec macro_weight;
    WeightFieldSpec micro_weight;
    ConsistencyForm consistency;
    AccountingPolicy accounting;
};
```

Lift, restrict, upscale, and downscale have distinct semantics but share the ordinary operator execution mechanism. A closure is one scoped request/response operation, not two independently schedulable half-calls. `OverlapSpec` validates partition of unity, consistency or energy terms, coverage, and no-double-counting; generic transfer plus unspecified weights is insufficient.

### Bounded Adaptive Sampling

Adaptive HMM sampling uses a fixed-capacity masked pool:

```cpp
struct SamplePoolSpec
{
    InstanceSetId instances;
    std::size_t minimum_samples;
    std::size_t maximum_samples;
    std::size_t batch_size;
    StatisticalStoppingSpec stopping;
    SampleIdentityPolicy identity;
    CapacityExhaustionPolicy exhaustion;
};
```

All pool members remain graph participants. Active masks control contribution, not topology. The plan defines global sample IDs, parameter generation and seeds, ownership, evaluation order, active-subset normalization, and failure handling. Inactive members contribute neutral values and still enter required collectives. Every rank reaches the same stopping decision; rank-local early exit is invalid. Capacity exhaustion is configurable per closure as deterministic failure or a returned `UnconvergedClosure` status, with deterministic failure the default. Continuing from an unconverged closure requires explicit caller policy and provenance; it is never silently treated as convergence.

### Multiscale Schedule Nodes

```cpp
struct MapNode
{
    InstanceSetId instances;
    ScheduleNodeRef body;
    ExecutionPolicy execution;
};

struct ReduceNode
{
    ValueRef inputs;
    ValueRef output;
    ReductionSpec reduction;
};

struct ClosureCallNode
{
    ClosureId closure;
    ValueRef request;
    ValueRef response;
    EvaluationMode mode;             // Trial, Residual, JacobianAction, CommittedAdvance
};
```

Within `MapNode::body`, `MappedInstanceRef{instances, CurrentMapInstance}` resolves to the current stable logical instance. Explicit stable keys may address a particular member outside a map when validation permits it. This binding is lexical and typed, not ambient mutable state. A `MapNode` remains a compact compiled operation rather than expanding millions of instances into `ParallelNode` records. Equation-free execution composes lift, map/advance or subcycle, restrict, and reduce. Overlap coupling uses the ordinary fixed-point or future nonlinear-solve nodes plus declared overlap operators.

A closure invoked from a macro advance, residual, or derivative action introduces nested execution. The runtime supplies a reentrant closure service and child transaction: stage the request, execute isolated micro instances, validate and reduce responses, return a staged value to the parent evaluation, and commit child state only when both closure policy and parent transaction permit. Parent rejection rolls back child state. Residual-mode closures are observationally pure; stateful microproblems therefore provide clone/reset or checkpoint/rollback. Committed history may be read as frozen input but is never advanced by a residual, derivative action, or rejected trial. History updates remain staged until the parent macro-step commits.

### Future Adaptive Extension Points

The static implementation preserves extension points needed for later adaptation: graph epochs and safe-point plan replacement; symbolic selectors; independently versioned instance and transfer mappings; lifecycle factories with clone/reset/checkpoint/migrate/restore; placement-independent state serialization; schedule indirection through instance tables; bounded active masks; plan quiescence while transactions or closures are active; and caches keyed by graph epoch, operator version, instance state, and request data. These mechanisms do not authorize runtime graph mutation in the initial implementation.

## Compiled Execution And Performance

The semantic hierarchy is compile-time/setup metadata, not a hot-loop dispatch chain. Graph compilation produces flattened immutable plans. Every configured operator stage appears exactly once and retains its semantic order; `fused_groups` references contiguous compatible ranges rather than adding a second execution path:

```cpp
struct CompiledConnectionPlan
{
    FieldBinding source;
    FieldBinding target;
    std::vector<CompiledOperatorStage> stages;
    FusedKernelGroups fused_groups;
    OptionalDerivativePlan derivative;
    OptionalTransposePlan derivative_transpose;
    CommunicationPlan communication;
    CommitPlan commit;
};
```

Runtime follows direct pre-resolved views and kernels. It must not repeatedly traverse graph/component/port/registry metadata.

Performance rules:

- No virtual call per DoF, element, quadrature point, or component value.
- Type erasure only at coarse `advance`, `apply`, residual, and Jacobian boundaries.
- Inner loops use concrete kernels, compact function tables, or compiled execution objects.
- Stable IDs resolve to dense runtime indices during setup.
- Identity transforms disappear; compatible transforms are fused.
- Unit scaling, signs, permutations, and frame operations may fuse into transfer kernels.
- Communication routes, ownership, PETSc mappings, and transfer rows are cached.
- Small callables use inline storage where practical.
- Static plugins may specialize/inline kernels.
- Dynamic plugins pay one coarse function-table dispatch.
- Python/remote calls are batched to amortize interpreter/transport overhead.
- Repeated apply/residual/Jacobian operations are allocation-stable after setup where advertised.
- Debug metadata is separate from hot execution records.
- Plans are immutable for thread safety and reproducibility.

## Plugin Architecture

Plugins are capability providers selected after semantic graph validation. The plugin layer has static C++ and dynamic C ABI paths that both construct the same type-erased runtime interfaces.

### Static Registry

```cpp
class KARMAPluginRegistry
{
public:
    void register_component(ComponentPluginDescriptor);
    void register_transform(TransformPluginDescriptor);
    void register_transfer(TransferPluginDescriptor);
    void register_transport(TransportPluginDescriptor);
    void register_solver(SolverPluginDescriptor);

    PluginResolution resolve(const PluginRequest&) const;
    RegistryInventory inventory() const;
};
```

Descriptors contain stable plugin ID/version, implementation kind, semantic contracts, capabilities, supported scalar/index/device/MPI ABIs, factory, configuration schema, and provenance.

Registration rejects duplicate IDs/versions unless an explicit replacement policy is used. Resolution is deterministic and never silently falls back to a lower-capability implementation.

Static factories return type-erased owning handles. Built-in numerical kernels may retain concrete fast paths in compiled plans.

### Dynamic Plugin C ABI

C++ ABI compatibility is not assumed across shared-library boundaries. Dynamic plugins export one permanent unversioned bootstrap entry point that negotiates an ABI interval:

```c
KARMA_PLUGIN_EXPORT
karma_status_v1 karma_plugin_query(
    uint32_t host_min_abi,
    uint32_t host_max_abi,
    const karma_host_api_v1* host,
    karma_plugin_descriptor_v1* plugin);
```

A host cannot discover arbitrary future versioned symbol names, so descending probes of `karma_plugin_query_vN` are not the compatibility mechanism. The descriptor reports the plugin-supported interval and selected size-prefixed function table.

The ABI uses fixed-width integers, explicit byte counts, opaque handles, function tables, and caller-owned buffers. No STL, exceptions, PETSc/MOAB C++ objects, or compiler-specific RTTI cross the boundary.

```c
typedef struct karma_plugin_api_v1 {
    uint32_t struct_size;
    uint32_t abi_major;
    uint32_t abi_minor;
    const char* plugin_id;
    const char* plugin_version;

    int (*describe)(karma_buffer_v1* out);
    int (*create)(const karma_create_args_v1*, karma_handle_v1*);
    int (*destroy)(karma_handle_v1);
    int (*capabilities)(karma_handle_v1, karma_buffer_v1* out);
    int (*invoke)(karma_handle_v1, const karma_call_v1*, karma_call_result_v1*);
} karma_plugin_api_v1;
```

`describe` and capability payloads use a canonical versioned serialization format validated by the host. High-volume calls use opaque field views and batched operations, not generic value-by-value messages.

ABI negotiation checks major compatibility, known struct prefixes, scalar/index representation, endianness where relevant, MPI ABI/communicator ownership, device memory support, thread level, and required host services. The caller sets each input/output table's `struct_size` to its available capacity before the call; the callee validates it and writes only the known prefix. Required size is returned separately when discovery is needed.

The preferred stable boundary carries a framework-owned communicator/session handle. An optional native MPI adapter accepts `MPI_Fint` produced by `MPI_Comm_c2f` only when host and plugin use a configuration-verified compatible MPI runtime; `MPI_Fint` is not assumed fixed-width. `MPI_ABI_Comm` may be offered when both toolchains support the MPI 5 standard ABI. Native `MPI_Comm` never appears in the stable plugin ABI.

Errors are numeric status plus bounded UTF-8 diagnostics retrievable through host-provided buffers. Exceptions must be caught inside the plugin. A failed `destroy` leaves the handle live, valid, and retryable; it cannot partially invalidate the object. Indeterminate cleanup quarantines the plugin and keeps its library loaded. At process shutdown, leaking that mapping is safer than unloading code that may retain callbacks, threads, or objects.

Dynamic library lifecycle:

1. Load library without global symbol pollution.
2. Resolve the unversioned bootstrap query symbol.
3. Negotiate ABI interval, table prefixes, and capabilities.
4. Validate configuration schema.
5. Construct plugin instance.
6. Compile its interfaces into graph plans.
7. Destroy instances before unloading.
8. Refuse unload while plans or field views remain alive.

### Host API

The host function table supplies logging, allocation, cancellation polling, time, stable-ID helpers, communicator duplication, field-view queries, and optional PETSc/MOAB adapter services. Plugins request optional services by capability rather than linking private host symbols.

## Fortran Component Adapter

Fortran is the first non-C++ language target. A bridge module built with the component's Fortran toolchain implements the C ABI through `ISO_C_BINDING`; C++ never depends on a compiler-specific Fortran C++ ABI.

Array arguments use descriptors from `ISO_Fortran_binding.h`:

```c
typedef void* karma_component_handle_v1;

typedef struct karma_fortran_input_v1 {
    uint32_t struct_size;
    const CFI_cdesc_t* descriptor;
    uint64_t bridge_abi_id;
} karma_fortran_input_v1;

typedef struct karma_fortran_output_v1 {
    uint32_t struct_size;
    CFI_cdesc_t* descriptor;
    uint64_t bridge_abi_id;
} karma_fortran_output_v1;

int karma_component_advance_v1(
    karma_component_handle_v1 handle,
    const karma_fortran_input_v1* const* inputs,
    size_t input_count,
    karma_fortran_output_v1* const* outputs,
    size_t output_count,
    double time,
    double dt,
    char* message,
    size_t message_capacity);
```

Arbitrary-rank descriptors are passed as a table of pointers rather than as a contiguous `CFI_cdesc_t` array, whose element size is rank-dependent. `CFI_cdesc_t` remains inside a Fortran bridge compiled against the same processor runtime; it is not the general dynamic-plugin field ABI. `bridge_abi_id` identifies the verified compiler/runtime bridge. Input wrappers carry a pointer to a const descriptor; output wrappers carry a mutable descriptor because allocatable-result creation updates descriptor metadata. The bridge preserves descriptor rank, extents, lower bounds, element length, attributes, and byte strides. It never assumes contiguity or row-major order. A component that requires contiguous arrays declares that requirement; any pack/unpack operation is explicit and accounted for. Typed assumed-rank or `type(*), dimension(..)` bridge procedures may be used where supported.

Caller-allocated outputs are preferred. Fortran allocatable results are allocated through an allocatable descriptor and released by an explicit routine compiled with the same Fortran runtime; C++ never calls `free` or `delete` on them. Pointer, allocatable, and ordinary descriptor attributes remain distinct, and pointers to temporary or automatic arrays may not escape a call. A null descriptor pointer represents an absent optional argument under the versioned ABI convention. Character data uses `character(kind=c_char)` buffers with explicit lengths; logical values use `logical(c_bool)` and C `_Bool`; default character and logical representations do not cross the boundary.

The host converts a communicator with `MPI_Comm_c2f` and passes the resulting `MPI_Fint` representation only through a configuration-verified bridge. `MPI_Fint` is not assumed equivalent to `int` or `integer(c_int)`. Generic adapters do not expose `mpi_f08` derived-type handles; a component requiring `mpi_f08` establishes or duplicates its communicator through an MPI-implementation-specific Fortran bridge. Components may duplicate but never free the host communicator, and they declare MPI thread-level and collective-order requirements.

Fortran procedures return status codes and caller-owned bounded diagnostic buffers. `stop`, `error stop`, language exceptions, and nonlocal unwinding may not cross the ABI. The host reconciles failures collectively before another MPI or PETSc collective.

The initial statically linked Fortran adapter supports lifecycle, metadata and capability queries, field import/export, `advance`, checkpoint/rollback/commit, parameters, and diagnostics. Residual and optional Jacobian-vector actions follow the deferred intrusive contract; multiscale closure execution follows the static multiscale runtime. Dynamic loading is a later use of the same versioned C call schema. It does not promise arbitrary C++ callbacks, direct PETSc/MOAB C++ object ownership, asynchronous coroutines, or zero-copy access to every noncontiguous array. Conformance tests specifically target descriptor/runtime mismatches, stride and lower-bound errors, hidden copies, MPI-handle ownership, thread level, and collective order.

## Surrogate And Emulator Components

An emulator is not safely interchangeable with a prognostic model merely because its ports have matching names and shapes. The complete future substitution contract also validates physical guarantees, uncertainty semantics, batching meaning, and accuracy envelopes. The initial contract can enforce applicability, state/history behavior, derivative behavior, determinism, distribution compatibility, and artifact identity; until the deferred records exist, it must reject rather than claim substitution based on physical guarantees, uncertainty, batching, or accuracy. The initial contract records the four load-bearing properties:

```cpp
enum class SupportLevel { None, Declared, RuntimeChecked, EmpiricallyTested, Certified };
enum class StateKind { Stateless, ExplicitState, HiddenCheckpointable, HiddenNotCheckpointable };
enum class DeterminismLevel { None, Statistical, RepeatableConfiguration, Numerical, Bitwise };
enum class DerivativeKind
{
    None,
    FiniteDifference,
    AnalyticJvp,
    AnalyticVjp,
    ADGeneratedJvp,
    ADGeneratedVjp,
    AssembledJacobian
};

struct ApplicabilityDomain
{
    ArtifactId feature_schema;
    ArtifactId training_domain;
    ArtifactId normalization;
    OODDetectorSpec detector;
    Real warning_threshold;
    Real rejection_threshold;
    DistributedDiagnosticReduction reduction;
    SupportLevel validation;
};

struct DeterminismSpec
{
    DeterminismLevel level;
    Real numerical_tolerance;
    bool deterministic_reductions;
    bool deterministic_kernels;
    bool tf32_enabled;
    bool autotuning_enabled;
    HardwareScope hardware_scope;
    ArtifactId software_stack;
};

struct DerivativeCapability
{
    DerivativeKind kind;
    FieldSpaceId input_space;
    FieldSpaceId output_space;
    bool distributed;
    bool transpose_consistent;
    SupportLevel verification;
    Real expected_relative_error;
    Real cost_relative_to_evaluation;
};

struct SurrogateContract
{
    ArtifactId model_artifact;
    ArtifactId reference_model;
    ApplicabilityDomain applicability;
    StateKind state_kind;
    OODAction ood_action;
};
```

`HiddenNotCheckpointable` makes a recurrent or adaptive model ineligible for rollback-based iteration even when its ordinary fields look stateless. A surrogate may provide cheap AD-generated JVP/VJP while the reference solver provides only finite differences; capabilities are negotiated per implementation rather than ranked by model type. AD availability does not establish correctness across clipping, stochastic layers, OOD branches, unsupported exported operations, or hidden-state mutation.

On every inference, rank-local OOD and invariant diagnostics are reconciled through the declared distributed reduction before state commit. The configured action may fail closed, reject the step, request a preauthorized model switch, or continue with an explicit warning only where the experiment permits it. A model switch is never silent: it is prevalidated, collectively agreed, exposed to the caller, and recorded with detector version, scores, thresholds, affected fields and ranks, and resulting implementation identity.

Physical guarantees, uncertainty-output semantics, and batching semantics are mandatory future extensions before a surrogate may claim those capabilities. The design must distinguish a monitored property from a guaranteed property; checking mass does not guarantee conservation. Future batching records distinguish independent samples, time sequences, spatial subdomains, and ensemble members so centralized accelerator batching cannot silently violate domain ownership. Until those records exist, such claims are unavailable rather than inferred.

## Embedded Python Adapter

Embedded Python is an adapter plugin, not part of the core ABI. The adapter owns interpreter integration and presents the standard component interfaces.

```cpp
struct PythonComponentSpec
{
    std::string module;
    std::string factory;
    PythonEnvironmentSpec environment;
    PythonArrayPolicy arrays;
    PythonExecutionPolicy execution;
};
```

The Python object may implement black-box methods and optional residual/Jacobian/transpose/preconditioner methods. Method discovery occurs once during setup and becomes a capability record.

Array exposure rules:

- NumPy views may borrow contiguous host memory only for the call lifetime.
- Writable output uses staged buffers and commits after Python returns successfully.
- PETSc/MOAB ownership is never transferred to Python.
- Device arrays use explicit DLPack or plugin-supported protocols with stream/lifetime synchronization.
- Noncontiguous or incompatible layouts use explicit staging.
- Python objects must not retain borrowed pointers unless the adapter gives an owning capsule.

Execution rules:

- GIL acquisition is coarse and batched.
- MPI thread-level requirements are validated before setup.
- Python exceptions become deterministic bounded diagnostics.
- A failed Python call cannot partially publish target fields.
- Checkpoint/restore/release may use native callbacks, canonical state serialization, or adapter-managed copies.
- PyTorch, JAX, ONNX Runtime, and mpi4py integration are optional capabilities, not core dependencies.
- Embedded Python implicit calls require deterministic linearization tokens and declared derivative semantics.

Subinterpreters and free-threaded Python support must be capability-gated; the initial implementation may use one interpreter per process.

## Out-Of-Process Adapter

A remote adapter implements the same component contracts over a replaceable transport.

```cpp
class KARMARemoteTransport
{
public:
    virtual HandshakeResult connect(const EndpointSpec&) = 0;
    virtual RequestToken submit(const RemoteCall&) = 0;
    virtual RemoteResult wait(RequestToken, Deadline) = 0;
    virtual void cancel(RequestToken) = 0;
    virtual void close() noexcept = 0;
};
```

Transport plugins may use MPI intercommunicators, shared memory, sockets, gRPC, UCX, or site-specific mechanisms. Transport is not embedded in field semantics.

The handshake exchanges protocol version, component/field specifications, capabilities, scalar/index formats, compression, maximum message sizes, determinism, checkpoint/restore/rollback, cancellation, and supported residual/Jacobian operations.

Remote calls use deterministic IDs:

```cpp
struct RemoteCallId
{
    SessionUuid session;
    GraphEpoch epoch;
    MacroStep macro_step;
    TransactionGeneration transaction;
    NonlinearIteration iteration;
    ComponentInstanceId component;
    OperationKind operation;
    std::uint64_t sequence;
};
```

Retries resend the same ID. `SessionUuid` is persistent and globally unique for the run; restart either restores its remote cache namespace and sequence state or creates a new session and invalidates the old namespace explicitly. Cache-retention bounds and the retry horizon are negotiated. A compliant service returns the cached identical result or proves the prior call was not committed. Duplicate execution with externally visible side effects is forbidden.

Large fields use negotiated chunking, checksums, stable entry ordering, and optional zero-copy/shared-memory handles. Metadata and numerical payload channels may be separate. Remote field descriptors include semantic field ID/version and runtime distribution signature.

### Synchronous Implicit Remote Calls

The solver-facing adapter exposes synchronous residual/Jacobian semantics while internally submitting asynchronous transport requests. Independent calls may overlap, but completion and error selection are deterministic.

An implicit-capable remote component must provide:

- Timeout/deadline support.
- Cancellation acknowledgment.
- Deterministic request IDs.
- Idempotent retry behavior.
- Checkpoint/restore/release and rollback confirmation.
- Linearization-token consistency.
- Residual/Jacobian capability declarations.
- Explicit behavior after connection loss or partial response.
- Bounded resource usage and message sizes.

Timeout policy may cancel, retry deterministically, rollback and reduce time step, switch only to an explicitly configured compatible implementation, or abort. Silent model substitution is forbidden.

### Fault And State Model

Remote state transitions are explicit:

```text
Disconnected -> Handshaking -> Ready -> Executing
Executing -> Ready | Cancelling | Failed
Cancelling -> Ready | Failed
Failed -> Restoring -> Ready | Terminal
```

The graph commits no connection or component output until all relevant remote calls succeed and local validation passes. Recovery restores the graph checkpoint and connection histories before retry.

## Plugin Security And Provenance

Plugin descriptors record build ID, source/version provenance, ABI, configuration hash, and optional signatures. Dynamic and remote plugin loading uses explicit allowlists and search paths. Configuration never loads an arbitrary library or endpoint solely from untrusted field data.

Remote authentication, encryption, authorization, and sandboxing belong to transport/deployment policy but are visible in capabilities and provenance.

## Graph Compilation

Graph construction and execution are separate phases. The compiler consumes immutable specifications, registry inventory, runtime resources, and requested solver policy:

```cpp
class KARMACouplingGraphCompiler
{
public:
    GraphValidationReport validate(
        const CouplingGraphSpec&,
        const RegistryInventory&,
        const RuntimeEnvironment&) const;

    CompiledCouplingGraph compile(
        const CouplingGraphSpec&,
        const KARMAPluginRegistry&,
        const RuntimeEnvironment&) const;
};
```

Compilation is transactional. It creates plugin instances, resolves ports, builds transformation chains, selects transfer and solver plugins, establishes communicators, computes ownership and communication plans, allocates staging/history, and prepares rollback state. A failed compile destroys all temporary resources and publishes no graph.

The compiler produces:

- Dense component, port, field, connection, and block indices.
- Immutable component execution records.
- Fused connection kernels.
- Communication plans.
- State and history layouts.
- A schedule instruction stream.
- Nonlinear block layouts and solver plans.
- Canonical graph and capability signatures.
- An introspection/provenance snapshot.

Specifications remain available for diagnostics but hot execution uses compact records.

### Compatibility Resolution

Compatibility is evaluated with explicit reason codes:

```cpp
struct CompatibilityResult
{
    bool compatible;
    CompatibilityKind kind;
    std::vector<RequiredTransform> inserted_transforms;
    std::vector<CompatibilityDiagnostic> diagnostics;
};
```

Kinds include exact, lossless transformed, explicitly lossy, unsupported, and ambiguous. Lossy compatibility requires explicit graph authorization. Ambiguous resolution is rejected.

### Capability Closure

A graph capability is the closure of all component, connection, transport, scheduler, and solver capabilities. For example, an implicit graph has exact transpose actions only if every participating component and connection supplies compatible transpose behavior. Deterministic retry is available only if every stateful participant and transport supports it.

The compiler records why a requested capability is unavailable and identifies the first incompatible component/connection/plugin.

## Scheduler Runtime

```cpp
class KARMACouplingScheduler
{
public:
    void initialize(CompiledCouplingGraph&);
    StepResult execute_window(const TimeWindow&, ExecutionContext&);
    void cancel(const CancellationToken&);
    SchedulerPosition save_position() const;
    void restore_position(const SchedulerPosition&);
};
```

The scheduler executes a flattened instruction stream, not the specification tree. Instructions are coarse operations:

```cpp
enum class ScheduleOpcode
{
    AcquireCheckpoint,
    ReleaseCheckpoint,
    Advance,
    Exchange,
    Map,
    Reduce,
    CallClosure,
    BeginParallel,
    EndParallel,
    BeginIteration,
    CheckConvergence,
    Relax,
    EndIteration,
    BeginNonlinearSolve,
    EndNonlinearSolve,
    Rendezvous,
    Commit,
    Rollback,
    EmitEvent
};
```

Each instruction references dense runtime indices and precompiled callable records. Schedule builders produce the tree; compilation lowers it to opcodes, branch tables, iteration frames, and resource dependencies.

### Time And State Versions

Every observable field value is labeled by component state version and temporal validity. The scheduler advances versions only on commit. Predictor, corrector, lagged, and history fields retain distinct version identities.

```cpp
struct ExecutionStamp
{
    GraphEpoch epoch;
    MacroStep macro_step;
    Substep substep;
    NonlinearIteration nonlinear_iteration;
    TimePoint time;
};
```

Connections and components reject stale or mismatched stamps.

### Concurrency And Communicator Topology

Parallel nodes build dependency groups. The initial backend may execute groups sequentially while preserving semantics; later task runtimes may overlap independent native, managed-runtime, remote, and communication operations. Completion and error selection remain deterministic by canonical transaction epoch, not wall-clock arrival.

Each compiled graph has a transaction domain containing every rank that can affect publication. In the initial single MPI universe this is an explicit union intracommunicator. Independently launched MPMD programs may later supply an equivalent intercommunicator or coordinator control plane; that is not an initial delivery commitment.

Ranks need not execute every computational instruction, but all transaction-domain ranks execute the same ordered sequence of failure and publication epochs. At each epoch participating ranks perform local work, encode exceptions as local status, reconcile on component communicators, then reconcile one status over the transaction domain before entering a later collective or publishing. Nonparticipants contribute a successful no-op status. Error selection is deterministic over `(canonical_epoch, component_id, source_rank, error_code)` and the selected bounded diagnostic is broadcast before any rank throws.

Synchronization occurs at declared failure boundaries: before a new collective dependency, after an operation whose failure could diverge control flow, and before prepare/commit. It is not required after every local instruction. The compiled plan records stable epoch IDs, participant sets, communicators, and the collective sequence. No rank may enter a component collective after a peer could have failed without first reconciling that failure.

### Rendezvous And Adaptive Time

A rendezvous negotiates a common accepted time without allowing one component to publish beyond another component's rollback horizon. Adaptive components propose time increments; the scheduler applies policy constraints and retries from a shared checkpoint when a step is rejected.

## Nonlinear Solver Composition

The implicit graph exposes one type-erased nonlinear operator:

```cpp
class CoupledNonlinearOperator
{
public:
    void residual(const CoupledStateView&, CoupledResidualView&, const LinearizationToken&);
    void jacobian_action(const CoupledStateView& direction, CoupledResidualView& result,
                         const LinearizationToken&);
    void jacobian_transpose_action(const CoupledResidualView& direction, CoupledStateView& result,
                                   const LinearizationToken&);
    void assemble_jacobian(JacobianBlockWriter&, const LinearizationToken&);
    void assemble_preconditioner(PreconditionerBlockWriter&, const LinearizationToken&);
};
```

### Block Layout

The compiler assigns each implicit state port a stable block ID and constructs distributed block layouts without requiring matching meshes or rank ownership. A block records component, semantic field, runtime binding, communicator relation, owned stable keys, and current algebraic indices.

```cpp
struct CoupledBlockLayout
{
    BlockId id;
    ComponentInstanceId component;
    FieldId field;
    DistributedLayout layout;
    StableKeyResolver keys;
};
```

Nested or monolithic vectors are solver-backend choices. The semantic graph refers only to block IDs and field contracts.

### Residual Evaluation

For one linearization token:

1. Stage candidate component states.
2. Apply implicit connection forward transformations.
3. Bind transformed inputs without publishing component-visible committed state.
4. Evaluate component residuals.
5. Reconcile failures collectively.
6. Assemble the coupled residual view.
7. Release temporary borrows while retaining token-scoped caches.

No residual evaluation commits prognostic state or temporal history.

### Jacobian Actions

A Jacobian action composes:

- Component diagonal state derivatives.
- Component input sensitivities.
- Connection transformation derivatives.
- Spatial/temporal transfer derivative actions.
- Cross-component block accumulation.

If exact actions are unavailable, graph policy may select approved finite differences, automatic differentiation, secant updates, or lagged approximations. The fallback and probe scale are part of the compiled solver signature.

Finite differences perturb stable coupled blocks, evaluate the complete graph residual transactionally, validate finite probes, and restore the exact linearization state after every probe.

### Assembled Blocks

`JacobianBlockWriter` accepts stable row/column block IDs and runtime-local algebraic indices. Components may write diagonal and declared off-diagonal blocks. Connections may write transfer-derived blocks. Duplicate contributions are additive and ownership-aware. Final assembly occurs only after all providers succeed.

### PETSc Backend

The first solver plugin maps the coupled operator to PETSc SNES/KSP and may use:

- Monolithic AIJ matrices.
- `MatNest` block structures.
- `MatShell` matrix-free actions.
- Hybrid assembled preconditioner plus matrix-free operator.
- `PCFieldSplit` and Schur-complement policies.
- Nonlinear preconditioners and composite SNES methods.

PETSc callbacks catch all exceptions, reconcile rank-local failures, and place deterministic diagnostics in the PETSc error stack. Positive convergence reasons and independent residual checks are required before graph commit.

Solver options are namespaced by graph/component/block IDs to avoid plugin collisions. Ambient PETSc options cannot silently override semantic requirements such as exact versus approximate Jacobians or conservation policy.

### Partitioned And Monolithic Solvers

The same graph may compile to:

- Explicit schedule only.
- Fixed-point scheduler with optional acceleration.
- Partitioned nonlinear block Jacobi/Gauss-Seidel.
- Monolithic Newton.
- JFNK with physics-based block preconditioning.
- Hybrid explicit/implicit blocks.

Changing solver policy does not change component or field specifications, but compilation verifies the required capabilities again.

## State, Transaction, And Fault Semantics

### Graph Transaction

A macro step owns one graph transaction:

```cpp
class GraphTransaction
{
public:
    const GraphSnapshot& baseline() const;
    ComponentTransaction& component(ComponentInstanceId);
    ConnectionTransaction& connection(ConnectionId);
    LocalValidationResult validate_local() noexcept;
    [[nodiscard]] CollectiveValidationResult reconcile_validation(
        const LocalValidationResult&) noexcept;
    [[nodiscard]] GraphCommitResult commit(Deadline) noexcept;
    [[nodiscard]] GraphRollbackResult rollback(
        Deadline,
        RecoveryPolicy) noexcept;
};
```

The checkpoint covers component state, connection history, scheduler counters, random generator state where deterministic replay is promised, plugin adapter state, and remote checkpoint tokens. Checkpoints are LIFO-nestable so macro steps, subcycles, closure calls, rejected trials, and finite-difference probes can compose. Nesting may use copy-on-write, deltas, or backend-native savepoints rather than full copies.

`Checkpoint` is an RAII ownership handle with explicit `restore(Deadline)` and `release(Deadline)` operations. Its destructor performs local cleanup or transfers remote-release obligations to a session cleanup queue; it never starts MPI or remote rollback during unwinding and never pretends remote release succeeded. The session drains outstanding releases before transport shutdown.

Validation is split from reconciliation. Every rank first returns a nonthrowing local status, then `reconcile_validation` selects and broadcasts one deterministic graph-wide result. A throwing convenience wrapper, if provided, throws only after collective reconciliation. Unbounded dynamic allocation is not required to report failure: results carry code, rank, phase, and a bounded diagnostic handle or buffer.

Commit order is canonical and two-phase:

1. Prepare and reconcile validation for every participant.
2. Persist a graph-wide commit decision, then publish committed versions.

Every executable graph participant must guarantee that publication is infallible after successful prepare, or provide durable idempotent completion/recovery from the recorded commit decision. `GraphCommitResult` reports `committed`, `aborted`, `recovery_required`, or `indeterminate`; a failed acknowledgment after the decision is not treated as rollback permission. Local PETSc/MOAB bindings must stage or preflight every operation needed to make post-decision publication satisfy this contract. A plugin may report weak publication for diagnostics, but it is ineligible for all executable graph transactions until a separate isolated weak-transaction model with explicit partial-state visibility is designed.

Local `FieldWriteTransaction::rollback()` is infallible. Connection rollback returns a communicator-reconciled `CollectiveRollbackResult`. Graph rollback returns one of `rolled_back`, `restored`, `terminal_failure`, or `indeterminate`; `noexcept` means failures are returned rather than propagated as C++ exceptions. For remote participants, prepare, abort, restore, and commit acknowledgments are explicit and deadline-bound. If atomic distributed commit cannot be guaranteed, the capability records the limitation for diagnostics and the plugin is ineligible for executable graph transactions.

### Failure Classes

Failures are classified as validation, capability, numerical, timeout, cancellation, transport, resource, plugin, and terminal state corruption. Policies dispatch by class and operation.

```cpp
struct FailurePolicy
{
    RetryPolicy retry;
    RollbackPolicy rollback;
    TimeStepReductionPolicy time_step;
    FallbackImplementationPolicy fallback;
    TerminalPolicy terminal;
};
```

Retries are bounded, deterministic, and use the same call ID. Backoff affects waiting only, not numerical state. Implementation substitution is allowed only when explicitly configured and compatibility-validated at graph compile time.

### Cancellation

Cancellation tokens form a hierarchy: graph, schedule window, nonlinear solve, component call, and transport request. The initial component model is cooperative: `request_cancellation` may be invoked concurrently with an active call only when the component advertises that thread-safe capability; otherwise the scheduler sets a host-owned token that the component polls at its declared granularity. Managed-runtime adapters must define how polling acquires runtime state. Cancellation does not imply rollback completion; the scheduler waits for or times out rollback acknowledgment according to policy.

### Timeout

Deadlines use monotonic host time and are propagated to plugins/transports. Timeout handling must distinguish an unstarted, cancelled, completed-but-unreceived, and indeterminate call. Indeterminate state triggers restore or terminal failure, never blind retry.

### Determinism And Replay

A deterministic component declares which inputs, configuration, random seeds, external datasets, hardware modes, and parallel reductions affect replay. The graph records this provenance in checkpoints. Bitwise determinism and numerical reproducibility are distinct capabilities.

## Configuration And Introspection

The graph is constructible through direct C++ specifications and a canonical validated configuration document. Configuration parsing is separate from runtime semantics.

```cpp
class CouplingConfigurationLoader
{
public:
    CouplingGraphSpec parse(const ConfigurationDocument&) const;
    ConfigurationSchema schema(const RegistryInventory&) const;
};
```

The initial option layer should use a `KARMAOptionHandler` with a PETSc-backed default. Alternative MOAB or standalone parsers are adapters. Tests and drivers should not manually consume application options through scattered `argv` parsing.

### Configuration Requirements

- Explicit schema version.
- Stable component, port, field, connection, plugin, and schedule IDs.
- Plugin-specific options validated against plugin schemas.
- Unit/frame/temporal transformations visible in configuration or compiler report.
- No arbitrary dynamic library or remote endpoint loading outside configured allowlists.
- Environment-variable substitution limited to explicitly declared fields.
- Secrets referenced indirectly, never embedded in generated provenance or logs.

### Introspection

```cpp
struct CouplingGraphReport
{
    CanonicalGraphSignature signature;
    RegistryInventory plugins;
    std::vector<ResolvedComponentReport> components;
    std::vector<ResolvedConnectionReport> connections;
    ScheduleReport schedule;
    SolverReport solver;
    ResourceReport resources;
    ReproducibilityReport reproducibility;
};
```

The report exposes selected implementations, inserted transforms, transfer algorithms and guarantees, communicator layouts, map signatures, derivative paths/fallbacks, schedule lowering, solver blocks, remote endpoints by safe aliases, and all capability limitations.

A dry-run mode validates and compiles without initializing expensive model state. A graph-diff tool compares semantic and implementation changes between configurations/checkpoints.

## Checkpoint And Restart

A graph checkpoint includes:

- Graph specification/signature.
- Plugin identities, versions, ABI, and configuration hashes.
- Component committed states and versions.
- Connection temporal histories and map signatures.
- Scheduler position and iteration state at a safe boundary.
- Remote state tokens or canonical snapshots.
- Determinism/replay provenance.

Checkpoints are written only at declared safe points unless every active participant supports transactional in-flight snapshots. Restart may select different compatible implementations only through explicit migration adapters and compatibility validation.

## Testing And Compatibility Strategy

### Specification Tests

- Field identity/version/unit/frame/shape compatibility.
- Port direction/requirement/update semantics.
- Transform type inference and forbidden implicit conversions.
- Conservation and constraint capability rejection.
- Canonical graph signatures and graph-diff behavior.

### Plugin Tests

- Static registration and duplicate rejection.
- Dynamic ABI major/minor negotiation and truncated struct prefixes.
- Malformed capability/configuration payloads.
- Plugin exception containment and resource cleanup.
- Refusal to unload live plugin instances.
- Provenance and allowlist enforcement.

### Component Contract Tests

A reusable compliance harness runs against native C++, Fortran, dynamic, managed-runtime, and remote components:

- Lifecycle ordering.
- Bind/initialize/finalize retry behavior.
- Checkpoint/restore/release, including nested LIFO use.
- Cancellation and timeout.
- Deterministic replay.
- Residual/Jacobian consistency where advertised.
- Output preservation after failure.
- Thread/MPI capability compliance.
- Fortran descriptor rank, stride, lower-bound, allocation, and communicator ownership.
- Emulator applicability, hidden-state, determinism, and derivative-capability enforcement.

### Transformation Tests

- Exact units and frame conversions.
- Component permutations and tensor conventions.
- Forward/derivative/transpose dot-product tests.
- Finite-difference derivative convergence.
- Affine fill excluded from derivatives.
- Conservation, positivity, and boundedness claims.
- Online versus persisted transfer equivalence.

### Scheduler And Multiscale Tests

- Explicit DAG ordering and parallel independence.
- Subcycling and rendezvous histories.
- Lie/Marchuk and Strang convergence order.
- Predictor/corrector and IMEX stage ordering.
- Fixed-point convergence, divergence, first-iteration relaxation, Aitken, and filtered Anderson/IQN.
- Per-field mass-weighted convergence metrics and zero-scale absolute tolerances.
- Acceleration restart after transfer, layout, graph-epoch, and rejected-step changes.
- One macro-step checkpoint restored before every implicit trial.
- Map/reduce closure execution and nested child-transaction rollback.
- Fixed-capacity adaptive sampling with inactive collective participants and identical rank decisions.
- Capacity exhaustion under both configured policies.
- Lift/evolve/restrict, ensemble reductions, and overlap no-double-counting.
- Rollback after each possible instruction failure.
- Event and conditional reproducibility.

### Nonlinear Solver Tests

- Independent residual oracle.
- Jacobian finite-difference and transpose dot-product tests.
- Assembled versus matrix-free agreement.
- Heterogeneous off-diagonal transfer blocks.
- Block preconditioner structure.
- PETSc convergence reason and independent final residual.
- Zero-owned block/rank participation.

### Remote And Python Tests

- Zero-copy/staged array lifetime violations.
- GIL and MPI thread-level enforcement.
- Python exception translation.
- Remote handshake compatibility.
- Timeout before/during/after execution.
- Duplicate request IDs and idempotent retry.
- Lost responses, partial chunks, corruption, reconnect, restore, and cancellation.
- Remote residual/Jacobian linearization-token mismatch.

### Numerical And Parallel Matrix

- Serial and 2/3/4/5-rank execution.
- Replicated and independently partitioned layouts.
- Bitwise comparisons where promised; otherwise scale-aware tolerances and recorded maximum error.
- Rank-count-independent stable-key comparisons.
- Empty partitions and zero-owned blocks.
- Different source/target rank counts for persisted maps and remote adapters.
- CPU/device and supported scalar/index ABI variants when available.

### Performance Measurement And Gates

- No per-DoF virtual dispatch, verified by code inspection and representative profiles.
- Allocation-stable warmed component, transform, transfer, fixed-point, residual, and Jacobian calls where advertised.
- Compiled-plan dispatch, communication volume, message count, managed-runtime batching, strong/weak scaling, load imbalance, and map/history/checkpoint memory are measured from the first delivery.
- A layer may report measurements before budgets exist, but those observations are not called performance contracts. Numeric pass/fail budgets are established against direct-call and current Phase 8 baselines before performance claims gate a release.
- Regressions are compared at fixed problem, compiler, build type, MPI layout, and hardware; cross-platform comparisons are informational unless normalized by an approved benchmark protocol.

### Mutation And Adversarial Tests

- Wrong field or component identity.
- Dropped/reordered transform.
- Incorrect derivative sign or transpose.
- Duplicate connection contribution.
- Stale linearization token.
- Skipped convergence check.
- Partial target commit.
- Retry with changed call ID.
- Silent plugin fallback.
- Remote duplicate side effect.
- Row/column stable-key misassociation.
- Corrupted map/checkpoint/plugin metadata.

## Compatibility And Versioning

Semantic, graph schema, dynamic ABI, remote protocol, map/checkpoint, and plugin configuration versions are independent. Compatibility matrices are explicit and tested.

- Major changes are incompatible unless a migration adapter is selected.
- Minor versions may add optional fields/function-table tails using size-prefixed structures.
- Patch versions do not change contracts.
- Unknown required capability bits are rejected.
- Unknown optional metadata is preserved where possible but never interpreted silently.
- Migration is explicit, provenance-recorded, and transactional.

## Migration From The Phase 8 Coupling Layer

The proposed `KARMAField` is a contract expansion, not source-compatible continuity. The current class has no `FieldSpec`, uses a two-alternative storage binding, materializes `FieldEntryKey` entries, and exposes keyed batch commits. The new runtime adds semantic specifications, read views, transactions, more storage kinds, access and ownership contracts, and dense compiled indices.

Migration therefore proceeds explicitly:

1. Define `StableEntryKey` as the versioned semantic successor or precise wrapper of the existing `FieldEntryKey`; do not maintain two undocumented identity systems.
2. Preserve keyed batch import/export at persistence and diagnostic boundaries while compiling dense runtime staging plans.
3. Adapt `karma_fem_transfer`, `karma_quasi_interpolation`, `karma_transfer_map`, map I/O, coverage, diagnostics, and conservative-transfer consumers before removing old constructors or names.
4. Version persisted transfer-map schemas if key meaning, ordering, or binding signatures change. Existing map readers either migrate explicitly or reject with a versioned diagnostic.
5. Preserve the current serial and partitioned 2/4/5-rank transfer baselines throughout migration; add 3-rank coverage where the graph layer requires it.
6. Rename `PetscFieldBinding` only as part of the explicit migration, not through a compatibility alias without a concrete external consumer.

Graph checkpoints and existing `KARMAMoabIO` checkpoints have separate ownership. Component adapters serialize their committed model state through the existing checkpoint facilities where applicable; graph checkpoints own orchestration state, connection histories, capability signatures, and references to component state records. One schema never silently embeds or supersedes the other.

## Phase Boundary And Delivery Strategy

Remaining Phase 8 numerical transfer hardening is explicitly deferred where no verified backend exists. The current `TransferCapabilities` and installed backends do not establish a general derivative/adjoint path, TempestRemap is unavailable in the verified stack, and the volumetric conservative backend remains deferred in the roadmap. This does not block the fixed-point path. It blocks exact monolithic transfer derivatives and conservation claims until an implementation satisfies the corresponding capability and dot-product/conservation tests.

Implementation proceeds as a thin vertical slice in reviewable layers:

1. Semantic IDs/specifications, compatibility engine, ordered operator inference, and exact-unit rules.
2. Runtime field transactions, batched staging, port registry, local validation, and collective reconciliation.
3. Minimal static registry, immutable graph compiler, explicit scheduler, and component compliance harness.
4. Transfer guarantees, frozen iteration maps, field metrics, and the current Phase 8 backend adapters.
5. Macro-step checkpoints, graph transaction domain, rollback outcomes, and failure epochs.
6. Fixed-point Jacobi/Gauss-Seidel with convergence metrics and constant relaxation, then Aitken.
7. Filtered Anderson/IQN-ILS with deterministic fixed-layout reductions and restart policy.
8. Stable shared C call schema plus the first statically linked Fortran lifecycle/field/advance adapter, with configure-time compiler/runtime/MPI bridge validation.
9. Static multiscale instance sets, map/reduce closures, bounded sampling, overlap semantics, nested execution, and corresponding Fortran closure support.
10. Dynamic loading for the established C ABI, configuration, introspection, provenance, checkpoint integration, security, and numeric performance gates.
11. Embedded Python and other managed-runtime adapters, then out-of-process transport and MPMD extensions.
12. Deferred coupled residual/Jacobian block abstraction, verified transfer derivatives, and PETSc monolithic backend.

Each layer preserves direct native fast paths and adds serial/2/3/4/5-rank tests where the capability is distributed. A later layer may not be described as supported merely because its interface is installed.

## Capability Comparison Rubric

Build-versus-adopt reviews classify each requirement at one of five levels:

- **L0, structurally incompatible:** the external system's core model conflicts with the requirement.
- **L1, external workaround:** the capability exists only by moving it outside the framework or collapsing components.
- **L2, intrusive/custom extension:** substantial framework-specific driver or component changes are required.
- **L3, native:** documented first-class abstractions represent the requirement.
- **L4, native and demonstrated:** maintained tests or published evidence establish it at the required distributed scale.

The comparison matrix keeps the following rows distinct:

| Area | Capability rows |
| --- | --- |
| Component boundary | Independent encapsulation; language-neutral ABI/protocol; in-process execution; out-of-process execution; MPMD/unequal-rank deployment |
| Distributed fields | Field exchange; partitioned-mesh semantics; ownership/ghosting; empty partitions; stable semantic identity |
| Partitioned algorithms | Explicit coupling; subcycling/multirate; iterated implicit coupling; Aitken/Anderson/quasi-Newton acceleration; transactional rollback |
| Monolithic algorithms | Shared unknown space; residual contribution; matrix-free JVP; VJP/transpose; assembled blocks; cross-component Newton/Krylov; block preconditioning |
| Transfer | Pluggable operators; conservative transfer; positivity/boundedness; transpose consistency; moving/nonmatching meshes |
| Runtime | CPU/device memory interoperability; asynchronous synchronization; collective-safe failure; checkpoint/restart; deterministic retry |
| Models and emulators | Model/emulator substitution; applicability/OOD enforcement; physical guarantees; uncertainty semantics; hidden state; batching; AD-generated actions; determinism |
| Governance | No silent fallback; versioned capability negotiation; units/frames/time semantics; provenance; language-independent conformance tests |
| Multiscale | Static instance sets; lift/evolve/restrict; closure requests; ensemble reduction; overlap accounting; bounded adaptive sampling; future safe-point graph replacement |

An implementation is not credited for "implicit coupling" when it supplies only fixed-point iteration, nor for "monolithic coupling" when all physics has been collapsed into one opaque participant. Matrix entries cite evaluated versions and evidence rather than product-level impressions.

## Design Invariants

- Semantic identity never depends on storage, handles, indices, ranks, or implementation names.
- Be opinionated about semantics and correctness; remain agnostic about implementations.
- No silent fallback, model substitution, or capability downgrade.
- No per-value polymorphic dispatch in hot kernels.
- All caller-visible mutation is transactional.
- Every distributed failure is reconciled before a participant throws or enters a later collective.
- Fixed-point trials begin from one macro-step checkpoint and publish only after convergence.
- Residual and derivative actions share one immutable linearization token where derivative capabilities are used.
- Remote retry is idempotent and rollback-aware.
- Heterogeneous operators own forward and declared derivative/transpose semantics.
- Graph compilation resolves ambiguity before execution.
- Runtime plans are immutable, dense-indexed, batch-oriented, and introspectable.
- Static multiscale active masks never alter graph topology or collective participation.
- Conservation, determinism, applicability, and derivative claims require verified capabilities.
- A model implementation can be replaced only when every contract represented and required by the selected schedule is compatible. Until physical-guarantee, uncertainty, batching, and accuracy-envelope records are implemented, substitution requiring any of them is rejected.

## Deferred Capabilities

The initial graph is immutable. Dynamic adaptive model refinement, unbounded runtime instance creation, graph mutation, general MPMD control planes, managed-runtime coexistence, uncertainty and batching contracts for emulators, physical-guarantee certification for learned models, monolithic Newton, and general transfer adjoints remain deferred. Their extension points are retained, but an installed interface or enum is not evidence of support.

Compatibility-relevant changes are versioned rather than silently changing semantics. Additional refinements may include richer field ontology, subinterpreter support, transport-specific zero-copy, plugin package/signature policy, and future preCICE/FMI/NUOPC interoperability adapters.
