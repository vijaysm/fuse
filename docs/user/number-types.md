# Number types and mixed precision

A coupled field does not have to be binary64. A component can keep a field in single or half
precision, in a posit, in double-double or quad-double for extra digits, or as forward-mode dual
numbers that carry derivatives, and still be coupled to components that use something else. This page
is the map: which types exist, what the coupler does with each, and where the details are. Read it
before [](field-pipeline.md), which has the full reference.

## The types

| Type | Header | What it is | Kind |
|---|---|---|---|
| `double` | built in | binary64, the default for every field | reference |
| `float` | built in | binary32 | Storage |
| `fuse::bfloat16` | `fuse/model/bfloat16.hpp` | 8 exponent bits, 7 fraction bits | Storage |
| `fuse::binary16` | `fuse/model/binary16.hpp` | IEEE half precision | Storage |
| `fuse::num::posit16`, `fuse::num::posit32` | `fuse/num/posit.hpp` | `posit<16,2>` and `posit<32,2>` of the 2022 Posit Standard | Storage |
| `fuse::num::dd`, `fuse::num::qd` | `fuse/num/dd.hpp`, `fuse/num/qd.hpp` | double-double (about 106 bits) and quad-double (about 212 bits) | Arithmetic |
| `fuse::num::dual<S, N>` | `fuse/num/dual.hpp` | a value and its gradient in `N` parameters | ScalarArray |
| your own type | your header | trivially copyable, with a `fuse::value_traits<T>` | any |

The kind says how the coupler treats the type. **Storage** types are a compact way to hold binary64
numbers: the coupler widens them to binary64 exactly, does all its work there, and rounds once into
the receiving field. **Arithmetic** types are one number spread over several binary64 limbs, so the
coupler does its sums and remaps in the type. **ScalarArray** types are several independent scalars
(a value and its derivatives) that the coupler moves and combines scalar by scalar.

## A mixed-precision coupling

A binary64 producer on 8 cells sends a field through a remap to a consumer on 5 cells that keeps it
in `bfloat16`. The graph does not mention a precision: a field declares components, and the type is
whatever the component binds.

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include <Kokkos_Core.hpp>

#include <fuse/backend/serial_backend.hpp>
#include <fuse/base/error.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/bfloat16.hpp>
#include <fuse/model/value_traits.hpp>
#include <fuse/runtime/runner.hpp>

namespace u = fuse::units;

// Heat flux on 8 cells, computed in binary64.
class Producer final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "producer"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg, const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
    return reg.bind(g.component(*cid).fields[0],
                    fuse::FieldRef{q_.data(), fuse::Layout{.stride = 1, .count = q_.size()},
                                   fuse::MemorySpace::Host});
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    for (std::size_t i = 0; i < q_.size(); ++i) q_[i] = 1.0 + 0.37 * static_cast<double>(i);
    return fuse::Result<void>::success();
  }

 private:
  std::vector<double> q_ = std::vector<double>(8, 0.0);
};

// The consumer keeps its copy in bfloat16.
class Consumer final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "consumer"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg, const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    return reg.bind(g.component(*cid).fields[0],
                    fuse::BasicFieldRef<fuse::bfloat16>(
                        q_.data(), fuse::Layout{.stride = 1, .count = q_.size()},
                        fuse::MemorySpace::Host));
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  const std::vector<fuse::bfloat16>& values() const { return q_; }

 private:
  std::vector<fuse::bfloat16> q_ = std::vector<fuse::bfloat16>(5);
};

int main(int argc, char** argv) {
  fuse::initialize();
  Kokkos::ScopeGuard kokkos(argc, argv);

  fuse::CouplingGraph g;
  fuse::SerialMeshBackend backend;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:8"});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:5"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{
      .name = "producer", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto c = g.add_component(fuse::ComponentDecl{
      .name = "consumer", .ranks = fuse::RankSet{1, 1}, .interval = fuse::Rational{1}});
  const auto q_out = g.add_field(p, fuse::FieldSpec{.name = "q", .region = r1, .components = 1,
                                                    .units = u::kKelvin, .intent = fuse::Intent::Provides});
  const auto q_in = g.add_field(c, fuse::FieldSpec{.name = "q", .region = r2, .components = 1,
                                                   .units = u::kKelvin, .intent = fuse::Intent::Requires});
  g.add_channel("link", q_out, q_in);

  (void)backend.load(g.mesh(m1));
  (void)backend.load(g.mesh(m2));
  const auto lowered = fuse::lower(g);
  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));

  fuse::ComponentSet components;
  components.add(std::make_unique<Producer>());
  auto consumer = std::make_unique<Consumer>();
  const Consumer* cons = consumer.get();
  components.add(std::move(consumer));

  fuse::RunnerConfig cfg;
  cfg.measure_storage_rounding = true;                // count what the roundings did
  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  fuse::FieldRegistry registry;
  for (auto& comp : components) (void)comp.register_fields(registry, g);
  if (auto init = runner.initialize(components, registry, cfg); !init.ok()) {
    std::printf("%s\n", init.error().message.c_str());
    return 1;
  }
  (void)runner.run_period(components);

  for (const auto& v : cons->values()) std::printf("%.6f ", fuse::value_traits<fuse::bfloat16>::to_binary64(v));
  std::printf("\n");
  for (const auto& s : runner.stats().storage_fields) {
    std::printf("%s: %s, unit roundoff %.3g, %lld values rounded, largest relative change %.3g\n",
                s.field.c_str(), s.format.c_str(), s.unit_roundoff,
                static_cast<long long>(s.values), s.max_relative_change);
  }
}
```

Expected output (exact digits can vary by platform):

```text
1.140625 1.695312 2.296875 2.890625 3.453125 
consumer.q: bfloat16, unit roundoff 0.00391, 5 values rounded, largest relative change 0.00194
```

Each consumer value is the binary64 result of the remap, rounded once to `bfloat16`. The producer's
binary64 field was never narrowed. `RunStats::storage_fields` lists every field the run converts, and
with `RunnerConfig::measure_storage_rounding` set it also counts the values rounded and the largest
relative change the rounding made.

## What the coupler does

- **Storage fields** are read into a binary64 image after the component runs and rounded into the
  field when a value is delivered, to nearest with ties to even. Remaps, accumulations, the
  conservation check and the transport all see binary64. See [](field-pipeline.md#storage-types-binary32-bfloat16-and-binary16)
  and, for the posits, [](field-pipeline.md#storage-types-posit16-and-posit32).
- **Arithmetic fields** (`dd`, `qd`) are remapped, accumulated, merged, scaled and conservation-checked
  in their own type, so a conservation defect is the round-off of the type rather than of binary64. A
  channel cannot connect an Arithmetic field to a field of another type. See
  [](field-pipeline.md#double-double-and-quad-double).
- **Gradient fields** (`dual`) are moved as scalars. A remap `y = W x` has weights that do not depend
  on the field, so the gradient of `y` is `W` applied to the gradient of `x`. Minimum and maximum
  accumulation of a field with a gradient is refused (`E-RT-GRADIENT-MINMAX`). See
  [](field-pipeline.md#forward-mode-gradients).
- **Accumulating binary64 fields in double-double.** `RunnerConfig::accumulation_type` set to
  `AccumulationType::DoubleDouble` accumulates the row sums of a remap and the integrals of the
  conservation check of an ordinary binary64 field in double-double and rounds once.
- **Reverse mode** (adjoints) works on binary64 fields only; fields of the other types are refused by
  name (`E-ADJ-FIELD-TYPE`). See [](field-pipeline.md#reverse-mode-adjoints).

A field in a time window or an iteration region must be binary64, and a field that does not fit one of
the rules above is refused when the run is initialized with `E-RT-FIELD-TYPE` or `E-RT-ARITHMETIC-*`
rather than converted silently.

## From C, Fortran and plugins

A participant says what its storage holds with `element_type` in the field view, and a provider plugin
says it in the row it returns from `describe`:

| `element_type` | Storage you give | The graph counts |
|---|---|---|
| `FUSE_ELEMENT_F64` | `double` | 1 per value |
| `FUSE_ELEMENT_F32` | `float` | 1 |
| `FUSE_ELEMENT_BF16`, `FUSE_ELEMENT_F16` | `uint16_t` bit patterns | 1 |
| `FUSE_ELEMENT_POSIT16`, `FUSE_ELEMENT_POSIT32` | `uint16_t`, `uint32_t` bit patterns | 1 |
| `FUSE_ELEMENT_DD` | pairs of `double` (`hi`, `lo`) | 2 |
| `FUSE_ELEMENT_QD` | quadruples of `double` | 4 |

`gradient_size = g` marks the last `g` components of each entity as the gradient of the leading
ones. `FUSE_ELEMENT_I64` names identifiers and is not a coupled value. The full contract is in
[](c-abi.md#fields-field_handle-and-register_field), [](fortran.md) and [](providers.md).

## Restart

A restart file stores each of these types as itself or as its scalars: `put_field<T>()` takes the
Storage types directly, and `fuse::put_values<T>()` stores `dd`, `qd` and `dual` fields as their
binary64 scalars, bit for bit. See [](restart.md#typed-fields).

## Limits

- The Runner does not write restart files for a component's arrays; the component stores its own state.
- The image a Storage field needs costs 8 bytes per value on the arc, and MPI and the transport move
  binary64.
- Remap weights and mesh coordinates are binary64 in every case.
- Iteration regions, relaxation and residuals over `dd` or `qd` fields are refused
  (`E-RT-ARITHMETIC-OP`); mixed types across a channel or a merge of Arithmetic fields are refused
  (`E-RT-ARITHMETIC-MIX`).

## See also

- [](field-pipeline.md) for the reference of every type, the adjoint sweep and the portable kernels.
- [](point-cloud-remap.md#gradients-with-respect-to-the-point-positions) for the derivative of
  remap weights with respect to the point positions.
- [](c-abi.md), [](fortran.md), [](providers.md) and [](restart.md).
