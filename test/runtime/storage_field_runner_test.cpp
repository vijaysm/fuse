// Fields of a Storage value type (binary32, bfloat16, binary16, posit16, posit32) through the
// Runner and a real remap (plans/2026-10-22-universal-number-types.md, stage
// U3). The components hold their arrays in the narrow type; everything the
// coupler does between them is binary64. So the value a consumer receives is the
// binary64 result of the same graph run with binary64 fields, rounded ONCE to the
// consumer's type -- bit for bit -- and a producer's value reaches the remap
// widened exactly. The reference is therefore the same graph with binary64
// components whose producer quantizes through the producer's type.
//
// The binary has a `.device` twin on a device build (FUSE_EXECUTION_SPACE=device
// resolves RunnerConfig's default): host-registered fields then go through a
// device mirror of their image, and the last tests register the fields in device
// memory, so the widening and the rounding run as kernels.

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include "fuse_test_main.hpp"
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/value_traits.hpp>
#include <fuse/num/posit.hpp>
#include <fuse/runtime/runner.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using fuse::bfloat16;
using fuse::binary16;
using fuse::num::posit16;
using fuse::num::posit32;
using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::RunnerConfig;
using fuse::SerialMeshBackend;
using fuse::StepContext;
using fuse::value_traits;
namespace u = fuse::units;

namespace {

constexpr bool kHasDeviceBackend =
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
    true;
#else
    false;
#endif

using DeviceMem = Kokkos::DefaultExecutionSpace::memory_space;

/// The component's array, in host memory or in device memory.
template <class T>
class Store {
 public:
  Store(std::size_t n, bool device) : n_(n), device_(device) {
    if (device_) {
      dev_ = Kokkos::View<T*, DeviceMem>(
          Kokkos::view_alloc(std::string("store"), Kokkos::WithoutInitializing), n);
      std::vector<T> zeros(n, value_traits<T>::from_binary64(0.0));
      write(zeros);
    } else {
      host_.assign(n, value_traits<T>::from_binary64(0.0));
    }
  }
  [[nodiscard]] T* data() { return device_ ? dev_.data() : host_.data(); }
  [[nodiscard]] fuse::MemorySpace space() const {
    return device_ ? fuse::MemorySpace::Device : fuse::MemorySpace::Host;
  }
  void write(const std::vector<T>& v) {
    if (device_) {
      Kokkos::deep_copy(dev_, Kokkos::View<const T*, Kokkos::HostSpace,
                                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(v.data(), v.size()));
    } else {
      host_ = v;
    }
  }
  [[nodiscard]] std::vector<T> read() const {
    if (!device_) return host_;
    std::vector<T> out(n_, value_traits<T>::from_binary64(0.0));
    Kokkos::deep_copy(Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                          out.data(), out.size()),
                      dev_);
    return out;
  }
  [[nodiscard]] std::vector<double> read_binary64() const {
    std::vector<double> out;
    for (const T& x : read()) out.push_back(value_traits<T>::to_binary64(x));
    return out;
  }

 private:
  std::size_t n_;
  bool device_;
  std::vector<T> host_;
  Kokkos::View<T*, DeviceMem> dev_;
};

/// The value the producer computes for cell `i` at `step`: not representable in
/// a 16-bit format, positive, and within the normal range of all of them.
double pattern(std::size_t i, int step, double scale) {
  const double x = static_cast<double>(i);
  return scale * (1.0 + 0.37 * x + 0.1 * step + 0.013 * x * x);
}

/// `value` rounded to Q and widened back: what a Q-typed producer holds.
template <class Q>
double through(double value) {
  return value_traits<Q>::to_binary64(value_traits<Q>::from_binary64(value));
}

/// Writes pattern() rounded through Q, stored as T, on every advance.
template <class T, class Q>
class Producer final : public fuse::Component {
 public:
  Producer(std::size_t cells, double scale, bool device, bool hold = false)
      : store_(cells, device), cells_(cells), scale_(scale), hold_(hold) {}
  /// The application writes the field itself, outside any advance().
  void poke(const std::vector<double>& values) {
    std::vector<T> v;
    for (const double x : values) v.push_back(value_traits<T>::from_binary64(x));
    store_.write(v);
  }
  [[nodiscard]] std::string name() const override { return "producer"; }
  fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::BasicFieldRef<T>(store_.data(), fuse::Layout{1, 0, cells_, 1},
                                                   store_.space()));
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override {
    ++steps_;
    if (hold_) return fuse::Result<void>::success();
    std::vector<T> v;
    for (std::size_t i = 0; i < cells_; ++i) {
      v.push_back(value_traits<T>::from_binary64(through<Q>(pattern(i, steps_, scale_))));
    }
    store_.write(v);
    return fuse::Result<void>::success();
  }

 private:
  Store<T> store_;
  std::size_t cells_;
  double scale_;
  bool hold_;
  int steps_ = 0;
};

template <class T>
class Consumer final : public fuse::Component {
 public:
  Consumer(std::size_t cells, bool device) : store_(cells, device), cells_(cells) {}
  [[nodiscard]] std::string name() const override { return "consumer"; }
  fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::BasicFieldRef<T>(store_.data(), fuse::Layout{1, 0, cells_, 1},
                                                   store_.space()));
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override {
    seen_.push_back(store_.read_binary64());
    return fuse::Result<void>::success();
  }
  [[nodiscard]] const Store<T>& store() const { return store_; }
  /// What the field held at the start of each of its steps.
  [[nodiscard]] const std::vector<std::vector<double>>& seen() const { return seen_; }

 private:
  std::vector<std::vector<double>> seen_;

  Store<T> store_;
  std::size_t cells_;
};

struct Outcome {
  std::vector<double> consumer;           // widened
  std::vector<std::vector<double>> seen;  // the consumer's field at the start of each step
  std::vector<fuse::StorageFieldStats> storage;
  bool ran = false;
};

/// Producer on `src` cells with period `src_dt`, consumer on `dst` cells with
/// period 1, joined by one channel and the backend's remap between two line
/// meshes of different resolution, so the remap is not the identity.
template <class PT, class PQ, class CT>
Outcome run(fuse::AccumOp accum, Rational src_dt, bool device_fields, bool measure,
            double scale = 1.0, std::size_t src = 8, std::size_t dst = 5, int periods = 1,
            Rational dst_dt = Rational{1}, bool hold = false,
            const std::function<void(int, Producer<PT, PQ>&)>& between = {}) {
  CouplingGraph g;
  SerialMeshBackend backend;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:" + std::to_string(src)});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:" + std::to_string(dst)});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{.name = "producer", .ranks = fuse::RankSet{0, 1}, .interval = src_dt});
  const auto c = g.add_component(fuse::ComponentDecl{.name = "consumer", .ranks = fuse::RankSet{1, 1}, .interval = dst_dt});
  const auto s = g.add_field(p, fuse::FieldSpec{.name = "out", .region = r1, .components = 1, .units = u::kKelvin, .intent = fuse::Intent::Provides});
  const auto t = g.add_field(c, fuse::FieldSpec{.name = "in", .region = r2, .components = 1, .units = u::kKelvin, .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", s, t);
  if (accum != fuse::AccumOp::Instant) g.channel(ch).accumulate(accum);
  EXPECT_TRUE(backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(backend.load(g.mesh(m2)).ok());
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  auto prod = std::make_unique<Producer<PT, PQ>>(src, scale, device_fields, hold);
  Producer<PT, PQ>* prod_ptr = prod.get();
  auto cons = std::make_unique<Consumer<CT>>(dst, device_fields);
  const Consumer<CT>* cons_ptr = cons.get();
  components.add(std::move(prod));
  components.add(std::move(cons));

  RunnerConfig cfg;  // Default: FUSE_EXECUTION_SPACE decides, the `.device` twin sets it
  if (device_fields) cfg.execution_space = RunnerConfig::RunnerExecutionSpace::Device;
  cfg.measure_storage_rounding = measure;
  Runner runner{g, lowered.dag, sched.plan, backend};
  FieldRegistry registry;
  Outcome out;
  for (auto& comp : components) {
    auto r = comp.register_fields(registry, g);
    EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
  }
  const auto init = runner.initialize(components, registry, cfg);
  EXPECT_TRUE(init.ok()) << (init.ok() ? "" : init.error().message);
  if (!init.ok()) return out;
  for (int k = 0; k < periods; ++k) {
    if (between) between(k, *prod_ptr);
    const auto rp = runner.run_period(components);
    EXPECT_TRUE(rp.ok()) << (rp.ok() ? "" : rp.error().message);
  }
  out.consumer = cons_ptr->store().read_binary64();
  out.seen = cons_ptr->seen();
  out.storage = runner.stats().storage_fields;
  out.ran = true;
  return out;
}

std::uint64_t bits(double x) {
  std::uint64_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}

/// The consumer must hold the binary64 reference rounded once to CT, bit for bit.
template <class PT, class PQ, class CT>
void expect_single_rounding(fuse::AccumOp accum, Rational src_dt, bool device_fields,
                            int periods = 1) {
  const Outcome got = run<PT, PQ, CT>(accum, src_dt, device_fields, false, 1.0, 8, 5, periods);
  const Outcome ref = run<double, PQ, double>(accum, src_dt, device_fields, false, 1.0, 8, 5, periods);
  ASSERT_TRUE(got.ran && ref.ran);
  ASSERT_EQ(got.consumer.size(), ref.consumer.size());
  bool moved = false;
  for (std::size_t j = 0; j < ref.consumer.size(); ++j) {
    EXPECT_EQ(bits(got.consumer[j]), bits(through<CT>(ref.consumer[j]))) << "cell " << j;
    moved = moved || ref.consumer[j] != 0.0;
    // the reference value is not representable in CT, so the rounding was real
    // (not vacuous) in the narrow cases
  }
  EXPECT_TRUE(moved);
}

}  // namespace

// ---------------------------------------------------------------- host fields

TEST(StorageFieldRunner, ABinary32ConsumerHoldsTheBinary64ResultRoundedOnce) {
  expect_single_rounding<float, float, float>(fuse::AccumOp::Instant, Rational{1}, false);
}
TEST(StorageFieldRunner, ABFloat16ConsumerHoldsTheBinary64ResultRoundedOnce) {
  expect_single_rounding<bfloat16, bfloat16, bfloat16>(fuse::AccumOp::Instant, Rational{1}, false);
}
TEST(StorageFieldRunner, ABinary16ConsumerHoldsTheBinary64ResultRoundedOnce) {
  expect_single_rounding<binary16, binary16, binary16>(fuse::AccumOp::Instant, Rational{1}, false);
}

TEST(StorageFieldRunner, APosit16ConsumerHoldsTheBinary64ResultRoundedOnce) {
  expect_single_rounding<posit16, posit16, posit16>(fuse::AccumOp::Instant, Rational{1}, false);
}
TEST(StorageFieldRunner, APosit32ConsumerHoldsTheBinary64ResultRoundedOnce) {
  expect_single_rounding<posit32, posit32, posit32>(fuse::AccumOp::Instant, Rational{1}, false);
}
TEST(StorageFieldRunner, PositsMixWithTheOtherFormatsAcrossTheBoundary) {
  expect_single_rounding<float, float, posit16>(fuse::AccumOp::Instant, Rational{1}, false);
  expect_single_rounding<posit16, posit16, binary16>(fuse::AccumOp::Instant, Rational{1}, false);
  expect_single_rounding<posit32, posit32, double>(fuse::AccumOp::Instant, Rational{1}, false);
  expect_single_rounding<double, posit16, posit32>(fuse::AccumOp::Instant, Rational{1}, false);
  expect_single_rounding<posit16, posit16, posit16>(fuse::AccumOp::Mean, Rational{1, 4}, false);
  expect_single_rounding<posit32, posit32, posit16>(fuse::AccumOp::Sum, Rational{1, 4}, false, 3);
}

TEST(StorageFieldRunner, FormatsMixAcrossTheBoundaryAndBinary64IsOneOfThem) {
  expect_single_rounding<float, float, bfloat16>(fuse::AccumOp::Instant, Rational{1}, false);
  expect_single_rounding<bfloat16, bfloat16, double>(fuse::AccumOp::Instant, Rational{1}, false);
  expect_single_rounding<double, binary16, binary16>(fuse::AccumOp::Instant, Rational{1}, false);
  // a binary64 producer into a binary16 consumer: the producer is not quantized
  const Outcome got = run<double, double, binary16>(fuse::AccumOp::Instant, Rational{1}, false, false);
  const Outcome ref = run<double, double, double>(fuse::AccumOp::Instant, Rational{1}, false, false);
  for (std::size_t j = 0; j < ref.consumer.size(); ++j) {
    EXPECT_EQ(bits(got.consumer[j]), bits(through<binary16>(ref.consumer[j]))) << j;
  }
}

TEST(StorageFieldRunner, AccumulationOverSubstepsRunsInBinary64AndIsRoundedOnce) {
  for (const auto accum : {fuse::AccumOp::Sum, fuse::AccumOp::Mean}) {
    expect_single_rounding<bfloat16, bfloat16, bfloat16>(accum, Rational{1, 4}, false);
    expect_single_rounding<binary16, binary16, float>(accum, Rational{1, 4}, false);
    expect_single_rounding<float, float, binary16>(accum, Rational{1, 4}, false, 3);
  }
}

TEST(StorageFieldRunner, TheRunReportsEachConvertedFieldAndItsUnitRoundoff) {
  const Outcome o = run<bfloat16, bfloat16, binary16>(fuse::AccumOp::Instant, Rational{1}, false, false);
  ASSERT_TRUE(o.ran);
  ASSERT_EQ(o.storage.size(), 2u);
  const fuse::StorageFieldStats* provided = nullptr;
  const fuse::StorageFieldStats* delivered = nullptr;
  for (const auto& s : o.storage) (s.delivered ? delivered : provided) = &s;
  ASSERT_TRUE(provided && delivered);
  EXPECT_EQ(provided->format, "bfloat16");
  EXPECT_EQ(provided->significand_bits, 8);
  EXPECT_EQ(provided->unit_roundoff, std::ldexp(1.0, -8));
  EXPECT_GE(provided->widenings, 1);
  EXPECT_EQ(provided->narrowings, 0);
  EXPECT_EQ(delivered->format, "binary16");
  EXPECT_EQ(delivered->significand_bits, 11);
  EXPECT_EQ(delivered->unit_roundoff, std::ldexp(1.0, -11));
  EXPECT_EQ(delivered->narrowings, 1);  // one delivery in one period
  EXPECT_EQ(delivered->values, 0);      // not measured unless asked
  EXPECT_EQ(delivered->field.find("consumer"), 0u);
}

TEST(StorageFieldRunner, AnOrdinaryBinary64RunReportsNoStorageFields) {
  const Outcome o = run<double, double, double>(fuse::AccumOp::Instant, Rational{1}, false, true);
  ASSERT_TRUE(o.ran);
  EXPECT_TRUE(o.storage.empty());
}

TEST(StorageFieldRunner, MeasuredRoundingStaysWithinTheUnitRoundoffInTheNormalRange) {
  struct Case { int bits; double u; };
  const Outcome b16 = run<double, double, binary16>(fuse::AccumOp::Instant, Rational{1}, false, true);
  const Outcome bf = run<double, double, bfloat16>(fuse::AccumOp::Instant, Rational{1}, false, true);
  const Outcome f32 = run<double, double, float>(fuse::AccumOp::Instant, Rational{1}, false, true);
  const Outcome p16 = run<double, double, posit16>(fuse::AccumOp::Instant, Rational{1}, false, true);
  const Outcome p32 = run<double, double, posit32>(fuse::AccumOp::Instant, Rational{1}, false, true);
  for (const auto* o : {&b16, &bf, &f32, &p16, &p32}) {
    ASSERT_TRUE(o->ran);
    const fuse::StorageFieldStats* d = nullptr;
    for (const auto& s : o->storage) if (s.delivered) d = &s;
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->values, 5);
    EXPECT_EQ(d->overflow, 0);
    EXPECT_EQ(d->underflow, 0);
    EXPECT_GT(d->max_relative_change, 0.0);
    EXPECT_LE(d->max_relative_change, d->unit_roundoff);
  }
}

TEST(StorageFieldRunner, ValuesBeyondAPositsRangeAreCountedAsOverflowAndClamped) {
  // posit16 stops at 2^56, about 7.2e16: the values here are 1e18 and more
  const Outcome o = run<double, double, posit16>(fuse::AccumOp::Instant, Rational{1}, false, true,
                                                 1.0e18, 8, 5, 3);
  ASSERT_TRUE(o.ran);
  const fuse::StorageFieldStats* d = nullptr;
  for (const auto& s : o.storage) if (s.delivered) d = &s;
  ASSERT_NE(d, nullptr);
  EXPECT_EQ(d->format, "posit16");
  EXPECT_EQ(d->significand_bits, 12);
  EXPECT_EQ(d->values, 15);
  EXPECT_EQ(d->overflow, 15);
  EXPECT_EQ(d->underflow, 0);
  for (const double v : o.consumer) EXPECT_EQ(v, posit16::maxpos().to_double());
}

TEST(StorageFieldRunner, ValuesBeyondTheFormatsRangeAreCountedAsOverflow) {
  // three periods: the counts accumulate over deliveries
  const Outcome o = run<double, double, binary16>(fuse::AccumOp::Instant, Rational{1}, false, true,
                                                  1.0e5, 8, 5, 3);
  ASSERT_TRUE(o.ran);
  const fuse::StorageFieldStats* d = nullptr;
  for (const auto& s : o.storage) if (s.delivered) d = &s;
  ASSERT_NE(d, nullptr);
  EXPECT_EQ(d->narrowings, 3);
  EXPECT_EQ(d->values, 15);
  EXPECT_EQ(d->overflow, 15);
  for (const double v : o.consumer) EXPECT_TRUE(std::isinf(v));
}

TEST(StorageFieldRunner, AValueTheApplicationWritesBetweenPeriodsIsWhatTheRemapReads) {
  // The producer holds (its advance() writes nothing); the application writes
  // the field before each period. Component code runs in every period, so the
  // image is refreshed from what the application wrote.
  std::vector<std::vector<double>> written;
  const auto between = [&](int k, Producer<bfloat16, bfloat16>& p) {
    std::vector<double> v;
    for (std::size_t i = 0; i < 8; ++i) v.push_back(pattern(i, 100 + k, 1.0));
    written.push_back(v);
    p.poke(v);
  };
  const Outcome got = run<bfloat16, bfloat16, double>(fuse::AccumOp::Instant, Rational{1}, false, false,
                                                      1.0, 8, 5, 2, Rational{1, 2}, true, between);
  ASSERT_TRUE(got.ran);
  // reference: the same graph with binary64 fields, the same poked values
  const auto between_ref = [&](int k, Producer<double, bfloat16>& p) {
    std::vector<double> v;
    for (std::size_t i = 0; i < 8; ++i) v.push_back(through<bfloat16>(pattern(i, 100 + k, 1.0)));
    p.poke(v);
  };
  const Outcome ref = run<double, bfloat16, double>(fuse::AccumOp::Instant, Rational{1}, false, false,
                                                    1.0, 8, 5, 2, Rational{1, 2}, true, between_ref);
  ASSERT_TRUE(ref.ran);
  ASSERT_EQ(got.seen.size(), ref.seen.size());
  ASSERT_FALSE(got.seen.empty());
  for (std::size_t k = 0; k < ref.seen.size(); ++k) {
    ASSERT_EQ(got.seen[k].size(), ref.seen[k].size());
    for (std::size_t j = 0; j < ref.seen[k].size(); ++j) {
      EXPECT_EQ(bits(got.seen[k][j]), bits(ref.seen[k][j])) << "step " << k << " cell " << j;
    }
  }
}

// -------------------------------------------------------------- device fields

TEST(StorageFieldRunner, FieldsRegisteredInDeviceMemoryAreConvertedByKernels) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "this build has no device backend";
  expect_single_rounding<bfloat16, bfloat16, bfloat16>(fuse::AccumOp::Instant, Rational{1}, true);
  expect_single_rounding<float, float, binary16>(fuse::AccumOp::Instant, Rational{1}, true);
  expect_single_rounding<binary16, binary16, double>(fuse::AccumOp::Mean, Rational{1, 4}, true);
  expect_single_rounding<double, bfloat16, float>(fuse::AccumOp::Sum, Rational{1, 4}, true, 2);
  expect_single_rounding<posit16, posit16, posit32>(fuse::AccumOp::Instant, Rational{1}, true);
  expect_single_rounding<float, float, posit16>(fuse::AccumOp::Mean, Rational{1, 4}, true);
}

TEST(StorageFieldRunner, MeasuredRoundingOnTheDeviceMatchesTheHost) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "this build has no device backend";
  const Outcome dev = run<double, double, binary16>(fuse::AccumOp::Instant, Rational{1}, true, true);
  const Outcome host = run<double, double, binary16>(fuse::AccumOp::Instant, Rational{1}, false, true);
  ASSERT_TRUE(dev.ran && host.ran);
  const fuse::StorageFieldStats *a = nullptr, *b = nullptr;
  for (const auto& s : dev.storage) if (s.delivered) a = &s;
  for (const auto& s : host.storage) if (s.delivered) b = &s;
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->values, b->values);
  EXPECT_EQ(a->max_relative_change, b->max_relative_change);
}
