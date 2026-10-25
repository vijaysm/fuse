# Mutation evidence — `base`

Run against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Outcome | Killed by |
|---|---|---|---|
| `rational-gcd-skipped` — normalise() forgets to divide by the gcd, so 6/8 stays 6/8 | `include/fuse/base/rational.hpp` | **KILLED** (test static_assert fails) | `base.rational` |
| `rational-compare-64bit` — operator<=> compares in int64 and overflows above sqrt(INT64_MAX) | `include/fuse/base/rational.hpp` | **KILLED** (test fails) | `base.rational` |
| `rational-record-accepts-unreduced` — from_record() repairs 2/4 to 1/2 instead of rejecting it (I4) | `include/fuse/base/rational.hpp` | **KILLED** (test fails) | `base.rational` |
| `units-exponent-wraps` — exponent narrowing wraps silently instead of rejecting (I4/I6) | `include/fuse/base/units.hpp` | **KILLED** (test fails) | `base.units` |
| `units-divide-adds` — operator/ adds exponents instead of subtracting | `include/fuse/base/units.hpp` | **KILLED** (test static_assert fails) | `base.units` |
| `time-level-ignores-iterate` — TimeLevel equality ignores the iterate index | `include/fuse/base/time_level.hpp` | **KILLED** (test fails) | `base.time_level` |
| `diagnostics-first-failure-only` — Diagnostics keeps only the first defect (the U2 anti-pattern) | `include/fuse/base/result.hpp` | **KILLED** (test fails) | `base.result` |
| `version-accepts-leading-zero` — parse accepts 01.2.3, breaking canonical round-trip | `src/base/version.cpp` | **KILLED** (test fails) | `base.version` |
| `version-compatible-across-major` — compatibility ignores the major version | `src/base/version.cpp` | **KILLED** (test fails) | `base.version` |
| `bytes-no-length-prefix` — put_string drops the length prefix so adjacent strings collide | `include/fuse/base/bytes.hpp` | **KILLED** (test fails) | `base.bytes` |
| `error-assert-compiled-out` — FUSE_ASSERT becomes a no-op (the `assert` it replaces) | `include/fuse/base/error.hpp` | **KILLED** (test fails) | `base.error` |
| `error-slices-to-base` — raise() throws the Error base, so catch(const Refusal&) is dead | `include/fuse/base/error.hpp` | **KILLED** (test fails) | `base.error` |
| `rational-try-skips-the-fit-check` — the non-throwing arithmetic stops checking that the reduced result fits, so an overflowing value narrows silently instead of reporting -- and the throwing operators inherit it, since they are layered on this | `include/fuse/base/rational.hpp` | **KILLED** (test fails) | `base.rational` |
| `rational-lcm-denominator-overflows` — lcm_denominator() reverts to the direct int64_t std::lcm call, reintroducing the overflow gcd_rational's sibling fix removed (R-15) | `include/fuse/base/rational.hpp` | **KILLED** (test fails) | `base.rational` |

**14 killed / 14 total.**
