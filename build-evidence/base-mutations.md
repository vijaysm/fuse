# Mutation evidence — `base`

Run 2026-09-17 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `rational-gcd-skipped` — normalise() forgets to divide by the gcd, so 6/8 stays 6/8 | `include/fuse/base/rational.hpp` | **KILLED** (test static_assert fails) | `base.rational` |
| `rational-compare-64bit` — operator<=> compares in int64 and overflows above sqrt(INT64_MAX) | `include/fuse/base/rational.hpp` | **KILLED** (test fails) | `base.rational` |
| `rational-record-accepts-unreduced` — from_record() repairs 2/4 to 1/2 instead of refusing it (I4) | `include/fuse/base/rational.hpp` | **KILLED** (test fails) | `base.rational` |
| `units-exponent-wraps` — exponent narrowing wraps silently instead of refusing (I4/I6) | `include/fuse/base/units.hpp` | **KILLED** (test fails) | `base.units` |
| `units-divide-adds` — operator/ adds exponents instead of subtracting | `include/fuse/base/units.hpp` | **KILLED** (test static_assert fails) | `base.units` |
| `time-level-ignores-iterate` — TimeLevel equality ignores the iterate index | `include/fuse/base/time_level.hpp` | **KILLED** (test fails) | `base.time_level` |
| `diagnostics-first-failure-only` — Diagnostics keeps only the first defect (the U2 anti-pattern) | `include/fuse/base/result.hpp` | **KILLED** (test fails) | `base.result` |
| `version-accepts-leading-zero` — parse accepts 01.2.3, breaking canonical round-trip | `src/base/version.cpp` | **KILLED** (test fails) | `base.version` |
| `version-compatible-across-major` — compatibility ignores the major version | `src/base/version.cpp` | **KILLED** (test fails) | `base.version` |
| `bytes-no-length-prefix` — put_string drops the length prefix so adjacent strings collide | `include/fuse/base/bytes.hpp` | **KILLED** (test fails) | `base.bytes` |
| `error-assert-compiled-out` — FUSE_ASSERT becomes a no-op (the `assert` it replaces) | `include/fuse/base/error.hpp` | **KILLED** (test fails) | `base.error` |
| `error-slices-to-base` — raise() throws the Error base, so catch(const Refusal&) is dead | `include/fuse/base/error.hpp` | **KILLED** (test fails) | `base.error` |

**12 killed / 12 total.**
