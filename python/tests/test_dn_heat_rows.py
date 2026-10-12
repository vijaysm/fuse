"""The dn_heat driver test: byte-for-byte agreement with examples/dn_heat/main.cpp.

Per the owner's ruling (2026-10-12): examples/dn_heat/dn_component.hpp's DirichletSlab/
NeumannSlab are reimplemented as Python pfuse.Component subclasses (dn_heat_components.py,
copied from the C++ formulas line-for-line, same arithmetic grouping) and driven through the
REAL fuse::Runner via pfuse.Session -- only the physics components are Python; the graph
authoring, the accelerator, the scheduler and the runtime are the identical fuse:: C++ code
examples/dn_heat/main.cpp's binary ran when golden/dn_heat_full_output.txt was captured.

MEASURED FIRST, as instructed, before any assertion was written:

  1. C++'s std::sin()/the kPi literal and Python's math.sin()/math.pi produce BIT-IDENTICAL
     doubles on this machine (arm64 macOS, AppleClang, both built/run with -ffp-contract=off)
     for the t values examples/dn_heat/main.cpp's own 20-coupling-cycle grid actually uses --
     checked directly with a byte-for-byte hex-of-the-IEEE-754-bit-pattern comparison between
     a standalone C++ probe and this interpreter (see this track's structured report for the
     exact hex values compared).
  2. Given (1), every non-Newton row this file runs (explicit/picard/aitken/iqn-ils/iqn+qr2/
     severed/expl-long/iqn-long) was run once, by hand, before this file was written, and its
     formatted row AND its final-period residual history were found to match
     golden/dn_heat_full_output.txt byte for byte, with no exceptions and no near-misses that
     would have needed a derived tolerance instead.

Because every row measured here came out EXACTLY equal, every assertion below is an exact
string/array equality, not a tolerance -- there is nothing to derive a bound for. The three
Newton rows (newton/newton-repeat/newton-nl) and the bit-reproducibility "repeat" row are
NOT run here: a Newton-solved region needs a native ImplicitComponent, which no
Python-authored graph can supply in pfuse v1 (docs/dev/python-surface.md, test_newton.py).
"""

import pathlib

import pytest

pfuse = pytest.importorskip("pfuse", reason="Stage B has not built the _pfuse extension yet")
import dn_heat_components as dn  # noqa: E402  (after the importorskip, deliberately)

GOLDEN = pathlib.Path(__file__).parent / "golden" / "dn_heat_full_output.txt"


def _reference_line(label):
    """Grep the exact reference row for `label` out of the golden file -- never retyped."""
    prefix = "  {:<9} ".format(label)
    for line in GOLDEN.read_text().splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError("no reference row for {!r} in {}".format(label, GOLDEN))


def _reference_residual_line(label):
    prefix = "  {:<9} ".format(label)
    seen_table = False
    for line in GOLDEN.read_text().splitlines():
        if line.lstrip().startswith("residual history"):
            seen_table = True
            continue
        if seen_table and line.startswith(prefix):
            return line
    raise AssertionError("no residual-history row for {!r} in {}".format(label, GOLDEN))


PICARD_CFG = pfuse.AccelConfig(kind=pfuse.AccelKind.None_)
AITKEN_CFG = pfuse.AccelConfig(kind=pfuse.AccelKind.Aitken, initial_relaxation=0.5)
IQN_CFG = pfuse.AccelConfig(kind=pfuse.AccelKind.IQN_ILS, initial_relaxation=0.5, reuse=8,
                            filter=pfuse.FilterKind.None_)
IQN_QR2_CFG = pfuse.AccelConfig(kind=pfuse.AccelKind.IQN_ILS, initial_relaxation=0.5, reuse=8,
                                filter=pfuse.FilterKind.QR2, limit=1.0e-2)


class TestNonNewtonRowsMatchByteForByte:
    def test_explicit(self):
        o = dn.run_case(PICARD_CFG, 1, dn.PERIODS, dn.Wiring.Connected)
        assert dn.format_row("explicit", o) == _reference_line("explicit")

    def test_picard(self):
        o = dn.run_case(PICARD_CFG, dn.MAX_ITERATIONS, dn.PERIODS, dn.Wiring.Connected)
        assert dn.format_row("picard", o) == _reference_line("picard")
        assert dn.format_residuals("picard", o.last_residuals) == _reference_residual_line("picard")

    def test_aitken(self):
        o = dn.run_case(AITKEN_CFG, dn.MAX_ITERATIONS, dn.PERIODS, dn.Wiring.Connected)
        assert dn.format_row("aitken", o) == _reference_line("aitken")
        assert dn.format_residuals("aitken", o.last_residuals) == _reference_residual_line("aitken")

    def test_iqn_ils(self):
        o = dn.run_case(IQN_CFG, dn.MAX_ITERATIONS, dn.PERIODS, dn.Wiring.Connected)
        assert dn.format_row("iqn-ils", o) == _reference_line("iqn-ils")
        assert dn.format_residuals("iqn-ils", o.last_residuals) == _reference_residual_line("iqn-ils")

    def test_iqn_plus_qr2(self):
        o = dn.run_case(IQN_QR2_CFG, dn.MAX_ITERATIONS, dn.PERIODS, dn.Wiring.Connected)
        assert dn.format_row("iqn+qr2", o) == _reference_line("iqn+qr2")

    def test_severed(self):
        o = dn.run_case(PICARD_CFG, dn.MAX_ITERATIONS, dn.PERIODS, dn.Wiring.Severed)
        assert dn.format_row("severed", o) == _reference_line("severed")
        assert dn.format_residuals("severed", o.last_residuals) == _reference_residual_line("severed")

    def test_explicit_long(self):
        o = dn.run_case(PICARD_CFG, 1, dn.LONG_PERIODS, dn.Wiring.Connected)
        assert dn.format_row("expl-long", o) == _reference_line("expl-long")

    def test_iqn_ils_long(self):
        o = dn.run_case(IQN_CFG, dn.MAX_ITERATIONS, dn.LONG_PERIODS, dn.Wiring.Connected)
        assert dn.format_row("iqn-long", o) == _reference_line("iqn-long")


class TestNewtonRowsAreOutOfScope:
    """Explicit, not implicit: there is no test above for 'newton'/'newton-repeat'/
    'newton-nl' because no Python-authored graph can supply the native ImplicitComponent a
    Newton-solved region needs (test_newton.py proves the refusal that makes this true)."""

    def test_the_golden_file_still_has_newton_rows_this_file_does_not_cover(self):
        """A sanity check on the fixture itself, not on pfuse: if golden/dn_heat_full_output.txt
        ever stopped having a 'newton' row, the claim in this file's own docstring (that Newton
        rows are deliberately excluded, not silently missing) would need re-examining."""
        text = GOLDEN.read_text()
        assert "\n  newton " in text
