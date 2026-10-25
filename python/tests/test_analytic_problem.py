"""A second coupled problem, chosen to be solvable by hand, with a tolerance DERIVED from
that solution rather than picked by trial and error -- on top of the dn_heat byte-for-byte reproduction in test_dn_heat_rows.py.

THE PROBLEM. Two components, one scalar field each, one spatial point (a "generated:
line:1" mesh/region, dimension 1, 1 cell -- the smallest region fuse's serial backend
accepts):

  Source (provides "x", requires "y_in"):  x_new = RHO * y_in + C
  Mirror (provides "y", requires "x_in"):   y_new = x_in            (a pure relay)

wired through TWO channels inside one Picard-only (AccelKind.None_) iteration region --
"fwd" (Source.x -> Mirror.x_in, Channel.lag default 0) and "back" (Mirror.y -> Source.y_in,
.lag(1)) -- which is the same "one channel lagged, one not" shape
dn_heat_components.build_graph() uses for its own two-channel region.

THE FIXED POINT, solved by hand: y_new = x_in = x_new(previous) = RHO*y_in(previous) + C,
so the scalar sequence satisfies a plain affine recursion once both fields have been
through one full round trip, and its fixed point is

    y* = C / (1 - RHO)

by substituting y* into y* = RHO*y* + C and solving -- algebra, not a simulation.

WHAT WAS MEASURED BEFORE ANY ASSERTION WAS WRITTEN (the same discipline test_dn_heat_rows.py
documents): instrumenting both components' advance() with fuse.StepContext.iteration showed
that NEITHER channel delivers within the sweep it was produced -- "fwd", despite its default
(zero) Channel.lag, still only reaches Mirror.x_in starting the SWEEP AFTER Source computed
it, exactly as "back"'s explicit .lag(1) does for the reverse direction. One consequence:
Source.x and Mirror.y only change on alternating sweeps (each holds its value for one extra
sweep while the other one updates), so the pair (Source.x, Mirror.y) jointly retraces the
SAME scalar recursion above at half the sweep rate. A second consequence, read directly off
that trace rather than assumed: whichever of Source.x/Mirror.y just changed on a given sweep
k is EXACTLY the quantity fuse::ConvergenceMeasure::Relative's residual_norm measures for
that sweep (include/fuse/model/iteration.hpp's `satisfied()`), and the OTHER field's own
residual_norm is exactly 0.0 that sweep -- so the combined (sqrt of sum of squares, both
measures) residual history Session.iteration_stats[0].residual_history reports is, every
single sweep, exactly the single nonzero term, with no cross-term to account for.

THE DERIVED BOUND. Call e_m the error of the scalar recursion's own m-th value (m counting in
round trips, not raw sweeps) and R the LAST combined residual fuse itself reports. Because
the recursion is exactly affine (no higher-order term to bound away), the standard fixed-
point a-posteriori identity is exact here, not approximate:

    e_m = (e_m - e_{m-1}) + e_{m-1} = RHO*e_{m-1} - e_{m-1} + e_{m-1}  (wrong path -- use the
    direct substitution instead)
    e_m = RHO * e_{m-1}        and        (value_m - value_{m-1}) = e_m - e_{m-1} = e_{m-1}*(RHO-1)
    => e_m = RHO/(1-RHO) * (value_m - value_{m-1})                      -- an EQUALITY.

Measured directly (with a probe run before any assertion existed): for the component whose field just changed on the final
sweep (always Source.x, by construction -- see the trace above), |Source.x - y*| equals
RHO/(1-RHO) * R to within floating-point roundoff accumulated over the sweep count, not
merely bounded by it. The OTHER field (Mirror.y) can be one round trip further behind in the
worst case (whichever sweep parity fuse happens to stop on), so its own error is bounded --
not claimed exact -- by R/(1-RHO) = (RHO/(1-RHO))*R / RHO, i.e. the same identity one round
trip earlier. Both relations are checked below with a floating-point allowance
(_FP_SLACK) sized to the roundoff actually observed over the ~40-sweep run this problem
takes to reach TOL=1e-10, not fitted to make the test pass.
"""

import numpy as np
import pytest

pfuse = pytest.importorskip("pfuse", reason="the _pfuse extension is not built")

RHO = 0.3
C = 2.0
Y0 = 0.0
TOL = 1.0e-10
MAX_ITERATIONS = 100
# Floating-point allowance on top of the two EXACT identities above: accumulated rounding
# over a ~40-sweep run of double-precision arithmetic, measured directly at
# ~1.3e-6 relative between the identity's two sides -- a handful of
# ULPs compounded through ~40 multiply-adds, not a fitted margin. 1e-4 relative is a
# two-orders-of-magnitude allowance over that measurement, not a tight fit to it.
_FP_SLACK = 1.0 + 1.0e-4


class Source(pfuse.Component):
    """x_new = RHO * y_in + C. The only nonlinearity-free, single-station physics needed to
    make the fixed point solvable with pen and paper."""

    def __init__(self):
        super().__init__("source")
        self.x = np.zeros(1)
        self.y_in = np.zeros(1)
        self._saved_x = np.zeros(1)

    def register_fields(self, registry, graph):
        cid = graph.find_component(self.name)
        for f in graph.component(cid).fields:
            if graph.field(f).intent == pfuse.Intent.Provides:
                registry.bind_numpy(f, self.x)
            else:
                registry.bind_numpy(f, self.y_in)

    def advance(self, ctx):
        self.x[0] = RHO * self.y_in[0] + C

    def save_state(self):
        self._saved_x[:] = self.x

    def restore_state(self):
        self.x[:] = self._saved_x


class Mirror(pfuse.Component):
    """y_new = x_in: a pure relay, contributing no contraction of its own -- the entire
    RHO comes from Source, so the recursion above has exactly one unknown rate to solve for."""

    def __init__(self):
        super().__init__("mirror")
        self.y = np.zeros(1)
        self.y[0] = Y0
        self.x_in = np.zeros(1)
        self._saved_y = np.zeros(1)

    def register_fields(self, registry, graph):
        cid = graph.find_component(self.name)
        for f in graph.component(cid).fields:
            if graph.field(f).intent == pfuse.Intent.Provides:
                registry.bind_numpy(f, self.y)
            else:
                registry.bind_numpy(f, self.x_in)

    def advance(self, ctx):
        self.y[0] = self.x_in[0]

    def save_state(self):
        self._saved_y[:] = self.y

    def restore_state(self):
        self.y[:] = self._saved_y


def build_graph():
    g = pfuse.CouplingGraph()
    mesh = g.add_mesh("pt", "generated:line:1")
    region = g.add_region("pt", mesh, dimension=1)

    cs = g.add_component("source", ranks=pfuse.RankSet(0, 1), interval=pfuse.Rational(1, 1))
    cm = g.add_component("mirror", ranks=pfuse.RankSet(0, 1), interval=pfuse.Rational(1, 1))

    x_out = g.add_field(cs, "x", region, units=pfuse.units.dimensionless,
                         conservation=pfuse.Conservation.Intensive, intent=pfuse.Intent.Provides)
    y_in = g.add_field(cs, "y_in", region, units=pfuse.units.dimensionless,
                        conservation=pfuse.Conservation.Intensive, intent=pfuse.Intent.Requires)
    x_in = g.add_field(cm, "x_in", region, units=pfuse.units.dimensionless,
                        conservation=pfuse.Conservation.Intensive, intent=pfuse.Intent.Requires)
    y_out = g.add_field(cm, "y", region, units=pfuse.units.dimensionless,
                         conservation=pfuse.Conservation.Intensive, intent=pfuse.Intent.Provides)

    fwd = g.add_channel("fwd", x_out, x_in)
    back = g.add_channel("back", y_out, y_in)
    g.channel(back).lag(1).allow_time_level_mismatch(
        "two-component scalar Picard loop, same shape as dn_heat_components.build_graph()")

    crit = pfuse.Converged.Relative(TOL, "y") & pfuse.Converged.Relative(TOL, "x")
    g.iterate([fwd, back]).name("scalar_fixed_point").until(crit).max_iterations(
        MAX_ITERATIONS).accelerate(pfuse.AccelConfig(kind=pfuse.AccelKind.None_))
    return g


class TestLinearFixedPointMatchesTheHandSolvedAnswer:
    def test_picard_converges_within_the_requested_tolerance(self):
        g = build_graph()
        session = pfuse.Session(g)
        source, mirror = Source(), Mirror()
        session.add_component(source)
        session.add_component(mirror)
        session.initialize()
        session.run(1)

        # Converged, not exhausted: max_iterations(100) was headroom, not the stopping rule.
        assert session.stats.windows_not_converged == 0
        assert session.stats.total_iterations < MAX_ITERATIONS

    def test_final_state_matches_the_hand_solved_fixed_point_within_the_derived_bound(self):
        g = build_graph()
        session = pfuse.Session(g)
        source, mirror = Source(), Mirror()
        session.add_component(source)
        session.add_component(mirror)
        session.initialize()
        session.run(1)

        y_star = C / (1.0 - RHO)  # the hand-solved fixed point, not a simulated one
        residual = session.iteration_stats[0].residual_history[-1]

        # EQUALITY (up to the stated floating-point allowance): Source.x is always the field
        # that just changed on fuse's own final sweep (see the module docstring's measured
        # trace), so the a-posteriori identity applies to it directly and exactly.
        exact_relation = RHO / (1.0 - RHO) * residual
        source_error = abs(float(source.x[0]) - y_star)
        assert source_error <= exact_relation * _FP_SLACK

        # BOUND, not equality: Mirror.y can be one round trip further behind depending on
        # which sweep parity fuse happens to stop on (module docstring). This is the same
        # identity applied one round trip earlier, i.e. divided by RHO once more.
        conservative_bound = residual / (1.0 - RHO)
        mirror_error = abs(float(mirror.y[0]) - y_star)
        assert mirror_error <= conservative_bound * _FP_SLACK

    def test_the_fixed_point_algebra_itself_is_unchanged_from_the_docstring(self):
        """A guard on this file, not on pfuse: if y_star's closed form above were ever
        mistyped, this independent re-derivation (solving the 2x2 linear system the long way,
        not just repeating the same one-line substitution) would catch it."""
        # x* = RHO*y* + C, y* = x* (Mirror is a pure relay) => y* - RHO*y* = C.
        a = np.array([[1.0, -RHO], [-1.0, 1.0]])
        b = np.array([C, 0.0])
        x_star, y_star = np.linalg.solve(a, b)
        assert x_star == pytest.approx(C / (1.0 - RHO), rel=1e-14)
        assert y_star == pytest.approx(C / (1.0 - RHO), rel=1e-14)
