"""Python reimplementations of examples/dn_heat/dn_component.hpp's DirichletSlab/NeumannSlab.

Per the owner's ruling (2026-10-12): the dn_heat byte-for-byte driver test reimplements the
two slabs as Python pfuse.Component subclasses rather than wrapping the existing native
classes in a test-only extension. The formulas below are copied from dn_component.hpp
line-for-line (not approximated), and the arithmetic grouping matches the C++ expressions'
own left-to-right evaluation order exactly, which is what IEEE-754 bit-identical reproduction
with C++ needs -- see docs/dev/python-surface.md's "The dn_heat driver test" and
test_dn_heat_rows.py's own module docstring for the direct sin()/pi bit-pattern comparison
that measured this is a reasonable thing to attempt BEFORE any row comparison runs.

Only the AFFINE case (beta = 0, dn_component.hpp's own default) is reimplemented: this is
what every non-Newton row of examples/dn_heat/main.cpp uses. The nonlinear (beta != 0) and
Newton rows need a native ImplicitComponent, which pfuse v1 does not expose to Python
(docs/dev/python-surface.md) -- see that page's own statement that Python v1 cannot RUN a
Newton-solved region for that reason.
"""

import math

import numpy as np

import pfuse

# --- geometry and materials (dn_component.hpp's own kL1/kL2/kK2/kStrips/kStripRatio) --------
L1 = 1.0
L2 = 1.0
K2 = 1.0
STRIPS = 4
STRIP_RATIO = (0.50, 0.75, 0.90, 0.96)
PI = 3.14159265358979323846  # dn::kPi, bit-identical to math.pi -- see test_dn_heat_rows.py


def strip_ratio(i, n):
    s = (i * STRIPS) // (n if n != 0 else 1)
    return STRIP_RATIO[s if s < STRIPS else STRIPS - 1]


def k1_at(i, n):
    # dn::k1_at: strip_ratio(i, n) * kK2 * kL1 / kL2 -- left-to-right, matching the C++
    # expression's own grouping (all three constants are 1.0, but the grouping is kept for
    # the day they are not).
    return strip_ratio(i, n) * K2 * L1 / L2


def T_outer_1(t):
    return 340.0 + 40.0 * math.sin(2.0 * PI * t)


def T_outer_2(t):
    return 280.0 + 15.0 * math.sin(PI * t + 0.7)


def exact_u(i, n, t):
    r = strip_ratio(i, n)
    return (T_outer_2(t) + r * T_outer_1(t)) / (1.0 + r)


def exact_q(i, n, t):
    return (T_outer_1(t) - T_outer_2(t)) / (L1 / k1_at(i, n) + L2 / K2)


def l2_norm(v):
    dy = 1.0 / len(v)
    s = 0.0
    for x in v:
        s += x * x * dy
    return math.sqrt(s)


def l2_error(v, f):
    dy = 1.0 / len(v)
    s = 0.0
    for i in range(len(v)):
        e = v[i] - f(i)
        s += e * e * dy
    return math.sqrt(s)


def max_error(v, f):
    m = 0.0
    for i in range(len(v)):
        m = max(m, abs(v[i] - f(i)))
    return m


class Wiring:
    Connected = "Connected"
    Severed = "Severed"


class DirichletSlab(pfuse.Component):
    """dn::DirichletSlab at beta=0 (the affine map every non-Newton row uses)."""

    def __init__(self, name, n, wiring=Wiring.Connected):
        super().__init__(name)
        self.n = n
        self.wiring = wiring
        self.q = np.zeros(n)
        self.u_in = np.zeros(n)
        self.decoy = np.zeros(n)
        self.saved_q = np.zeros(n)
        self.time = 0.0
        self.saved_time = 0.0
        self.history = 0.0
        self.saved_history = 0.0

    def register_fields(self, registry, graph):
        cid = graph.find_component(self.name)
        for f in graph.component(cid).fields:
            provides = graph.field(f).intent == pfuse.Intent.Provides
            if provides:
                registry.bind_numpy(f, self.q)
            elif self.wiring == Wiring.Connected:
                registry.bind_numpy(f, self.u_in)
            else:
                registry.bind_numpy(f, self.decoy)

    def advance(self, ctx):
        self.history += 1.0
        t = float(ctx.t_end)
        T1 = T_outer_1(t)
        for i in range(self.n):
            # dn::DirichletSlab::advance: q_[i] = k1_eff(i, u_in_[i]) * (T1 - u_in_[i]) / kL1,
            # and k1_eff(i, T) at beta=0 is k1_at(i,n)*(1.0 + 0.0) == k1_at(i,n) exactly (IEEE
            # 754 x*1.0 == x) -- dn_component.hpp's own
            # Dn.NonlinearAtBetaZeroIsBitIdenticalToTheAffineMap comment.
            self.q[i] = k1_at(i, self.n) * (T1 - self.u_in[i]) / L1
        self.time = t

    def save_state(self):
        self.saved_q[:] = self.q
        self.saved_time = self.time
        self.saved_history = self.history

    def restore_state(self):
        self.q[:] = self.saved_q
        self.time = self.saved_time
        self.history = self.saved_history


class NeumannSlab(pfuse.Component):
    """dn::NeumannSlab."""

    def __init__(self, name, n, wiring=Wiring.Connected, initial_u=None):
        super().__init__(name)
        self.n = n
        self.wiring = wiring
        self.u = np.zeros(n)
        self.q_in = np.zeros(n)
        self.decoy = np.zeros(n)
        self.saved_u = np.zeros(n)
        self.u[:] = T_outer_2(0.0) if initial_u is None else initial_u
        self.time = 0.0
        self.saved_time = 0.0
        self.history = 0.0
        self.saved_history = 0.0

    def register_fields(self, registry, graph):
        cid = graph.find_component(self.name)
        for f in graph.component(cid).fields:
            provides = graph.field(f).intent == pfuse.Intent.Provides
            if provides:
                registry.bind_numpy(f, self.u)
            elif self.wiring == Wiring.Connected:
                registry.bind_numpy(f, self.q_in)
            else:
                registry.bind_numpy(f, self.decoy)

    def advance(self, ctx):
        self.history += 1.0
        t = float(ctx.t_end)
        T2 = T_outer_2(t)
        for i in range(self.n):
            # dn::NeumannSlab::advance: u_[i] = T2 + q_in_[i] * kL2 / kK2.
            self.u[i] = T2 + self.q_in[i] * L2 / K2
        self.time = t

    def save_state(self):
        self.saved_u[:] = self.u
        self.saved_time = self.time
        self.saved_history = self.history

    def restore_state(self):
        self.u[:] = self.saved_u
        self.time = self.saved_time
        self.history = self.saved_history


STATIONS = 32
PERIOD_DEN = 20
PERIODS = 20
LONG_PERIODS = 200
TOL = 1.0e-8
MAX_ITERATIONS = 1500


def build_graph(accel, max_iterations=MAX_ITERATIONS, stations=STATIONS):
    """examples/dn_heat/main.cpp's build_graph(AccelConfig, int) -- the same mesh/region/
    component/field/channel/convergence-criterion setup, authored from Python."""
    g = pfuse.CouplingGraph()
    src = "generated:line:{}".format(stations)
    m1 = g.add_mesh("iface_1", src)
    m2 = g.add_mesh("iface_2", src)
    r1 = g.add_region("iface_1", m1, dimension=1)
    r2 = g.add_region("iface_2", m2, dimension=1)

    cd = g.add_component("dirichlet", ranks=pfuse.RankSet(0, 1),
                        interval=pfuse.Rational(1, PERIOD_DEN))
    cn = g.add_component("neumann", ranks=pfuse.RankSet(1, 1),
                        interval=pfuse.Rational(1, PERIOD_DEN))

    q_out = g.add_field(cd, "q_wall", r1, units=pfuse.units.W_per_m2,
                        conservation=pfuse.Conservation.Extensive,
                        intent=pfuse.Intent.Provides)
    u_in = g.add_field(cd, "u_wall_in", r1, units=pfuse.units.K,
                       conservation=pfuse.Conservation.Intensive,
                       intent=pfuse.Intent.Requires)
    q_in = g.add_field(cn, "q_wall_in", r2, units=pfuse.units.W_per_m2,
                       conservation=pfuse.Conservation.Extensive,
                       intent=pfuse.Intent.Requires)
    u_out = g.add_field(cn, "u_wall", r2, units=pfuse.units.K,
                        conservation=pfuse.Conservation.Intensive,
                        intent=pfuse.Intent.Provides)

    flux = g.add_channel("flux", q_out, q_in)
    # Both flux fields are Extensive, and validate() requires a channel into an Extensive
    # field to declare that its transfer conserves it (E-CONN-CONSERVATION); between two
    # coincident meshes the remap is the identity, which does. Same as the C++ example.
    g.channel(flux).conservative()
    temp = g.add_channel("wall_temperature", u_out, u_in)
    g.channel(temp).lag(1).allow_time_level_mismatch(
        "resolved by the Dirichlet-Neumann implicit coupling group")

    crit = pfuse.Converged.Relative(TOL, "u_wall") & pfuse.Converged.Relative(TOL, "q_wall")
    g.iterate([flux, temp]).name("dirichlet_neumann").until(crit).max_iterations(
        max_iterations).accelerate(accel)
    return g


class Outcome:
    def __init__(self):
        self.periods = 0
        self.sweeps = 0
        self.not_converged = 0
        self.u_final = None
        self.q_final = None
        self.u_l2 = 0.0
        self.u_max = 0.0
        self.q_l2 = 0.0
        self.q_max = 0.0
        self.last_residuals = []


def run_case(accel, max_iterations, periods, wiring, stations=STATIONS):
    """examples/dn_heat/main.cpp's run_case(AccelConfig, ...), authored from Python and
    driven through the REAL fuse::Runner via pfuse.Session -- only the physics components
    are Python; the graph, the accelerator, the scheduler and the runtime are the same
    fuse:: C++ code the reference binary ran."""
    o = Outcome()
    o.periods = periods
    graph = build_graph(accel, max_iterations, stations)
    session = pfuse.Session(graph)
    dirichlet = DirichletSlab("dirichlet", stations, wiring)
    neumann = NeumannSlab("neumann", stations, wiring)
    session.add_component(dirichlet)
    session.add_component(neumann)
    session.initialize()
    session.run(periods)

    o.sweeps = session.stats.total_iterations
    o.not_converged = session.stats.windows_not_converged
    if session.iteration_stats:
        o.last_residuals = list(session.iteration_stats[0].residual_history)

    t = neumann.time
    o.u_final = np.array(neumann.u)
    o.q_final = np.array(dirichlet.q)
    o.u_l2 = l2_error(o.u_final, lambda i: exact_u(i, stations, t))
    o.u_max = max_error(o.u_final, lambda i: exact_u(i, stations, t))
    o.q_l2 = l2_error(o.q_final, lambda i: exact_q(i, stations, t))
    o.q_max = max_error(o.q_final, lambda i: exact_q(i, stations, t))
    return o


def format_row(label, o):
    """examples/dn_heat/main.cpp's print_row(): '  %-9s %6lld %8lld %6lld  %10.3e %10.3e  %10.3e %10.3e'."""
    return "  {:<9} {:6d} {:8d} {:6d}  {:10.3e} {:10.3e}  {:10.3e} {:10.3e}".format(
        label, o.periods, o.sweeps, o.not_converged, o.u_l2, o.u_max, o.q_l2, o.q_max)


def format_residuals(label, h):
    """examples/dn_heat/main.cpp's print_residuals()."""
    head = h[:6]
    parts = ["  {:<9} {:4d} sweeps:".format(label, len(h))]
    parts += [" {:.2e}".format(x) for x in head]
    if len(h) > len(head) + 1:
        parts.append(" ...")
    if len(h) > len(head):
        parts.append(" {:.2e}".format(h[-1]))
    return "".join(parts)
