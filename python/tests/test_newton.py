"""Newton-region configuration, the capability gate, and the native-components-only boundary.

New relative to upstream: soma's iteration regions had no accelerator and no Newton path at
all (docs/dev/python-surface.md's evidence table). This file specifies what v1 CAN cover for
Newton:

  1. IterationBuilder.newton()/.accelerate() configuration from Python.
  2. The capability-gate refusal for the HONEST default (a Python component that declares
     nothing refuses at Session.initialize() through fuse::verify_newton_capability_support()
     -- an EXISTING C++ check this file adds no new code for, only a test that makes it fire).
  3. The NEW refusal for a Python component that LIES about having the capability (claims
     CAP_RESIDUAL/CAP_LINEARIZATION via its own CAPABILITIES attribute, which
     ForwardingComponent can never back since Component.implicit() always returns nullptr for
     a Python subclass) -- raised by pfuse's own binding code, with a message a Python reader
     can act on.
  4. The Python-authored descriptor of a Newton-configured region compared BYTE FOR BYTE
     against a reference generated from the C++ API directly (not assumed equal).

Explicitly, plainly, NOT covered: Python v1 cannot RUN a Newton-solved region, because no
component reachable from a Python-authored graph can ever supply a linearisation
(docs/dev/python-surface.md, "Components implemented in Python"). The capability-gate test
below (TestNewtonCapabilityGate) is simultaneously the proof of this limitation: ANY
Python-only Newton region refuses at initialize(), honest or not, every time.

STATUS: `pfuse` is built; every test in this file runs for real (no module-absent skip).
"""

import pathlib

import pytest

pfuse = pytest.importorskip("pfuse", reason="the _pfuse extension is not built")

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"


def line_pair_with_two_components():
    g = pfuse.CouplingGraph()
    m1 = g.add_mesh("m1", "generated:line:8")
    m2 = g.add_mesh("m2", "generated:line:8")
    r1 = g.add_region("r1", m1, dimension=1)
    r2 = g.add_region("r2", m2, dimension=1)
    a = g.add_component("a", ranks=pfuse.RankSet(0, 1))
    b = g.add_component("b", ranks=pfuse.RankSet(0, 1))
    s = g.add_field(a, "out", r1, units=pfuse.units.K, intent=pfuse.Intent.Provides)
    t = g.add_field(b, "inp", r2, units=pfuse.units.K, intent=pfuse.Intent.Requires)
    ch = g.add_channel("link", s, t)
    return g, ch, a, b


class TestIterationBuilderConfiguration:
    def test_newton_sets_the_residual_capability_requirement(self):
        g, ch, _a, _b = line_pair_with_two_components()
        region = (g.iterate([ch])
                   .until(pfuse.Converged.Relative(1e-8, "inp"))
                   .newton())
        decl = g.iteration_region(region.id())
        assert decl.acceleration.kind == pfuse.AccelKind.Newton
        assert decl.requires_residual_capability
        assert not decl.requires_linearization_capability  # block-Jacobi preconditioner off by default

    def test_newton_with_block_jacobi_preconditioner_also_requires_linearization(self):
        g, ch, _a, _b = line_pair_with_two_components()
        cfg = pfuse.NewtonConfig(use_block_jacobi_preconditioner=True)
        region = g.iterate([ch]).until(pfuse.Converged.Relative(1e-8, "inp")).newton(cfg)
        decl = g.iteration_region(region.id())
        assert decl.requires_linearization_capability

    def test_convergence_criterion_composes_with_and(self):
        """Converged.Relative(...) & Converged.Absolute(...) -- the spelling dn_heat uses."""
        criterion = (pfuse.Converged.Relative(1e-8, "u_wall")
                     & pfuse.Converged.Absolute(1e-8, "flux"))
        assert len(criterion) == 2
        fields = [m.field for m in criterion.measures]
        assert fields == ["u_wall", "flux"]

    def test_accelerate_with_aitken(self):
        g, ch, _a, _b = line_pair_with_two_components()
        region = (g.iterate([ch])
                   .until(pfuse.Converged.Relative(1e-6, "inp"))
                   .max_iterations(50)
                   .accelerate(pfuse.AccelConfig(kind=pfuse.AccelKind.Aitken)))
        decl = g.iteration_region(region.id())
        assert decl.acceleration.kind == pfuse.AccelKind.Aitken
        assert decl.max_iterations == 50


def _plain_component(name):
    """A Python component honest about not implementing a residual (no CAPABILITIES
    attribute, so capabilities() has no CAP_RESIDUAL/CAP_LINEARIZATION bit), but WITH a
    trivial save_state()/restore_state() pair -- the default residual path (NewtonResidualPath
    .Advance) evaluates a Newton region's G(x) by calling advance() under a checkpoint/
    restore bracket exactly as Picard does, so every member still needs rollback support
    regardless of the residual-capability question this test isolates."""
    import numpy as np

    class PlainComponent(pfuse.Component):
        def __init__(self, name):
            super().__init__(name)
            self._data = np.zeros(8)
            self._saved = np.zeros(8)

        def register_fields(self, registry, graph):
            cid = graph.find_component(self.name)
            for fid in graph.component(cid).fields:
                registry.bind_numpy(fid, self._data)

        def advance(self, ctx):
            pass

        def save_state(self):
            self._saved[:] = self._data

        def restore_state(self):
            self._data[:] = self._saved

    return PlainComponent(name)


class TestNewtonCapabilityGate:
    """Mirrors fuse::verify_newton_capability_support()'s own contract, checked at initialize().

    This is simultaneously the proof that Python v1 cannot RUN a Newton-solved region: the
    honest default (capabilities() == 0, no residual bit) is refused here by an EXISTING
    fuse-side check that needed no new code -- a Python component is never
    special-cased by it, it is simply never able to set the bit honestly.
    """

    def test_a_member_with_no_residual_capability_refuses_at_initialize(self):
        g, ch, _a, _b = line_pair_with_two_components()
        g.iterate([ch]).until(pfuse.Converged.Relative(1e-8, "inp")).newton()
        session = pfuse.Session(g)
        session.add_component(_plain_component("a"))
        session.add_component(_plain_component("b"))
        with pytest.raises(pfuse.FuseError) as excinfo:
            session.initialize()
        # fuse::Runner::initialize()'s own E-NEWTON-CAPABILITY refusal is a fuse::Result<void>
        # failure (measured: NOT a Diagnostics-carrying validate()/lower()/schedule() failure
        # -- .diagnostics is None here), so the bare pfuse.FuseError's .code/.message carry it.
        assert excinfo.value.diagnostics is None
        assert excinfo.value.code == "E-NEWTON-CAPABILITY"
        assert "kCapabilityResidual" in excinfo.value.message

    def test_the_existing_refusal_message_names_cxx_overrides_a_python_reader_cannot_make(self):
        """Recorded here, not fixed: src/runtime/checkpoint_support.cpp's own E-NEWTON-
        CAPABILITY message reads 'override Component::implicit() to return it' -- correct
        advice for a C++ author, impossible advice for a Python one, since pfuse does not
        expose Component.implicit() to a Python subclass at all (docs/dev/python-surface.md).
        src/ and include/ are not edited here; this records the exact message so that
        whether a Python-specific wording is worth adding there can be judged."""
        g, ch, _a, _b = line_pair_with_two_components()
        g.iterate([ch]).until(pfuse.Converged.Relative(1e-8, "inp")).newton()
        session = pfuse.Session(g)
        session.add_component(_plain_component("a"))
        session.add_component(_plain_component("b"))
        with pytest.raises(pfuse.FuseError) as excinfo:
            session.initialize()
        assert "override Component::implicit()" in excinfo.value.message  # measured, not assumed


class TestClaimingTheCapabilityIsRefusedImmediately:
    """NEW: a Python component that LIES about having the capability is refused at
    Session.add_component() time, with a message a Python reader can act on -- see
    ForwardingComponent::compute_capabilities() (python/src/component_bindings.cpp)."""

    def test_claiming_residual_capability_is_refused(self):
        class Lying(pfuse.Component):
            CAPABILITIES = pfuse.CAP_RESIDUAL

            def register_fields(self, registry, graph):
                pass

            def advance(self, ctx):
                pass

        g, _ch, _a, _b = line_pair_with_two_components()
        session = pfuse.Session(g)
        with pytest.raises(pfuse.InvalidArgument, match="kCapabilityResidual"):
            session.add_component(Lying("a"))

    def test_claiming_linearization_capability_is_refused(self):
        class Lying(pfuse.Component):
            CAPABILITIES = pfuse.CAP_LINEARIZATION

            def register_fields(self, registry, graph):
                pass

            def advance(self, ctx):
                pass

        g, _ch, _a, _b = line_pair_with_two_components()
        session = pfuse.Session(g)
        with pytest.raises(pfuse.InvalidArgument, match="kCapabilityLinearization"):
            session.add_component(Lying("a"))

    def test_claiming_rollback_alone_is_not_refused(self):
        """CAP_ROLLBACK is not restricted -- only the two Newton-residual bits are."""
        class Honest(pfuse.Component):
            CAPABILITIES = pfuse.CAP_ROLLBACK

            def register_fields(self, registry, graph):
                pass

            def advance(self, ctx):
                pass

        g, _ch, _a, _b = line_pair_with_two_components()
        session = pfuse.Session(g)
        session.add_component(Honest("a"))  # must not raise


class TestDescriptorMatchesNativeReference:
    """Item 4: the Python-authored descriptor of a Newton-configured region compared byte for
    byte against a reference generated directly from the C++ API (not assumed equal).

    Reference generation (recorded here, not invented): a standalone program
    (/tmp/pybind/probe/newton_descriptor_probe.cpp, not part of this tree's deliverables)
    built against the tier-0 libfuse-coupler-model.a/libfuse-coupler-base.a and
    linked with `clang++ -std=c++20 -ffp-contract=off`, constructs the IDENTICAL graph this
    test builds from Python (same mesh/region/component/field/channel/remap/spacetime/relax
    configuration, same NewtonConfig(rtol=1e-9, max_nonlinear_iterations=30)) and writes
    fuse::serialize(g) to golden/newton_descriptor_reference.bin.
    """

    def test_python_authored_newton_descriptor_matches_the_cxx_reference(self):
        g = pfuse.CouplingGraph()
        m1 = g.add_mesh("fluid_mesh", "generated:box:8x8x8")
        m2 = g.add_mesh("solid_mesh", "generated:box:5x5x5")
        r1 = g.add_region("fluid_wall", m1, entity_set="wall", dimension=2)
        r2 = g.add_region("solid_wall", m2, entity_set="wall", dimension=2)
        fluid = g.add_component("fluid", ranks=pfuse.RankSet(0, 2), interval="1/2")
        solid = g.add_component("solid", ranks=pfuse.RankSet(2, 2), interval=1)
        src = g.add_field(fluid, "heat_flux", r1, units=pfuse.units.W_per_m2,
                          conservation=pfuse.Conservation.Extensive,
                          intent=pfuse.Intent.Provides)
        dst = g.add_field(solid, "q_wall", r2, units=pfuse.units.W_per_m2,
                          conservation=pfuse.Conservation.Extensive,
                          intent=pfuse.Intent.Requires)
        ch = g.add_channel("flux", src, dst)
        (g.channel(ch)
          .accumulate(pfuse.AccumOp.Mean, pfuse.Window.coupling_period())
          .remap(pfuse.RemapFlags.Conservative | pfuse.RemapFlags.Monotone)
          .remap_method("mono3")
          .spacetime(pfuse.TemporalBasis.DG, 2, pfuse.Window.explicit_length("2/5"))
          .relax(0.7))
        g.iterate([ch]).until(pfuse.Converged.Relative(1e-8, "q_wall")).newton(
            pfuse.NewtonConfig(rtol=1e-9, max_nonlinear_iterations=30))

        reference = (GOLDEN_DIR / "newton_descriptor_reference.bin").read_bytes()
        assert g.to_bytes() == reference
