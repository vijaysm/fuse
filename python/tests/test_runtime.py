"""Driving a real coupled run from Python through fuse::Runner.

Transplanted from upstream soma's SessionDrivesPythonPeers (spec section 6: a Python class is a
peer component). Two of them are wired into a real graph, lowered, scheduled and run through
Session (which owns a fuse::Runner), and the NUMBERS that arrive in the consumer's numpy array
are checked -- mirroring test/runtime/runner_test.cpp's own TransfersAValueFromProducerToConsumer
and SubcyclingAdvancesTheFastComponentMoreOften, so a discrepancy between the C++ and Python
front-ends shows up as a value mismatch, not as silence.

Upstream's second half, ContextIsTheCAbi, is NOT carried over: pfuse does not bind
fuse_coupler_abi()/a Context mirror at all (the owner's own choice of pybind11 over the ABI --
see docs/dev/python-surface.md, "What this does not expose").

STATUS: acceptance test, written against the surface in docs/dev/python-surface.md; `pfuse` does
not exist yet (Stage B builds it).
"""

import gc

import numpy as np
import pytest

pfuse = pytest.importorskip("pfuse", reason="Stage B has not built the _pfuse extension yet")


class Producer(pfuse.Component):
    """Writes ``value * step`` into every entry of its field each advance()."""

    def __init__(self, cells, value=2.0):
        super().__init__("producer")
        self.data = np.zeros(cells)
        self.value = value
        self.steps = 0
        self.contexts = []

    def register_fields(self, registry, graph):
        cid = graph.find_component("producer")
        for fid in graph.component(cid).fields:
            registry.bind_numpy(fid, self.data)

    def advance(self, ctx):
        self.steps += 1
        self.contexts.append((ctx.t_begin, ctx.t_end, ctx.step, ctx.iteration))
        self.data[:] = self.value * self.steps


class Consumer(pfuse.Component):
    """Holds whatever it receives, and records what it saw each step."""

    def __init__(self, cells):
        super().__init__("consumer")
        self.data = np.zeros(cells)
        self.steps = 0
        self.seen = []

    def register_fields(self, registry, graph):
        cid = graph.find_component("consumer")
        for fid in graph.component(cid).fields:
            registry.bind_numpy(fid, self.data)

    def advance(self, ctx):
        self.steps += 1
        self.seen.append(float(self.data[0]))


def line_pair(accum=None, src_dt=1, dst_dt=1, cells=8):
    """Two matching 1-D line meshes and one channel -- the shape runner_test.cpp builds."""
    g = pfuse.CouplingGraph()
    m1 = g.add_mesh("m1", "generated:line:{}".format(cells))
    m2 = g.add_mesh("m2", "generated:line:{}".format(cells))
    r1 = g.add_region("r1", m1, dimension=1)
    r2 = g.add_region("r2", m2, dimension=1)
    p = g.add_component("producer", ranks=pfuse.RankSet(0, 1), interval=src_dt)
    c = g.add_component("consumer", ranks=pfuse.RankSet(1, 1), interval=dst_dt)
    s = g.add_field(p, "out", r1, units=pfuse.units.K, intent=pfuse.Intent.Provides)
    t = g.add_field(c, "inp", r2, units=pfuse.units.K, intent=pfuse.Intent.Requires)
    ch = g.add_channel("link", s, t)
    if accum is not None:
        g.channel(ch).accumulate(accum)
    return g


class TestSessionDrivesPythonPeers:
    def make(self, **kwargs):
        cells = kwargs.pop("cells", 8)
        g = line_pair(cells=cells, **kwargs)
        # fuse::RunnerConfig{}'s defaults ("copy" transport, no base_communicator) move whole
        # arcs without touching MPI -- see docs/dev/python-surface.md, "What this does not
        # expose." pfuse's Session construction below must use exactly this default.
        session = pfuse.Session(g)
        producer = Producer(cells)
        consumer = Consumer(cells)
        session.add_component(producer)
        session.add_component(consumer)
        return session, producer, consumer

    def test_transfers_a_value_from_producer_to_consumer(self):
        session, producer, consumer = self.make()
        session.initialize()
        session.run_period()
        assert producer.steps == 1
        np.testing.assert_allclose(consumer.data, np.full(8, 2.0), rtol=0, atol=0)

    def test_lockstep_advances_each_component_once_per_period(self):
        session, producer, consumer = self.make()
        session.initialize()
        session.run(3)
        assert producer.steps == 3
        assert consumer.steps == 3
        assert session.stats.periods == 3
        np.testing.assert_allclose(consumer.data, np.full(8, 6.0))

    def test_subcycling_advances_the_fast_component_more_often(self):
        session, producer, consumer = self.make(src_dt="1/4")
        session.initialize()
        session.run(1)
        assert producer.steps == 4
        assert consumer.steps == 1
        assert not session.plan.lockstep

    def test_mean_accumulation_averages_over_the_window(self):
        """Producer writes 2,4,6,8 over four substeps; the mean is 5."""
        session, _producer, consumer = self.make(accum=pfuse.AccumOp.Mean, src_dt="1/4")
        session.initialize()
        session.run_period()
        assert consumer.data[0] == pytest.approx(5.0, abs=1e-12)

    def test_the_step_context_carries_exact_rational_times(self):
        session, producer, _consumer = self.make(src_dt="1/4")
        session.initialize()
        session.run(1)
        begins = [c[0] for c in producer.contexts]
        ends = [c[1] for c in producer.contexts]
        assert begins == [pfuse.Rational(0), pfuse.Rational(1, 4),
                           pfuse.Rational(1, 2), pfuse.Rational(3, 4)]
        assert ends == [pfuse.Rational(1, 4), pfuse.Rational(1, 2),
                         pfuse.Rational(3, 4), pfuse.Rational(1)]
        assert all(c[3] == 0 for c in producer.contexts), \
            "iteration is 0 outside an implicit coupling group"

    def test_the_component_may_call_back_into_pfuse_while_advancing(self):
        """The GIL must be reacquired correctly by ForwardingComponent.

        If it were not, this would deadlock or abort rather than fail; the assertion is a
        formality -- the fact that it returns at all is the test.
        """
        seen = []

        class Chatty(Producer):
            def advance(self, ctx):
                seen.append(pfuse.version())
                seen.append(str(pfuse.Rational(1, 3)))
                super().advance(ctx)

        g = line_pair()
        session = pfuse.Session(g)
        session.add_component(Chatty(8))
        session.add_component(Consumer(8))
        session.initialize()
        session.run(2)
        assert seen[1::2] == ["1/3", "1/3"]

    def test_registry_knows_the_declared_shape_during_register_fields(self):
        """A shape mismatch is reported from the graph, not sniffed out of error text alone."""

        class WrongShape(Producer):
            def register_fields(self, registry, graph):
                cid = graph.find_component("producer")
                for fid in graph.component(cid).fields:
                    # The field is declared with one component; a (8, 3) array says three.
                    registry.bind_numpy(fid, np.zeros((8, 3)))

        session = pfuse.Session(line_pair())
        session.add_component(WrongShape(8))
        session.add_component(Consumer(8))
        with pytest.raises(pfuse.InvalidArgument) as excinfo:
            session.initialize()
        message = str(excinfo.value)
        assert "declared with 1 component" in message
        assert "supplies 3" in message

    def test_a_registry_stashed_by_a_component_outlives_the_session(self):
        """The `registry` argument must not become a dangling pointer once the session dies.

        A component that keeps the registry it was handed is unusual but not illegal; the
        object it keeps shares ownership of the session's registry (fuse's own KeepAlive
        pattern, see docs/dev/python-surface.md's numpy-contract section), so using it after
        the session is collected must still read live memory.
        """
        stash = {}

        class Hoarder(Producer):
            def register_fields(self, registry, graph):
                stash["registry"] = registry
                stash["field"] = graph.component(graph.find_component("producer")).fields[0]
                super().register_fields(registry, graph)

        session = pfuse.Session(line_pair())
        producer = Hoarder(8)
        session.add_component(producer)
        session.add_component(Consumer(8))
        session.initialize()
        session.run_period()

        del session
        gc.collect()

        view = stash["registry"].get_numpy(stash["field"])
        np.testing.assert_allclose(view, np.full(8, 2.0))
        view[0] = -1.0
        assert producer.data[0] == -1.0

    def test_an_exception_from_advance_reaches_the_caller_verbatim(self):
        class Boom(Exception):
            pass

        class Exploding(Producer):
            def advance(self, ctx):
                raise Boom("surrogate diverged")

        session = pfuse.Session(line_pair())
        session.add_component(Exploding(8))
        session.add_component(Consumer(8))
        session.initialize()
        with pytest.raises(Boom) as excinfo:
            session.run_period()
        assert str(excinfo.value) == "surrogate diverged"

    def test_an_exception_from_register_fields_reaches_the_caller(self):
        class Boom(Exception):
            pass

        class Exploding(Producer):
            def register_fields(self, registry, graph):
                raise Boom("no storage today")

        session = pfuse.Session(line_pair())
        session.add_component(Exploding(8))
        session.add_component(Consumer(8))
        with pytest.raises(Boom):
            session.initialize()

    def test_an_unbound_field_is_refused_before_the_hot_loop(self):
        """fuse::Runner::initialize()'s E-RT-UNBOUND is a fuse::Result failure -- a bare
        pfuse.FuseError carrying .code, per the owner's ruling (docs/dev/python-surface.md)."""
        class Lazy(Producer):
            def register_fields(self, registry, graph):
                pass  # binds nothing

        session = pfuse.Session(line_pair())
        session.add_component(Lazy(8))
        session.add_component(Consumer(8))
        with pytest.raises(pfuse.FuseError) as excinfo:
            session.initialize()
        assert excinfo.value.code == "E-RT-UNBOUND"

    def test_a_component_the_graph_does_not_declare_is_refused(self):
        class Stowaway(pfuse.Component):
            def __init__(self):
                super().__init__("stowaway")

            def register_fields(self, registry, graph):
                pass

            def advance(self, ctx):
                pass

        session = pfuse.Session(line_pair())
        # Session.add_component() wraps CouplingGraph::find_component() the same way
        # CouplingGraph.find_component() itself does (test_graph.py's own
        # test_find_component_raises_rather_than_returning_a_sentinel): a thrown
        # fuse::InvalidArgument, not a dedicated "not found" class.
        with pytest.raises(pfuse.InvalidArgument):
            session.add_component(Stowaway())

    def test_add_component_rejects_a_non_component(self):
        session = pfuse.Session(line_pair())
        with pytest.raises(pfuse.InvalidArgument):
            session.add_component(object())

    def test_duplicate_components_are_refused(self):
        session = pfuse.Session(line_pair())
        session.add_component(Producer(8))
        with pytest.raises(pfuse.InvalidArgument):
            session.add_component(Producer(8))

    def test_run_before_initialize_is_a_sequencing_error(self):
        """Session.run_period()/run() before initialize(): a thrown fuse::InvalidArgument
        from pfuse's own binding code (Session::require_initialized), not a dedicated
        "state" class -- the six fuse::Error-mirroring classes have none (the owner's
        ruling fixes that list at exactly six; see test_graph.py::TestExceptionHierarchy)."""
        session = pfuse.Session(line_pair())
        session.add_component(Producer(8))
        session.add_component(Consumer(8))
        with pytest.raises(pfuse.InvalidArgument):
            session.run_period()

    def test_the_session_owns_its_own_copy_of_the_graph(self):
        """Session's constructor copies: fuse::Runner holds const CouplingGraph& for its whole life."""
        g = line_pair()
        session = pfuse.Session(g)
        g.add_component("late_arrival")
        assert session.graph.component_count == 2
        assert g.component_count == 3

    def test_the_consumer_reads_the_previous_periods_value(self):
        session, _producer, consumer = self.make()
        session.initialize()
        session.run(3)
        # The exchange is explicit and one period lagged: the consumer's advance in period p
        # integrates against what period p-1 delivered.
        assert len(consumer.seen) == 3
        assert consumer.seen == [0.0, 2.0, 4.0]
