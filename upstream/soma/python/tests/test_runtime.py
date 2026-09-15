"""Driving a real coupled run from Python, both ways SOMA offers.

Two halves:

``SessionDrivesPythonPeers``
    Spec section 6: a Python class is a peer component. Two of them are wired
    into a real graph, lowered, scheduled and run through ``soma::Runner``, and
    the NUMBERS that arrive in the consumer's numpy array are checked. These
    mirror test/unit/runner_test.cpp's TransfersAValueFromProducerToConsumer,
    SubcyclingAdvancesTheFastComponentMoreOften and
    MeanAccumulationAveragesOverTheWindow, so a discrepancy between the C++ and
    Python front-ends shows up as a value mismatch rather than as silence.

``ContextIsTheCAbi``
    Spec section 3.2: ``soma.Context`` is ``include/soma/soma_c.h``, one method
    per entry point, driven with begin_step/end_step exactly as a C or Fortran
    caller would. The exact coupling windows are checked against the rational
    schedule, not against a tolerance.
"""

import os
import unittest

import numpy as np

import soma


DATA_DIR = os.environ.get("SOMA_TEST_DATA_DIR")


def yaml_path(name):
    if not DATA_DIR:
        raise unittest.SkipTest("SOMA_TEST_DATA_DIR is not set")
    return os.path.join(DATA_DIR, name)


# ---------------------------------------------------------------------------
# Python peer components
# ---------------------------------------------------------------------------


class Producer(soma.PyComponent):
    """Writes ``value * step`` into every entry of its field each advance."""

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


class Consumer(soma.PyComponent):
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
    """Two matching 1-D meshes and one channel, as runner_test.cpp builds."""
    g = soma.CouplingGraph()
    m1 = g.add_mesh("m1", "generated:line:{}".format(cells))
    m2 = g.add_mesh("m2", "generated:line:{}".format(cells))
    r1 = g.add_region("r1", m1, dimension=1)
    r2 = g.add_region("r2", m2, dimension=1)
    p = g.add_component("producer", first_rank=0, rank_count=1,
                        interval=src_dt)
    c = g.add_component("consumer", first_rank=1, rank_count=1,
                        interval=dst_dt)
    s = g.add_field(p, "out", r1, units=soma.units.K,
                    intent=soma.Intent.Provides)
    t = g.add_field(c, "inp", r2, units=soma.units.K,
                    intent=soma.Intent.Requires)
    ch = g.add_channel("link", s, t)
    if accum is not None:
        g.channel(ch).accumulate(accum)
    return g


class SessionDrivesPythonPeers(unittest.TestCase):
    def make(self, **kwargs):
        cells = kwargs.pop("cells", 8)
        g = line_pair(cells=cells, **kwargs)
        session = soma.Session(g)
        producer = Producer(cells)
        consumer = Consumer(cells)
        session.add_component(producer)
        session.add_component(consumer)
        return session, producer, consumer

    def test_transfers_a_value_from_producer_to_consumer(self):
        session, producer, consumer = self.make()
        session.initialize()
        session.run_period()
        self.assertEqual(producer.steps, 1)
        np.testing.assert_allclose(consumer.data, np.full(8, 2.0),
                                   rtol=0, atol=0)

    def test_lockstep_advances_each_component_once_per_period(self):
        session, producer, consumer = self.make()
        session.initialize()
        session.run(3)
        self.assertEqual(producer.steps, 3)
        self.assertEqual(consumer.steps, 3)
        self.assertEqual(session.stats.periods, 3)
        np.testing.assert_allclose(consumer.data, np.full(8, 6.0))

    def test_subcycling_advances_the_fast_component_more_often(self):
        session, producer, consumer = self.make(src_dt="1/4")
        session.initialize()
        session.run(1)
        self.assertEqual(producer.steps, 4)
        self.assertEqual(consumer.steps, 1)
        self.assertFalse(session.plan.lockstep)

    def test_mean_accumulation_averages_over_the_window(self):
        """Producer writes 2,4,6,8 over four substeps; the mean is 5."""
        session, _producer, consumer = self.make(accum=soma.AccumOp.Mean,
                                                 src_dt="1/4")
        session.initialize()
        session.run_period()
        self.assertAlmostEqual(consumer.data[0], 5.0, places=12)

    def test_the_step_context_carries_exact_rational_times(self):
        session, producer, _consumer = self.make(src_dt="1/4")
        session.initialize()
        session.run(1)
        begins = [c[0] for c in producer.contexts]
        ends = [c[1] for c in producer.contexts]
        self.assertEqual(begins, [soma.Rational(0), soma.Rational(1, 4),
                                  soma.Rational(1, 2), soma.Rational(3, 4)])
        self.assertEqual(ends, [soma.Rational(1, 4), soma.Rational(1, 2),
                                soma.Rational(3, 4), soma.Rational(1)])
        self.assertTrue(all(c[3] == 0 for c in producer.contexts),
                        "iteration is 0 outside an iteration region")

    def test_the_component_may_call_back_into_soma_while_advancing(self):
        """The GIL must be reacquired correctly by the forwarding component.

        If it were not, this would deadlock or abort rather than fail; the
        assertion is a formality, the fact that it returns is the test.
        """
        seen = []

        class Chatty(Producer):
            def advance(self, ctx):
                seen.append(soma.version())
                seen.append(str(soma.Rational(1, 3)))
                super().advance(ctx)

        g = line_pair()
        session = soma.Session(g)
        session.add_component(Chatty(8))
        session.add_component(Consumer(8))
        session.initialize()
        session.run(2)
        self.assertEqual(seen[1::2], ["1/3", "1/3"])

    def test_registry_knows_the_declared_shape_during_register_fields(self):
        """A shape mismatch is reported from the graph, not from error text."""

        class WrongShape(Producer):
            def register_fields(self, registry, graph):
                cid = graph.find_component("producer")
                for fid in graph.component(cid).fields:
                    # The field is declared with one component; a (8, 3) array
                    # says three.
                    registry.bind_numpy(fid, np.zeros((8, 3)))

        session = soma.Session(line_pair())
        session.add_component(WrongShape(8))
        session.add_component(Consumer(8))
        with self.assertRaises(soma.InvalidArgument) as cm:
            session.initialize()
        message = str(cm.exception)
        self.assertIn("declared with 1 component", message)
        self.assertIn("supplies 3", message)

    def test_a_registry_stashed_by_a_component_outlives_the_session(self):
        """The `registry` argument must not become a dangling pointer.

        A component that keeps the registry it was handed is doing something
        unusual but not illegal. The object it keeps shares ownership of the
        session's registry, so using it after the session is collected reads
        live memory -- which the asan leg is what actually proves.
        """
        stash = {}

        class Hoarder(Producer):
            def register_fields(self, registry, graph):
                stash["registry"] = registry
                stash["field"] = graph.component(
                    graph.find_component("producer")).fields[0]
                super().register_fields(registry, graph)

        session = soma.Session(line_pair())
        producer = Hoarder(8)
        session.add_component(producer)
        session.add_component(Consumer(8))
        session.initialize()
        session.run_period()

        del session
        import gc
        gc.collect()

        view = stash["registry"].get_numpy(stash["field"])
        np.testing.assert_allclose(view, np.full(8, 2.0))
        view[0] = -1.0
        self.assertEqual(producer.data[0], -1.0)

    def test_an_exception_from_advance_reaches_the_caller_verbatim(self):
        class Boom(Exception):
            pass

        class Exploding(Producer):
            def advance(self, ctx):
                raise Boom("surrogate diverged")

        session = soma.Session(line_pair())
        session.add_component(Exploding(8))
        session.add_component(Consumer(8))
        session.initialize()
        with self.assertRaises(Boom) as cm:
            session.run_period()
        self.assertEqual(str(cm.exception), "surrogate diverged")

    def test_an_exception_from_register_fields_reaches_the_caller(self):
        class Boom(Exception):
            pass

        class Exploding(Producer):
            def register_fields(self, registry, graph):
                raise Boom("no storage today")

        session = soma.Session(line_pair())
        session.add_component(Exploding(8))
        session.add_component(Consumer(8))
        with self.assertRaises(Boom):
            session.initialize()

    def test_an_unbound_field_is_refused_before_the_hot_loop(self):
        class Lazy(Producer):
            def register_fields(self, registry, graph):
                pass  # binds nothing

        session = soma.Session(line_pair())
        session.add_component(Lazy(8))
        session.add_component(Consumer(8))
        with self.assertRaises(soma.SomaRuntimeError) as cm:
            session.initialize()
        self.assertEqual(cm.exception.core_code, "E-RT-UNBOUND")

    def test_a_short_registration_is_refused(self):
        """Runner: a short array is a heap overflow in a loop with no checks."""
        class Short(Producer):
            def register_fields(self, registry, graph):
                cid = graph.find_component("producer")
                for fid in graph.component(cid).fields:
                    registry.bind_numpy(fid, np.zeros(3))  # region has 8

        session = soma.Session(line_pair())
        session.add_component(Short(8))
        session.add_component(Consumer(8))
        with self.assertRaises(soma.SomaRuntimeError) as cm:
            session.initialize()
        self.assertEqual(cm.exception.core_code, "E-RT-SIZE")

    def test_a_component_the_graph_does_not_declare_is_refused(self):
        class Stowaway(soma.PyComponent):
            def __init__(self):
                super().__init__("stowaway")

            def register_fields(self, registry, graph):
                pass

            def advance(self, ctx):
                pass

        session = soma.Session(line_pair())
        with self.assertRaises(soma.NotFound):
            session.add_component(Stowaway())

    def test_add_component_rejects_a_non_component(self):
        session = soma.Session(line_pair())
        with self.assertRaises(soma.InvalidArgument):
            session.add_component(object())

    def test_duplicate_components_are_refused(self):
        session = soma.Session(line_pair())
        session.add_component(Producer(8))
        with self.assertRaises(soma.InvalidArgument):
            session.add_component(Producer(8))

    def test_run_before_initialize_is_a_state_error(self):
        session = soma.Session(line_pair())
        session.add_component(Producer(8))
        session.add_component(Consumer(8))
        with self.assertRaises(soma.StateError):
            session.run_period()

    def test_the_session_owns_its_own_copy_of_the_graph(self):
        g = line_pair()
        session = soma.Session(g)
        # Mutating the caller's graph after construction must not reach the
        # session: the Runner holds a reference to the session's copy.
        g.add_component("late_arrival")
        self.assertEqual(session.graph.component_count, 2)
        self.assertEqual(g.component_count, 3)

    def test_the_consumer_reads_the_previous_periods_value(self):
        session, _producer, consumer = self.make()
        session.initialize()
        session.run(3)
        # runner_test.cpp pins the same convention (ExplicitExchangeIsLagged):
        # the exchange is explicit and one period lagged, so the consumer's
        # advance in period p integrates against what period p-1 delivered.
        self.assertEqual(len(consumer.seen), 3)
        self.assertEqual(consumer.seen, [0.0, 2.0, 4.0])


class ContextIsTheCAbi(unittest.TestCase):
    def test_version_and_abi_version(self):
        self.assertRegex(soma.version(), r"^\d+\.\d+\.\d+")
        self.assertEqual(soma.abi_version(), 1)

    def test_lifecycle_and_close(self):
        ctx = soma.Context()
        self.assertFalse(ctx.closed)
        ctx.close()
        self.assertTrue(ctx.closed)
        ctx.close()  # a second close is a no-op, not the ABI's double destroy
        with self.assertRaises(soma.StateError):
            ctx.initialize()

    def test_context_manager(self):
        with soma.Context() as ctx:
            self.assertFalse(ctx.closed)
        self.assertTrue(ctx.closed)

    def test_descriptor_round_trips_through_the_abi(self):
        g = line_pair()
        with soma.Context() as ctx:
            ctx.load_graph(g)
            self.assertEqual(ctx.save_graph(), g.to_descriptor())

    def test_a_graph_may_be_loaded_from_bytes(self):
        g = line_pair()
        with soma.Context() as ctx:
            ctx.load_graph(g.to_descriptor())
            self.assertEqual(ctx.field_name(0), "out")
            self.assertEqual(ctx.field_components(0), 1)

    def test_a_second_load_is_a_state_error(self):
        g = line_pair()
        with soma.Context() as ctx:
            ctx.load_graph(g)
            with self.assertRaises(soma.StateError) as cm:
                ctx.load_graph(g)
            self.assertEqual(cm.exception.code, soma.ERR_STATE)

    def test_a_bad_descriptor_is_rejected(self):
        with soma.Context() as ctx:
            with self.assertRaises(soma.InvalidArgument):
                ctx.load_graph(b"not a descriptor")

    def test_field_handle_and_its_failures(self):
        with soma.Context() as ctx:
            ctx.load_graph(line_pair())
            self.assertEqual(ctx.field_handle("producer", "out"), 0)
            self.assertEqual(ctx.field_handle("consumer", "inp"), 1)
            with self.assertRaises(soma.NotFound) as cm:
                ctx.field_handle("nobody", "out")
            self.assertEqual(cm.exception.code, soma.ERR_NOTFOUND)
            self.assertIn("nobody", ctx.last_error())
            with self.assertRaises(soma.NotFound):
                ctx.field_handle("producer", "no_such_field")

    def test_field_components_answers_from_the_graph_not_from_error_text(self):
        """The ABI has no soma_field_components(); the mirror provides it."""
        with soma.Context() as ctx:
            ctx.load_yaml(yaml_path("vector_field.yaml"))
            handle = ctx.field_handle("shell", "displacement")
            self.assertEqual(ctx.field_components(handle), 3)
            self.assertEqual(ctx.field_name(handle), "displacement")

    def test_a_shape_mismatch_names_both_counts(self):
        with soma.Context() as ctx:
            ctx.load_yaml(yaml_path("vector_field.yaml"))
            handle = ctx.field_handle("shell", "displacement")
            with self.assertRaises(soma.InvalidArgument) as cm:
                ctx.register_field(handle, np.zeros(5))  # 1 component
            message = str(cm.exception)
            self.assertIn("declared with 3 component", message)
            self.assertIn("supplies 1", message)
            # Nothing was registered, so the right array still binds.
            ctx.register_field(handle, np.zeros((5, 3)))

    def test_register_field_enforces_the_buffer_contract(self):
        with soma.Context() as ctx:
            ctx.load_graph(line_pair())
            handle = ctx.field_handle("producer", "out")
            with self.assertRaises(soma.InvalidArgument):
                ctx.register_field(handle, np.zeros(16)[::2])
            with self.assertRaises(soma.InvalidArgument):
                ctx.register_field(handle, np.zeros(8, dtype=np.float32))
            with self.assertRaises(soma.InvalidArgument):
                ctx.register_field(handle, [0.0] * 8)
            data = np.zeros(8)
            ctx.register_field(handle, data)
            with self.assertRaises(soma.StateError) as cm:
                ctx.register_field(handle, np.zeros(8))
            self.assertEqual(cm.exception.code, soma.ERR_STATE)

    def test_the_context_keeps_the_array_alive(self):
        ctx = soma.Context()
        ctx.load_graph(line_pair())
        handle = ctx.field_handle("producer", "out")
        ctx.register_field(handle, np.arange(8, dtype=np.float64))
        import gc
        gc.collect()
        # If the temporary had been collected, the pointer SOMA holds would
        # dangle; under the asan preset that is a use-after-free, not a
        # wrong answer.
        ctx.close()

    def test_end_to_end_with_exact_coupling_windows(self):
        """fluid runs at 1/2, solid at 1, so the windows are exact halves."""
        flux = np.zeros(16)
        wall = np.zeros(16)
        with soma.Context() as ctx:
            ctx.load_yaml(yaml_path("pair.yaml"))
            ctx.register_field(ctx.field_handle("fluid", "heat_flux"), flux)
            ctx.register_field(ctx.field_handle("solid", "q_wall"), wall)
            ctx.set_end_time(1)
            ctx.initialize()

            self.assertTrue(ctx.is_coupling_ongoing())
            self.assertEqual(ctx.begin_step("fluid"), (0.0, 0.5))
            ctx.end_step("fluid")
            self.assertEqual(ctx.begin_step("fluid"), (0.5, 1.0))
            ctx.end_step("fluid")
            self.assertEqual(ctx.begin_step("solid"), (0.0, 1.0))
            ctx.end_step("solid")
            # Both clocks have now reached the end time.
            self.assertFalse(ctx.is_coupling_ongoing())

    def test_without_an_end_time_the_run_never_ends_on_its_own(self):
        with soma.Context() as ctx:
            ctx.load_yaml(yaml_path("pair.yaml"))
            ctx.register_field(ctx.field_handle("fluid", "heat_flux"),
                               np.zeros(16))
            ctx.register_field(ctx.field_handle("solid", "q_wall"),
                               np.zeros(16))
            ctx.initialize()
            for _ in range(5):
                ctx.begin_step("fluid")
                ctx.end_step("fluid")
            self.assertTrue(ctx.is_coupling_ongoing())

    def test_set_end_time_takes_an_exact_rational(self):
        from fractions import Fraction
        with soma.Context() as ctx:
            ctx.set_end_time(Fraction(1, 3))
            ctx.set_end_time("2/7")
            ctx.set_end_time(soma.Rational(5, 2))
            ctx.set_end_time(4)
            with self.assertRaises(soma.InvalidArgument):
                ctx.set_end_time(0.5)

    def test_step_sequencing_is_enforced(self):
        with soma.Context() as ctx:
            ctx.load_yaml(yaml_path("pair.yaml"))
            ctx.register_field(ctx.field_handle("fluid", "heat_flux"),
                               np.zeros(16))
            ctx.register_field(ctx.field_handle("solid", "q_wall"),
                               np.zeros(16))
            with self.assertRaises(soma.StateError):
                ctx.begin_step("fluid")     # before initialize
            ctx.initialize()
            with self.assertRaises(soma.StateError):
                ctx.initialize()            # a second time
            ctx.begin_step("fluid")
            with self.assertRaises(soma.StateError):
                ctx.begin_step("fluid")     # already open
            ctx.end_step("fluid")
            with self.assertRaises(soma.StateError):
                ctx.end_step("fluid")       # not open
            with self.assertRaises(soma.NotFound):
                ctx.begin_step("nobody")

    def test_rollback_queries_are_false_outside_an_iteration_region(self):
        with soma.Context() as ctx:
            ctx.load_yaml(yaml_path("pair.yaml"))
            ctx.register_field(ctx.field_handle("fluid", "heat_flux"),
                               np.zeros(16))
            ctx.register_field(ctx.field_handle("solid", "q_wall"),
                               np.zeros(16))
            ctx.initialize()
            self.assertFalse(ctx.requires_checkpoint("fluid"))
            self.assertFalse(ctx.requires_restore("fluid"))

    def test_a_graph_that_does_not_validate_is_rejected_at_initialize(self):
        g = soma.CouplingGraph()
        m = g.add_mesh("m", "generated:line:4")
        r = g.add_region("r", m, dimension=1)
        a = g.add_component("a")
        b = g.add_component("b", first_rank=1)
        s = g.add_field(a, "T", r, units=soma.units.K,
                        intent=soma.Intent.Provides)
        t = g.add_field(b, "q", r, units=soma.units.W_per_m2,
                        intent=soma.Intent.Requires)
        g.add_channel("bad", s, t)
        with soma.Context() as ctx:
            ctx.load_graph(g)
            ctx.register_field(ctx.field_handle("a", "T"), np.zeros(4))
            ctx.register_field(ctx.field_handle("b", "q"), np.zeros(4))
            with self.assertRaises(soma.InvalidArgument) as cm:
                ctx.initialize()
            self.assertIn("E-UNITS", str(cm.exception))
            self.assertEqual(cm.exception.code, soma.ERR_INVALID)

    def test_run_and_run_period_are_reserved(self):
        """The ABI says so, and the binding must not paper over it."""
        with soma.Context() as ctx:
            ctx.load_graph(line_pair())
            ctx.register_field(ctx.field_handle("producer", "out"),
                               np.zeros(8))
            ctx.register_field(ctx.field_handle("consumer", "inp"),
                               np.zeros(8))
            ctx.initialize()
            with self.assertRaises(soma.Unsupported) as cm:
                ctx.run_period()
            self.assertEqual(cm.exception.code, soma.ERR_UNSUPPORTED)
            with self.assertRaises(soma.Unsupported):
                ctx.run(3)
            # ... and NotImplementedError, so `except NotImplementedError`
            # works for a caller who does not import soma's hierarchy.
            with self.assertRaises(NotImplementedError):
                ctx.run_period()

    def test_run_before_initialize_is_a_state_error_not_unsupported(self):
        with soma.Context() as ctx:
            ctx.load_graph(line_pair())
            with self.assertRaises(soma.StateError):
                ctx.run_period()

    def test_set_communicator_is_unsupported_without_mpi(self):
        with soma.Context() as ctx:
            try:
                ctx.set_communicator(0)
            except soma.Unsupported:
                pass  # a build with no MPI transport, which is the default
            else:
                self.skipTest("this build has an MPI transport")

    def test_last_error_is_populated_and_is_a_copy(self):
        with soma.Context() as ctx:
            self.assertEqual(ctx.last_error(), "")
            with self.assertRaises(soma.StateError):
                ctx.initialize()
            message = ctx.last_error()
            self.assertIn("no graph has been loaded", message)
            self.assertNotIn("\x00", message)


if __name__ == "__main__":
    unittest.main()
