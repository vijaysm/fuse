"""The NumPy buffer contract.

SOMA never owns field storage (soma_c.h, "FIELD STORAGE"): it records a pointer
and a layout, and the caller must keep that storage alive at a fixed address.
Every way a Python caller can break that -- a strided view, a transpose, a
dtype that is not float64, a read-only array, a temporary that would be
collected, an in-place resize that reallocates -- is either refused here or
made impossible, and each rejection has a test.

Run this file under the asan preset too: a lifetime bug that these assertions
cannot see is exactly what the sanitizer leg is for.
"""

import gc
import unittest

import numpy as np

import soma


F0 = soma.FieldId(0)
F1 = soma.FieldId(1)


class BindingAliasesTheCallersStorage(unittest.TestCase):
    def test_the_binding_is_a_view_not_a_copy(self):
        data = np.zeros(4)
        reg = soma.FieldRegistry()
        reg.bind_numpy(F0, data)
        view = reg.get_numpy(F0)
        view[1] = 7.5
        self.assertEqual(data[1], 7.5,
                         "the binding must alias the component's storage")

    def test_writes_to_the_original_are_seen_through_the_view(self):
        data = np.zeros(4)
        reg = soma.FieldRegistry()
        reg.bind_numpy(F0, data)
        data[:] = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(reg.get_numpy(F0), data)

    def test_the_view_shares_the_buffer_not_just_the_values(self):
        data = np.zeros(4)
        reg = soma.FieldRegistry()
        reg.bind_numpy(F0, data)
        view = reg.get_numpy(F0)
        self.assertEqual(view.__array_interface__["data"][0],
                         data.__array_interface__["data"][0],
                         "the view must start at the same address")
        self.assertFalse(view.flags.owndata,
                         "a view that owns its data is a copy")

    def test_two_dimensional_arrays_bind_interleaved(self):
        data = np.zeros((5, 3))
        reg = soma.FieldRegistry()
        reg.bind_numpy(F0, data)
        view = reg.get_numpy(F0)
        self.assertEqual(view.shape, (5, 3))
        view[2, 1] = -4.25
        self.assertEqual(data[2, 1], -4.25)

    def test_bookkeeping(self):
        reg = soma.FieldRegistry()
        self.assertEqual(reg.bound_count, 0)
        self.assertFalse(reg.is_bound(F0))
        reg.bind_numpy(F0, np.zeros(3))
        self.assertEqual(reg.bound_count, 1)
        self.assertTrue(reg.is_bound(F0))
        self.assertFalse(reg.is_bound(F1))


class RejectsWhatWouldBeTheWrongMemory(unittest.TestCase):
    def setUp(self):
        self.reg = soma.FieldRegistry()

    def test_rejects_non_contiguous_column_slice(self):
        data = np.zeros((4, 3))[:, 0]  # strided view
        self.assertFalse(data.flags.c_contiguous)
        with self.assertRaises(ValueError) as cm:
            self.reg.bind_numpy(F0, data)
        self.assertIn("C-contiguous", str(cm.exception))

    def test_rejects_a_step_slice(self):
        data = np.zeros(8)[::2]
        self.assertFalse(data.flags.c_contiguous)
        with self.assertRaises(soma.InvalidArgument):
            self.reg.bind_numpy(F0, data)

    def test_rejects_a_transpose(self):
        """A transposed (3, 5) array is F-contiguous, i.e. the wrong reading."""
        data = np.zeros((3, 5)).T
        self.assertEqual(data.shape, (5, 3))
        self.assertFalse(data.flags.c_contiguous)
        with self.assertRaises(soma.InvalidArgument):
            self.reg.bind_numpy(F0, data)

    def test_accepts_the_repaired_transpose_but_it_is_a_different_buffer(self):
        original = np.zeros((3, 5))
        repaired = np.ascontiguousarray(original.T)
        self.reg.bind_numpy(F0, repaired)
        self.reg.get_numpy(F0)[0, 0] = 9.0
        self.assertEqual(repaired[0, 0], 9.0)
        self.assertEqual(original[0, 0], 0.0,
                         "ascontiguousarray copies; the original is NOT bound")

    def test_rejects_float32(self):
        with self.assertRaises(soma.InvalidArgument) as cm:
            self.reg.bind_numpy(F0, np.zeros(4, dtype=np.float32))
        self.assertIn("float64", str(cm.exception))

    def test_rejects_int64(self):
        with self.assertRaises(soma.InvalidArgument):
            self.reg.bind_numpy(F0, np.zeros(4, dtype=np.int64))

    def test_rejects_big_endian_float64(self):
        data = np.zeros(4, dtype=">f8")
        if data.dtype == np.dtype(np.float64):
            self.skipTest("this machine is big-endian; >f8 is native")
        with self.assertRaises(soma.InvalidArgument):
            self.reg.bind_numpy(F0, data)

    def test_rejects_a_read_only_array(self):
        data = np.zeros(4)
        data.setflags(write=False)
        with self.assertRaises(soma.InvalidArgument) as cm:
            self.reg.bind_numpy(F0, data)
        self.assertIn("read-only", str(cm.exception))

    def test_rejects_a_list(self):
        with self.assertRaises(soma.InvalidArgument) as cm:
            self.reg.bind_numpy(F0, [1.0, 2.0, 3.0])
        self.assertIn("numpy.ndarray", str(cm.exception))

    def test_rejects_a_memoryview(self):
        with self.assertRaises(soma.InvalidArgument):
            self.reg.bind_numpy(F0, memoryview(bytearray(32)))

    def test_rejects_three_dimensions(self):
        with self.assertRaises(soma.InvalidArgument) as cm:
            self.reg.bind_numpy(F0, np.zeros((2, 3, 4)))
        self.assertIn("dimensions", str(cm.exception))

    def test_rejects_a_zero_dimensional_array(self):
        with self.assertRaises(soma.InvalidArgument):
            self.reg.bind_numpy(F0, np.float64(1.0).reshape(()))

    def test_rebinding_is_refused_not_silently_replaced(self):
        first = np.zeros(4)
        second = np.ones(4)
        self.reg.bind_numpy(F0, first)
        with self.assertRaises(soma.StateError) as cm:
            self.reg.bind_numpy(F0, second)
        self.assertIn("E-FIELD-REBIND", str(cm.exception))
        # The first binding is untouched: on any error the registry is left
        # exactly as it was.
        self.reg.get_numpy(F0)[0] = 3.0
        self.assertEqual(first[0], 3.0)
        self.assertEqual(second[0], 1.0)

    def test_get_numpy_of_an_unbound_field_raises(self):
        with self.assertRaises(soma.NotFound):
            self.reg.get_numpy(F1)

    def test_a_refused_binding_leaves_nothing_bound(self):
        with self.assertRaises(soma.InvalidArgument):
            self.reg.bind_numpy(F0, np.zeros(8)[::2])
        self.assertFalse(self.reg.is_bound(F0))
        self.assertEqual(self.reg.bound_count, 0)


class Lifetime(unittest.TestCase):
    """The registry must keep the array alive; SOMA has no other owner."""

    def test_a_temporary_expression_is_kept_alive(self):
        reg = soma.FieldRegistry()
        # Nothing in the caller's scope references this array after the call.
        reg.bind_numpy(F0, np.zeros(4) + 2.5)
        gc.collect()
        view = reg.get_numpy(F0)
        np.testing.assert_array_equal(view, [2.5, 2.5, 2.5, 2.5])
        view[3] = 1.0
        np.testing.assert_array_equal(reg.get_numpy(F0),
                                      [2.5, 2.5, 2.5, 1.0])

    def test_the_array_outlives_the_callers_last_name_for_it(self):
        reg = soma.FieldRegistry()
        data = np.arange(4, dtype=np.float64)
        reg.bind_numpy(F0, data)
        del data
        gc.collect()
        np.testing.assert_array_equal(reg.get_numpy(F0), [0.0, 1.0, 2.0, 3.0])

    def test_numpy_refuses_to_resize_a_bound_array(self):
        """A resize reallocates, which would dangle the pointer SOMA holds.

        numpy refuses to resize an array something else references, and the
        registry's strong reference is exactly such a reference. The control
        case first, so a numpy that had stopped doing refcount checks would
        skip rather than pass vacuously.
        """
        control = np.zeros(4)
        try:
            control.resize(8, refcheck=True)
        except ValueError:
            self.skipTest("this interpreter holds extra references to locals; "
                          "numpy's resize refcheck is not usable here")

        reg = soma.FieldRegistry()
        bound = np.zeros(4)
        reg.bind_numpy(F0, bound)
        with self.assertRaises(ValueError):
            bound.resize(8, refcheck=True)


if __name__ == "__main__":
    unittest.main()
