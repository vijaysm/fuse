"""The NumPy buffer contract, transplanted from upstream soma's test_buffers.py.

fuse::FieldRegistry never owns field storage (fuse/field/field_ref.hpp's own LIFETIME
CONTRACT): it records a pointer and a fuse::Layout, and the caller must keep that storage alive
at a fixed address. Every way a Python caller can break that -- a strided view, a transpose, a
dtype that is not float64, a read-only array, a temporary that would be collected, an in-place
resize that reallocates -- is either refused here or made impossible, and each rejection has a
test. See docs/dev/python-surface.md, "The numpy buffer contract and the lifetime rule," for the
design this file is written against.

STATUS: acceptance test, written against the surface in docs/dev/python-surface.md. `pfuse` does
not exist yet (Stage B builds it); collected now only to prove the test file itself is well-formed
Python and to fix, in writing, what Stage B's binding must satisfy. `pytest.importorskip` below
is what makes `python -m pytest --collect-only` report a clean SKIP rather than a collection
error while the extension is absent -- one of the two outcomes the plan step calls acceptable
("fail with a clear skip or import message, not a syntax error").
"""

import gc

import numpy as np
import pytest

pfuse = pytest.importorskip("pfuse", reason="Stage B has not built the _pfuse extension yet")

F0 = pfuse.FieldId(0)
F1 = pfuse.FieldId(1)


class TestBindingAliasesTheCallersStorage:
    def test_the_binding_is_a_view_not_a_copy(self):
        data = np.zeros(4)
        reg = pfuse.FieldRegistry()
        reg.bind_numpy(F0, data)
        view = reg.get_numpy(F0)
        view[1] = 7.5
        assert data[1] == 7.5, "the binding must alias the component's storage"

    def test_writes_to_the_original_are_seen_through_the_view(self):
        data = np.zeros(4)
        reg = pfuse.FieldRegistry()
        reg.bind_numpy(F0, data)
        data[:] = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(reg.get_numpy(F0), data)

    def test_the_view_shares_the_buffer_not_just_the_values(self):
        data = np.zeros(4)
        reg = pfuse.FieldRegistry()
        reg.bind_numpy(F0, data)
        view = reg.get_numpy(F0)
        assert (view.__array_interface__["data"][0]
                == data.__array_interface__["data"][0]), \
            "the view must start at the same address"
        assert not view.flags.owndata, "a view that owns its data is a copy"

    def test_two_dimensional_arrays_bind_interleaved(self):
        """C-contiguous (count, components) is AoS: fuse::Layout.component_stride == 0."""
        data = np.zeros((5, 3))
        reg = pfuse.FieldRegistry()
        reg.bind_numpy(F0, data)
        view = reg.get_numpy(F0)
        assert view.shape == (5, 3)
        view[2, 1] = -4.25
        assert data[2, 1] == -4.25

    def test_bookkeeping(self):
        reg = pfuse.FieldRegistry()
        assert reg.bound_count == 0
        assert not reg.is_bound(F0)
        reg.bind_numpy(F0, np.zeros(3))
        assert reg.bound_count == 1
        assert reg.is_bound(F0)
        assert not reg.is_bound(F1)


class TestRejectsWhatWouldBeTheWrongMemory:
    @pytest.fixture(autouse=True)
    def _registry(self):
        self.reg = pfuse.FieldRegistry()

    def test_rejects_non_contiguous_column_slice(self):
        data = np.zeros((4, 3))[:, 0]  # strided view
        assert not data.flags.c_contiguous
        with pytest.raises(ValueError, match="C-contiguous"):
            self.reg.bind_numpy(F0, data)

    def test_rejects_a_step_slice(self):
        data = np.zeros(8)[::2]
        assert not data.flags.c_contiguous
        with pytest.raises(pfuse.InvalidArgument):
            self.reg.bind_numpy(F0, data)

    def test_rejects_a_transpose(self):
        """A transposed (3, 5) array is F-contiguous, i.e. the wrong reading."""
        data = np.zeros((3, 5)).T
        assert data.shape == (5, 3)
        assert not data.flags.c_contiguous
        with pytest.raises(pfuse.InvalidArgument):
            self.reg.bind_numpy(F0, data)

    def test_accepts_the_repaired_transpose_but_it_is_a_different_buffer(self):
        original = np.zeros((3, 5))
        repaired = np.ascontiguousarray(original.T)
        self.reg.bind_numpy(F0, repaired)
        self.reg.get_numpy(F0)[0, 0] = 9.0
        assert repaired[0, 0] == 9.0
        assert original[0, 0] == 0.0, "ascontiguousarray copies; the original is NOT bound"

    def test_rejects_float32(self):
        with pytest.raises(pfuse.InvalidArgument, match="float64"):
            self.reg.bind_numpy(F0, np.zeros(4, dtype=np.float32))

    def test_rejects_int64(self):
        with pytest.raises(pfuse.InvalidArgument):
            self.reg.bind_numpy(F0, np.zeros(4, dtype=np.int64))

    def test_rejects_big_endian_float64(self):
        data = np.zeros(4, dtype=">f8")
        if data.dtype == np.dtype(np.float64):
            pytest.skip("this machine is big-endian; >f8 is native")
        with pytest.raises(pfuse.InvalidArgument):
            self.reg.bind_numpy(F0, data)

    def test_rejects_a_read_only_array(self):
        data = np.zeros(4)
        data.setflags(write=False)
        with pytest.raises(pfuse.InvalidArgument, match="read-only"):
            self.reg.bind_numpy(F0, data)

    def test_rejects_a_list(self):
        with pytest.raises(pfuse.InvalidArgument, match="numpy.ndarray"):
            self.reg.bind_numpy(F0, [1.0, 2.0, 3.0])

    def test_rejects_a_memoryview(self):
        with pytest.raises(pfuse.InvalidArgument):
            self.reg.bind_numpy(F0, memoryview(bytearray(32)))

    def test_rejects_three_dimensions(self):
        with pytest.raises(pfuse.InvalidArgument, match="dimensions"):
            self.reg.bind_numpy(F0, np.zeros((2, 3, 4)))

    def test_rejects_a_zero_dimensional_array(self):
        with pytest.raises(pfuse.InvalidArgument):
            self.reg.bind_numpy(F0, np.float64(1.0).reshape(()))

    def test_rebinding_is_refused_not_silently_replaced(self):
        """fuse::FieldRegistry::bind's own E-FIELD-REBIND: a fuse::Result failure, not a thrown
        fuse::Error -- so it is a BARE pfuse.FuseError carrying .code, per the owner's ruling
        (2026-10-12) that every Result failure becomes FuseError with no further triage."""
        first = np.zeros(4)
        second = np.ones(4)
        self.reg.bind_numpy(F0, first)
        with pytest.raises(pfuse.FuseError, match="E-FIELD-REBIND") as excinfo:
            self.reg.bind_numpy(F0, second)
        assert excinfo.value.code == "E-FIELD-REBIND"
        # The first binding is untouched: on any error the registry is left exactly as it was
        # (fuse::FieldRegistry::bind's own documented guarantee).
        self.reg.get_numpy(F0)[0] = 3.0
        assert first[0] == 3.0
        assert second[0] == 1.0

    def test_get_numpy_of_an_unbound_field_raises(self):
        with pytest.raises(pfuse.FuseError) as excinfo:
            self.reg.get_numpy(F1)
        assert excinfo.value.code == "E-FIELD-UNBOUND"

    def test_a_refused_binding_leaves_nothing_bound(self):
        with pytest.raises(pfuse.InvalidArgument):
            self.reg.bind_numpy(F0, np.zeros(8)[::2])
        assert not self.reg.is_bound(F0)
        assert self.reg.bound_count == 0


class TestLifetime:
    """The registry must keep the array alive; fuse has no other owner (field_ref.hpp)."""

    def test_a_temporary_expression_is_kept_alive(self):
        reg = pfuse.FieldRegistry()
        # Nothing in the caller's scope references this array after the call.
        reg.bind_numpy(F0, np.zeros(4) + 2.5)
        gc.collect()
        view = reg.get_numpy(F0)
        np.testing.assert_array_equal(view, [2.5, 2.5, 2.5, 2.5])
        view[3] = 1.0
        np.testing.assert_array_equal(reg.get_numpy(F0), [2.5, 2.5, 2.5, 1.0])

    def test_the_array_outlives_the_callers_last_name_for_it(self):
        reg = pfuse.FieldRegistry()
        data = np.arange(4, dtype=np.float64)
        reg.bind_numpy(F0, data)
        del data
        gc.collect()
        np.testing.assert_array_equal(reg.get_numpy(F0), [0.0, 1.0, 2.0, 3.0])

    def test_numpy_refuses_to_resize_a_bound_array(self):
        """A resize reallocates, which would dangle the pointer fuse::FieldRef holds.

        numpy refuses to resize an array something else references (refcheck=True is resize's
        own default), and the registry's strong reference is exactly such a reference. The
        control case runs first: an UNBOUND array of the identical shape must resize
        successfully, so a numpy that had stopped doing the refcount check would be caught here
        as an environment skip rather than passing the bound case vacuously.
        """
        control = np.zeros(4)
        try:
            control.resize(8, refcheck=True)
        except ValueError:
            pytest.skip("this interpreter holds extra references to locals; "
                        "numpy's resize refcheck is not usable here")

        reg = pfuse.FieldRegistry()
        bound = np.zeros(4)
        reg.bind_numpy(F0, bound)
        with pytest.raises(ValueError):
            bound.resize(8, refcheck=True)
