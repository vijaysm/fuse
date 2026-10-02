// A plain HDF5 program. Compare against consumer_hdf5.cpp: neither this
// file nor that one includes anything from this repository's own
// transport/runtime headers, and the test that is supposed to check that
// mechanically (grep, not a claim -- see docs/user/lowfive.md) names
// exactly these two files.
#include "producer_hdf5.h"

#include <hdf5.h>

int demo_b_produce(const char* file, const char* dataset, long long fapl_in,
                   const double* data, long n) {
  const hid_t fapl = (fapl_in >= 0) ? static_cast<hid_t>(fapl_in) : H5P_DEFAULT;
  const hid_t f = H5Fcreate(file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  if (f < 0) return -1;

  const hsize_t dim = static_cast<hsize_t>(n);
  const hid_t   space = H5Screate_simple(1, &dim, nullptr);
  if (space < 0) {
    H5Fclose(f);
    return -2;
  }
  const hid_t d = H5Dcreate2(f, dataset, H5T_IEEE_F64LE, space, H5P_DEFAULT,
                             H5P_DEFAULT, H5P_DEFAULT);
  if (d < 0) {
    H5Sclose(space);
    H5Fclose(f);
    return -3;
  }
  const herr_t wrote =
      H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  H5Dclose(d);
  H5Sclose(space);
  H5Fclose(f);
  return wrote < 0 ? -4 : 0;
}
