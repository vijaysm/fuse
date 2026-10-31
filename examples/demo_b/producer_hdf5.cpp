// A plain HDF5 program. Compare against consumer_hdf5.cpp: neither this
// file nor that one includes anything from this repository's own
// transport/runtime headers, and the test that is supposed to check that
// mechanically (grep, not a claim -- see docs/user/lowfive.md) names
// exactly these two files.
#include "producer_hdf5.h"

#include <hdf5.h>

#include <string>

int demo_b_produce(const char* file, const char* dataset, long long fapl_in,
                   const double* data, long n) {
  const hid_t fapl = (fapl_in >= 0) ? static_cast<hid_t>(fapl_in) : H5P_DEFAULT;
  const hid_t f = H5Fcreate(file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  if (f < 0) return -1;

  // H5Dcreate2 does not create intermediate groups: "/grid/values" needs
  // "/grid" first, in plain HDF5 as under any VOL connector (LowFive's
  // metadata VOL otherwise records a single root child named "grid/values").
  // Create every missing parent in one call.
  const std::string path(dataset);
  const std::size_t last = path.find_last_of('/');
  if (last != std::string::npos && last > 0) {
    const hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
    if (lcpl < 0) {
      H5Fclose(f);
      return -5;
    }
    H5Pset_create_intermediate_group(lcpl, 1);
    const hid_t g = H5Gcreate2(f, path.substr(0, last).c_str(), lcpl,
                               H5P_DEFAULT, H5P_DEFAULT);
    H5Pclose(lcpl);
    if (g < 0) {
      H5Fclose(f);
      return -6;
    }
    H5Gclose(g);
  }
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
