// A plain HDF5 program. See producer_hdf5.cpp's own comment.
#include "consumer_hdf5.h"

#include <hdf5.h>

int demo_b_consume(const char* file, const char* dataset, long long fapl_in,
                   double* out, long n) {
  const hid_t fapl = (fapl_in >= 0) ? static_cast<hid_t>(fapl_in) : H5P_DEFAULT;
  const hid_t f = H5Fopen(file, H5F_ACC_RDONLY, fapl);
  if (f < 0) return -1;

  const hid_t d = H5Dopen2(f, dataset, H5P_DEFAULT);
  if (d < 0) {
    H5Fclose(f);
    return -2;
  }
  const herr_t read =
      H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
  (void)n;  // the caller sized `out`; HDF5 itself knows the dataspace extent

  H5Dclose(d);
  H5Fclose(f);
  return read < 0 ? -3 : 0;
}
