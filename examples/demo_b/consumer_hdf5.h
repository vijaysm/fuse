#ifndef DEMO_B_CONSUMER_HDF5_H
#define DEMO_B_CONSUMER_HDF5_H

#ifdef __cplusplus
extern "C" {
#endif

/* Reads `n` doubles from `dataset` inside `file` using plain HDF5 calls
 * (H5Fopen, H5Dopen2, H5Dread, H5Fclose) -- nothing else. See
 * producer_hdf5.h's own comment; the same reasoning applies here. Returns
 * 0 on success, negative on any HDF5 failure. */
int demo_b_consume(const char* file, const char* dataset, long long fapl,
                   double* out, long n);

#ifdef __cplusplus
}
#endif

#endif /* DEMO_B_CONSUMER_HDF5_H */
