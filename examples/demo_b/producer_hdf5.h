#ifndef DEMO_B_PRODUCER_HDF5_H
#define DEMO_B_PRODUCER_HDF5_H

#ifdef __cplusplus
extern "C" {
#endif

/* Writes `n` doubles to `dataset` inside `file` using plain HDF5 calls
 * (H5Fcreate, H5Dcreate2, H5Dwrite, H5Fclose) -- nothing else. Whatever
 * file-access property list the CALLER passes (`fapl`, an HDF5 hid_t) is
 * used for H5Fcreate, exactly as an ordinary HDF5 program would use
 * H5P_DEFAULT; this program has no idea whether that property list carries
 * an interception VOL connector or not, which is the entire point of Demo
 * B (docs/user/lowfive.md). Returns 0 on success, negative on any HDF5
 * failure. */
int demo_b_produce(const char* file, const char* dataset,
                    long long fapl, const double* data, long n);

#ifdef __cplusplus
}
#endif

#endif /* DEMO_B_PRODUCER_HDF5_H */
