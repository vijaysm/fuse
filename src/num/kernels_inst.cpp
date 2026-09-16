#include "fuse/num/kernels.hpp"

namespace fuse::num {

template void   copy<double>(const double*, double*, std::size_t);
template void   copy<float>(const float*, float*, std::size_t);
template double norm_l2<double>(const double*, std::size_t);
template float  norm_l2<float>(const float*, std::size_t);
template double weighted_sum<double>(const double*, const double*, std::size_t);

}  // namespace fuse::num
