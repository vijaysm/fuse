#include "fuse/transport/transport.hpp"

#include <unordered_set>

namespace fuse {

void dedupe_flat_source_offers(
    const std::map<int, std::vector<std::pair<std::uint64_t, std::size_t>>>&
        by_peer,
    std::vector<std::size_t>* ids, std::vector<std::size_t>* idxs) {
  std::unordered_set<std::uint64_t> seen;
  seen.reserve(by_peer.size());
  for (const auto& [peer, offers] : by_peer) {
    (void)peer;
    for (const auto& [gid, idx] : offers) {
      if (seen.insert(gid).second) {
        ids->push_back(static_cast<std::size_t>(gid));
        idxs->push_back(idx);
      }
    }
  }
}

}  // namespace fuse
