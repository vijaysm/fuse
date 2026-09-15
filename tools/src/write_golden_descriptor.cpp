// Regenerates a frozen descriptor golden file. Run BY A HUMAN, deliberately,
// when kGraphVersion is bumped -- never by a test, and never in CI: a golden
// file a test can rewrite is not a golden file.
//
//   fuse_write_golden_descriptor test/golden/waist-v2.bin
//
// The previous file is kept under its own name, so a reader can see what the
// bytes used to be (docs/dev/abi-design.md).
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "fuse/model/descriptor.hpp"
#include "fuse/model/freeze_fixture.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: fuse_write_golden_descriptor <path>\n";
    return 2;
  }
  const auto bytes = fuse::serialize(fuse::testing::freeze_fixture());
  std::ofstream out(argv[1], std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  std::cerr << "wrote " << bytes.size() << " bytes to " << argv[1] << "\n";
  return out ? 0 : 1;
}
