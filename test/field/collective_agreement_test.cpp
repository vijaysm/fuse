// fuse/field/collective_agreement.hpp at tier 0: the pure parts of the
// agreement (the message every rank carries, the Failure it is packed into,
// and the defaults of Failure's provenance fields) are pinned exactly, so the
// MPI rows (parallel.field_transaction_mpi.{2,3}, parallel.provider_subset_mpi.2)
// only have to show that the bytes arrive, not what they say. KNOWN_ISSUES.md
// D97 (2026-10-30): the original error and the rank it came from are known on
// every rank.

#include "fuse/field/collective_agreement.hpp"

#include <string>

#include <gtest/gtest.h>

#include "fuse/base/config.hpp"

namespace fuse {
namespace {

TEST(CollectiveAgreement, FailureProvenanceDefaultsToNoOrigin) {
  const Failure f{"E-X", "local"};
  EXPECT_EQ(f.origin_rank, -1);
  EXPECT_TRUE(f.origin_code.empty());
  EXPECT_EQ(f.origin_status, 0);
}

TEST(CollectiveAgreement, MessageCarriesRankCountOriginCodeAndOriginTextVerbatim) {
  AgreedRefusal r;
  r.origin_rank = 2;
  r.refusing_ranks = 1;
  r.communicator_size = 3;
  r.origin_code = "E-FIELD-TXN-INCOMPLETE";
  r.origin_message = "entry 1 was never staged";
  EXPECT_EQ(format_agreed_refusal("FieldTransaction::commit_collective (validation)", r),
            "FieldTransaction::commit_collective (validation): rank 2 of 3 refused "
            "(1 refusing rank; the lowest-numbered is reported) with "
            "E-FIELD-TXN-INCOMPLETE: entry 1 was never staged");
  r.refusing_ranks = 2;
  EXPECT_NE(format_agreed_refusal("p", r).find("(2 refusing ranks;"), std::string::npos);
}

TEST(CollectiveAgreement, AgreedFailureCarriesTheAgreedCodeAndTheOriginAsData) {
  AgreedRefusal r;
  r.origin_rank = 0;
  r.refusing_ranks = 1;
  r.communicator_size = 2;
  r.origin_code = "E-ABI-VERSION";
  r.origin_message = "plugin built against ABI 3, host offers 2";
  r.origin_status = -7;
  const Failure f = make_agreed_failure("E-ABI-PLUGIN-AGREEMENT-REFUSED", "fuse_initialize", r);
  EXPECT_EQ(f.code, "E-ABI-PLUGIN-AGREEMENT-REFUSED");
  EXPECT_EQ(f.origin_rank, 0);
  EXPECT_EQ(f.origin_code, "E-ABI-VERSION");
  EXPECT_EQ(f.origin_status, -7);
  EXPECT_EQ(f.message, format_agreed_refusal("fuse_initialize", r));
  EXPECT_NE(f.message.find("plugin built against ABI 3, host offers 2"), std::string::npos);
}

#if !FUSE_HAVE_MPI
// Without MPI there is no communicator to agree over; the helper refuses
// rather than hand back the local result as if it had been agreed.
TEST(CollectiveAgreement, WithoutMpiAnyCommunicatorIsRefusedNotSilentlyLocal) {
  const auto r = agree_outcome(/*fortran_communicator=*/1, Result<void>::success(), "E-X",
                               "E-X-COMM", "p");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-X-COMM");
}
#endif

}  // namespace
}  // namespace fuse
