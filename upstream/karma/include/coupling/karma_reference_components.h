#ifndef KARMA_COUPLING_REFERENCE_COMPONENTS_H
#define KARMA_COUPLING_REFERENCE_COMPONENTS_H

#include "karma_component_runtime.h"

/** In-tree native components, registered statically.
 *
 * These exist so the runtime contract has something real to drive that needs
 * neither PETSc nor MPI, which is what makes the compliance harness runnable
 * anywhere the library builds. They are reference implementations, not
 * physics: an exponential decay with an analytic step.
 */
namespace KARMA
{
/** Configuration record for the reference components. Little-endian, fixed
 * width, and size-checked by the factory - the same discipline the ABI uses,
 * because a configuration blob is a boundary too. */
struct ReferenceDecayConfiguration
{
    std::uint64_t element_count = 0;
    double rate = 0.0;
};

/** Serializes the configuration into the bytes the ABI `create` call carries. */
[[nodiscard]] std::vector<std::byte> encode_reference_decay_configuration(const ReferenceDecayConfiguration& value);

/** The semantic contract both reference components satisfy.
 *
 * Exposed rather than kept private so the ABI built-in and the registered
 * decay component are described by ONE declaration. They are separate
 * implementations on purpose - the built-in exists to exercise the ABI's own
 * guards and the registered one computes an analytic decay - but a single
 * shared contract is what stops their declared shape drifting apart
 * unnoticed, which is the risk that having two of them creates.
 */
[[nodiscard]] ComponentSemantics reference_state_contract(const std::string& id, SemanticVersion version);

/** Registers `karma.reference.decay.baseline` and `karma.reference.decay.extended`, both
 * satisfying the `karma.contract.scalar-decay` contract. Two of them on
 * purpose: one implementation cannot demonstrate that selection refuses to
 * substitute. Idempotent per registry - a second call reports the duplicate
 * rather than replacing anything. */
[[nodiscard]] RegistryReason register_reference_components(ComponentRegistry& registry);
} // namespace KARMA

#endif
