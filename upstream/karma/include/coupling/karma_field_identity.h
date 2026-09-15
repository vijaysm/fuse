#ifndef KARMA_COUPLING_FIELD_IDENTITY_H
#define KARMA_COUPLING_FIELD_IDENTITY_H

#include "karma_types.h"

#include <array>
#include <compare>
#include <cstdint>

/** Stable field-entry identity, dependency-free.
 *
 * `DofKey` and `FieldEntryKey` are plain data: no PETSc, MOAB, or MPI type
 * appears here or in anything this header includes. They used to be declared
 * inside `karma_dof_map.h` and `karma_field.h`, which pull PETSc, MOAB, and
 * `mpi.h` for reasons that have nothing to do with identity - `KARMADofMap`
 * for its PETSc-numbered index maps, `KARMAField` for its `Vec`/MOAB `Tag`
 * bindings. Three coupling headers need only the identity types
 * (`karma_field_entry_encoding.h`, `karma_field_view.h`, and, through it,
 * `karma_field_transaction.h`), and a language adapter binding the
 * compliance harness in `karma_component_runtime.h` reaches
 * `karma_field_transaction.h` in turn - so pulling in `karma_field.h` for a
 * struct definition meant an MPI dependency arrived at the language-adapter
 * boundary for no reason. This header is what those three now include
 * instead; `karma_dof_map.h` and `karma_field.h` both include it too, so the
 * type is declared in exactly one place.
 */
namespace KARMA
{
/** Stable DoF identity serialized as:
 * variable, component, entity dimension, MOAB entity type, canonical slot,
 * identity count, then three words per identity. Conventional meshes use
 * positive MOAB GLOBAL_ID values in the first identity word; unused words are
 * zero. Intermediate cG entities use canonically ordered vertex identities.
 */
struct DofKey
{
    static constexpr std::size_t width = 32;
    std::array<std::uint64_t, width> words{};

    [[nodiscard]] auto begin() noexcept { return words.begin(); }
    [[nodiscard]] auto begin() const noexcept { return words.begin(); }
    [[nodiscard]] auto end() noexcept { return words.end(); }
    [[nodiscard]] auto end() const noexcept { return words.end(); }
    [[nodiscard]] std::uint64_t& operator[](std::size_t index) noexcept { return words[index]; }
    [[nodiscard]] const std::uint64_t& operator[](std::size_t index) const noexcept { return words[index]; }
    auto operator<=>(const DofKey&) const = default;
};

struct FieldEntryKey
{
    enum class Kind
    {
        MoabGlobalId,
        Dof
    };

    Kind kind = Kind::MoabGlobalId;
    std::uint64_t global_id = 0;
    unsigned component = 0;
    DofKey dof;
    auto operator<=>(const FieldEntryKey&) const = default;
};
} // namespace KARMA

#endif
