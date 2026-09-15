#ifndef __karma_moab_io_h__
#define __karma_moab_io_h__

#include "karma_system.h"

#include <mpi.h>
#include <petscvec.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace KARMA
{
namespace detail
{
enum class CheckpointPublicationState { Unpublished, Published, Indeterminate };

struct CheckpointGatherStats
{
    std::uint64_t local_records = 0;
    std::uint64_t received_records = 0;
};

[[nodiscard]] CheckpointGatherStats checkpoint_gather_stats() noexcept;
[[nodiscard]] bool checkpoint_values_are_finite( const PetscScalar* values, PetscInt count ) noexcept;
void copy_checkpoint_values( const PetscScalar* source, PetscScalar* destination, PetscInt count ) noexcept;
[[nodiscard]] std::vector< unsigned char > encode_checkpoint_scalar( PetscScalar value );
[[nodiscard]] PetscScalar decode_checkpoint_scalar( std::span< const unsigned char > bytes );
[[nodiscard]] CheckpointPublicationState
checkpoint_result_collective_failed( CheckpointPublicationState state ) noexcept;
[[nodiscard]] std::string checkpoint_publication_result_error( CheckpointPublicationState state,
                                                               std::string_view error );
void sync_checkpoint_file( std::string_view filename );
void sync_checkpoint_directory( std::string_view directory );
void remove_checkpoint_temporaries( std::string_view directory, std::string_view snapshot,
                                    std::string_view staged );
void check_checkpoint_barrier( int error, std::string_view operation );
[[noreturn]] void throw_checkpoint_publication_durability_error( std::string_view error );
}

/**
 * Exact H5M checkpoint I/O.
 *
 * `KARMA_FIELD_MANIFEST` is a versioned root tag containing every field
 * descriptor and ordered chunk record. Rank zero reads it once and broadcasts
 * the exact bytes. Coefficient data is stored independently using three bindings:
 *
 * - `GEOMETRY_VERTEX`: cG nodal coefficients whose interpolation points map
 *   bijectively to the complete geometry connectivity. Payload is one value
 *   per component on each geometry vertex.
 * - `GLOBAL_DOF_ARRAY`: cG fields without a complete geometry-vertex bijection,
 *   including every 3-D cG field. Fixed-width stable DofKey/value records are
 *   split into bounded, checksummed chunks on tagged meshsets.
 * - `ELEMENT_PACKED`: dG and SE fields. Payload order is component-major, then
 *   local basis ordinal.
 *
 * Logical cG edge/face associations never require checkpoint edge/face entities.
 *
 * Checkpoint publication replaces the destination atomically and then confirms
 * crash durability by syncing its parent directory. If that final sync fails,
 * `write_checkpoint` throws `InvalidArgument` stating that the destination has
 * already been atomically replaced but crash durability could not be confirmed;
 * the published destination is not rolled back. Temporary checkpoint files and
 * their reservation directory are removed individually. Missing paths are
 * accepted, while any other removal failure is reported collectively when the
 * communicator remains usable. After an MPI failure, cleanup is local and
 * best-effort so no further collective is attempted.
 *
 * A coefficient tag is sparse only when occupancy is at most 20 percent.
 * Occupancy above 20 percent always uses a dense tag. Partial dense records
 * carry an explicit applicability meshset referenced by the manifest.
 */
class KARMALoadedCheckpoint
{
  public:
    [[nodiscard]] KARMAMesh& mesh() noexcept { return *_mesh; }
    [[nodiscard]] const KARMAMesh& mesh() const noexcept { return *_mesh; }
    [[nodiscard]] KARMASystem& system() noexcept { return *_system; }
    [[nodiscard]] const KARMASystem& system() const noexcept { return *_system; }

  private:
    friend class KARMAMoabIO;
    KARMALoadedCheckpoint( std::unique_ptr< KARMAMesh > mesh, std::unique_ptr< KARMASystem > system );

    std::unique_ptr< KARMAMesh > _mesh;
    std::unique_ptr< KARMASystem > _system;
};

class KARMAMoabIO
{
  public:
    explicit KARMAMoabIO( KARMAMesh& mesh ) noexcept;
    explicit KARMAMoabIO( const KARMAMesh& mesh ) noexcept;

    void read_mesh( std::string_view filename, std::string_view options = {} );
    void write_mesh( std::string_view filename, std::string_view options = {} ) const;

    void write_checkpoint( std::string_view filename, const KARMASystem& system, Vec solution,
                           std::string_view options = {} );
    void restore_solution( const KARMASystem& system, Vec solution );

    [[nodiscard]] static KARMALoadedCheckpoint
    load_checkpoint( std::string_view filename, MPI_Comm communicator,
                     KARMABasisRegistry registry = KARMABasisRegistry() );

  private:
    KARMAMesh* _input = nullptr;
    const KARMAMesh* _output = nullptr;
};
}  // namespace KARMA

#endif
