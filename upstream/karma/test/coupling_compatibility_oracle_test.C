#include "karma_operator_split_checkpoint.h"
#include "karma_transient_reaction_diffusion_adapter.h"
#include "karma_transfer_map_io.h"

#include <petsc.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace KARMA;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void check(PetscErrorCode code)
{
    require(code == PETSC_SUCCESS, "Operator-split fixture PETSc operation failed");
}

ComponentSpec component(const std::string& id)
{
    return {id, {{"state", PortDirection::Output, false, {id + ".state", "1", 1, FieldRole::Prognostic}},
                 {"coupled", PortDirection::Input, true, {id + ".coupled", "1", 1, FieldRole::Input}}}, true, true};
}

OperatorSplitGraphSpec graph(SplitScheme scheme)
{
    return {"operator-split-oracle", {{{"U", component("U")}, {"V", component("V")}}},
            {{{"U", "state", "V", "coupled"}, {"V", "state", "U", "coupled"}}}, scheme};
}

// Explicit semantic fixtures captured from the existing coupler. Keep the
// component, connection, and instruction portions readable during migration.
std::string expected_graph_records()
{
    const std::string metadata =
        "|component=U:U:1:1:mesh=unknown:layout=unknown:basis=unknown:solver=unknown"
        "|coupled:0:1:U.coupled:1:1:1:2:0|state:1:0:U.state:1:1:0:2:0"
        "|component=V:V:1:1:mesh=unknown:layout=unknown:basis=unknown:solver=unknown"
        "|coupled:0:1:V.coupled:1:1:1:2:0|state:1:0:V.state:1:1:0:2:0"
        "|connection=U:state>V:coupled:transfer=persistent scalar cG projection:solver=PETSc options database"
        "|connection=V:state>U:coupled:transfer=persistent scalar cG projection:solver=PETSc options database";
    const std::string lie =
        "|instruction=checkpoint::::3|instruction=exchange::V:U:3|instruction=advance:U:::0"
        "|instruction=exchange::U:V:3|instruction=advance:V:::0"
        "|instruction=commit::::3|instruction=release::::3\n";
    const std::string strang =
        "|instruction=checkpoint::::3|instruction=exchange::V:U:3|instruction=advance:U:::1"
        "|instruction=exchange::U:V:3|instruction=advance:V:::0"
        "|instruction=exchange::V:U:3|instruction=advance:U:::2"
        "|instruction=commit::::3|instruction=release::::3\n";
    return "graph=operator-split-oracle|scheme=Lie" + metadata + lie +
           "graph=operator-split-oracle|scheme=Strang" + metadata + strang;
}

constexpr std::string_view expected_runtime_records =
    "map=1:8:1:17:23\n" // schema, scalar bytes, components, source/target signatures
    // scenario, status, iterations, checkpoints, restores, publications
    "fixed=0:converged:1:1:1:1\n" // publish exactly once
    "fixed=1:iteration-limit:3:1:4:0\n" // restore once more after the last trial
    "fixed=2:diverged:2:1:3:0\n" // never publish
    "fixed=3:nonfinite:1:1:2:0\n" // restore the checkpoint
    "fixed=4:sweep-failure:1:1:2:0\n" // all ranks roll back
    "fixed=5:rollback-failure:1:1:2:0\n" // terminal, never publish
    "restart=KARMAOSR:1:1:0.125:1\n"; // magic, schema, Strang, committed time/step

std::string graph_records()
{
    std::string records;
    for (const auto scheme : {SplitScheme::Lie, SplitScheme::Strang})
    {
        const auto compiled = compile_operator_split_graph(graph(scheme), PETSC_COMM_WORLD);
        auto reordered = graph(scheme);
        std::reverse(reordered.components.begin(), reordered.components.end());
        std::reverse(reordered.connections.begin(), reordered.connections.end());
        for (auto& item : reordered.components) std::reverse(item.component.ports.begin(), item.component.ports.end());
        require(compiled.signature() == compile_operator_split_graph(reordered, PETSC_COMM_WORLD).signature(),
                "Operator-split container reordering changed graph identity");
        records += compiled.signature() + '\n';
    }
    return records;
}

void key_ordering()
{
    // Explicit keys exercise variable, component, basis-slot and identity order.
    // Dense PETSc numbering and MOAB handles are deliberately absent.
    std::vector<DofKey> expected(5);
    for (auto& key : expected)
    {
        key[3] = 1;
        key[5] = 1;
        key[6] = 10;
    }
    expected[1][6] = 11;
    expected[2][4] = 1;
    expected[3][1] = 1;
    expected[4][0] = 1;
    auto shuffled = expected;
    std::reverse(shuffled.begin(), shuffled.end());
    std::sort(shuffled.begin(), shuffled.end());
    require(shuffled == expected && DofKey::width == 32, "Operator-split DoF key ordering/width changed");
    std::vector<FieldEntryKey> fields(4);
    fields[0].global_id = fields[1].global_id = 7;
    fields[1].component = 1;
    fields[2].global_id = 8;
    fields[3].kind = FieldEntryKey::Kind::Dof;
    fields[3].dof = expected[0];
    auto reversed = fields;
    std::reverse(reversed.begin(), reversed.end());
    std::sort(reversed.begin(), reversed.end());
    require(reversed == fields, "Operator-split field key ordering changed");
}

std::string transfer_records(const std::string& prefix, int rank)
{
    const auto filename = prefix + ".map." + std::to_string(rank);
    FieldEntryKey source;
    source.global_id = 7;
    FieldEntryKey target;
    target.kind = FieldEntryKey::Kind::Dof;
    target.dof[5] = 1;
    target.dof[6] = 11;
    KARMATransferMap map(TransferMapRepresentation::ProjectionFactors, 1);
    map.add_row({target, {source}, {0.5}, CoverageClass::Full});
    TransferMapFileMetadata written_metadata;
    written_metadata.source_signature = 17;
    written_metadata.target_signature = 23;
    save_transfer_map(filename, map, written_metadata);
    TransferMapFileMetadata metadata;
    const auto loaded = load_transfer_map(filename, &metadata);
    require(loaded.rows().size() == 1 && loaded.rows()[0].target == target &&
            loaded.rows()[0].source == std::vector<FieldEntryKey>{source} &&
            loaded.rows()[0].weights == std::vector<Real>{0.5} &&
            loaded.rows()[0].coverage == CoverageClass::Full &&
            loaded.representation() == TransferMapRepresentation::ProjectionFactors,
            "Operator-split map stable-key row changed");
    std::filesystem::remove(filename);
    return "map=" + std::to_string(metadata.schema_version) + ":" + std::to_string(metadata.scalar_width) +
           ":" + std::to_string(metadata.components) + ":" + std::to_string(metadata.source_signature) +
           ":" + std::to_string(metadata.target_signature) + '\n';
}

std::string fixed_point_records()
{
    PetscOwner<Vec, VecDestroy> value;
    PetscOwner<Mat, MatDestroy> mass;
    check(VecCreateMPI(PETSC_COMM_WORLD, 1, PETSC_DECIDE, value.put()));
    check(VecSet(value.get(), 0));
    PetscInt size = 0, begin = 0, end = 0;
    check(VecGetSize(value.get(), &size));
    check(VecGetOwnershipRange(value.get(), &begin, &end));
    check(MatCreateAIJ(PETSC_COMM_WORLD, 1, 1, size, size, 1, nullptr, 0, nullptr, mass.put()));
    check(MatSetValue(mass.get(), begin, begin, 2, INSERT_VALUES));
    check(MatAssemblyBegin(mass.get(), MAT_FINAL_ASSEMBLY));
    check(MatAssemblyEnd(mass.get(), MAT_FINAL_ASSEMBLY));
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    std::ostringstream records;
    for (int scenario = 0; scenario != 6; ++scenario)
    {
        int commits = 0, checkpoints = 0, restores = 0;
        FixedPointOptions options{3, 0, 0, 1, false};
        options.divergence_factor = 2;
        const auto result = KARMAFixedPointIterator(PETSC_COMM_WORLD, options).solve(
            {{"state", value.get(), mass.get(), {}}}, [&] { ++checkpoints; },
            [&] { ++restores; return scenario != 5; },
            [&](const auto& current, const auto& mapped) {
                if (scenario == 4 && rank == 0) throw std::runtime_error("oracle sweep failure");
                PetscScalar input = 0;
                check(VecGetValues(current[0].values, 1, &begin, &input));
                const Real output = scenario == 0 ? 0 : scenario == 2 ? 8 * input + 1 :
                    scenario == 3 ? std::numeric_limits<Real>::quiet_NaN() : input + 1;
                // Local array writes keep the asymmetric throw clear of collectives.
                PetscScalar* local = nullptr;
                check(VecGetArray(mapped[0].values, &local));
                local[0] = output;
                check(VecRestoreArray(mapped[0].values, &local));
            }, [&](const auto&) { ++commits; });
        const auto status_name = [](FixedPointStatus status) {
            switch (status)
            {
            case FixedPointStatus::Converged: return "converged";
            case FixedPointStatus::MaxIterations: return "iteration-limit";
            case FixedPointStatus::Diverged: return "diverged";
            case FixedPointStatus::Nonfinite: return "nonfinite";
            case FixedPointStatus::SweepFailure: return "sweep-failure";
            case FixedPointStatus::RollbackFailure: return "rollback-failure";
            }
            return "unknown";
        };
        records << "fixed=" << scenario << ':' << status_name(result.status) << ':'
                << result.iterations << ':' << checkpoints << ':' << restores << ':' << commits << '\n';
    }
    return records.str();
}

std::string restart_records(const std::string& prefix)
{
    KARMAMesh u_mesh(2, PETSC_COMM_WORLD), v_mesh(2, PETSC_COMM_WORLD);
    u_mesh.generate_box(2, 2, 0);
    v_mesh.generate_box(3, 2, 0);
    TransientReactionDiffusionNativeAdapter u("U", u_mesh, GrayScottSpecies::U, {}, 1.0);
    TransientReactionDiffusionNativeAdapter v("V", v_mesh, GrayScottSpecies::V, {}, 0.2);
    const auto compiled = compile_operator_split_graph(graph(SplitScheme::Strang), PETSC_COMM_WORLD);
    KARMACouplingScheduler scheduler(compiled, {&u, &v});
    scheduler.advance(0, 0.125);
    const OperatorSplitCheckpointBundle bundle{prefix + ".u.h5m", prefix + ".v.h5m", prefix + ".meta"};
    write_operator_split_checkpoint_bundle(bundle, compiled, {&u, &v}, scheduler.position());
    const auto prepared = prepare_operator_split_restart(bundle, compiled, {&u, &v});
    require(prepared.position().time == 0.125 && prepared.position().step == 1,
            "Operator-split safe-point position changed");
    std::ifstream input(bundle.metadata, std::ios::binary);
    const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input), {}};
    require(bytes.size() >= 64, "Operator-split restart metadata is truncated");
    const auto word = [&](std::size_t offset) {
        std::uint64_t result = 0;
        for (unsigned index = 0; index < 8; ++index) result |= std::uint64_t{bytes.at(offset + index)} << (index * 8);
        return result;
    };
    const std::string signature(bytes.begin() + 64, bytes.end());
    require(word(56) == signature.size() && signature == compiled.signature(),
            "Operator-split restart graph signature changed");
    // Payload file digests are intentionally not golden: HDF5 byte layout is not semantic identity.
    require(word(40) != 0 && word(48) != 0, "Operator-split restart payload digests missing");
    std::ostringstream records;
    records << "restart=" << std::string(bytes.begin(), bytes.begin() + 8) << ':' << word(8) << ':'
            << word(16) << ':' << std::bit_cast<double>(word(24)) << ':' << word(32) << '\n';
    return records.str();
}
}

int main(int argc, char** argv)
{
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int rank = 0, failure = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    try
    {
        require(argc == 2, "expected a unique fixture file prefix");
        // Reconcile local fixture/IO assertions before entering another collective.
        const auto collectively = [&](const auto& operation) {
            std::string result;
            int local_failure = 0;
            try { result = operation(); }
            catch (const std::exception& error)
            {
                std::cerr << "rank " << rank << ": " << error.what() << '\n';
                local_failure = 1;
            }
            int any_failure = 0;
            MPI_Allreduce(&local_failure, &any_failure, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
            require(any_failure == 0, "compatibility fixture failed collectively");
            return result;
        };
        collectively([] { key_ordering(); return std::string{}; });
        // Separate statements also make collective evaluation order explicit.
        auto records = collectively([] { return graph_records(); });
        records += collectively([&] { return transfer_records(argv[1], rank); });
        records += collectively([] { return fixed_point_records(); });
        records += collectively([&] { return restart_records(argv[1]); });
        if (rank == 0) std::cout << records;
        require(records == expected_graph_records() + std::string(expected_runtime_records),
                "coupling compatibility records changed; preserve semantics or provide an explicit migration");
    }
    catch (const std::exception& error)
    {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        failure = 1;
    }
    int global_failure = 0;
    MPI_Allreduce(&failure, &global_failure, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
    PetscFinalize();
    return global_failure;
}
