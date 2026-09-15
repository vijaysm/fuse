#include "karma_operator_split_graph.h"
#include "karma_transient_reaction_diffusion_adapter.h"
#include "karma_mesh.h"

#include <petsc.h>

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <limits>
#include <algorithm>

namespace
{
PetscReal distributed_max(Vec vector)
{
    PetscInt local_size = 0;
    VecGetLocalSize(vector, &local_size);
    const PetscScalar* values = nullptr;
    VecGetArrayRead(vector, &values);
    PetscReal local = 0.0;
    for (PetscInt index = 0; index < local_size; ++index)
        local = std::max(local, std::abs(PetscRealPart(values[index])));
    VecRestoreArrayRead(vector, &values);
    PetscReal global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD);
    return global;
}

KARMA::OperatorSplitGraphSpec graph()
{
    const auto make_component = [](const std::string& id) {
        const KARMA::FieldSpec output{id + ".state", "1", 1, KARMA::FieldRole::Prognostic};
        const KARMA::FieldSpec input{id + ".coupled", "1", 1, KARMA::FieldRole::Input};
        return KARMA::ComponentSpec{id, {{"state", KARMA::PortDirection::Output, false, output},
                                          {"coupled", KARMA::PortDirection::Input, true, input}}, true, true};
    };
    KARMA::OperatorSplitGraphSpec result;
    result.id = "gray-scott-real-transfer";
    result.components = {{{"U", make_component("U")}, {"V", make_component("V")}}};
    result.connections = {{{"U", "state", "V", "coupled"}, {"V", "state", "U", "coupled"}}};
    result.scheme = KARMA::SplitScheme::Strang;
    return result;
}
}

int main(int argc, char** argv)
{
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int status = 0;
    try
    {
        KARMA::KARMAMesh u_mesh(2, PETSC_COMM_WORLD), v_mesh(2, PETSC_COMM_WORLD);
        u_mesh.generate_box(2, 2, 0);
        v_mesh.generate_box(3, 2, 0);
        KARMA::TransientReactionDiffusionNativeAdapter u("U", u_mesh, KARMA::GrayScottSpecies::U, {}, 1.0);
        KARMA::TransientReactionDiffusionNativeAdapter v("V", v_mesh, KARMA::GrayScottSpecies::V, {}, 0.2);
        const auto compiled = KARMA::compile_operator_split_graph(graph(), PETSC_COMM_WORLD);
        KARMA::KARMACouplingScheduler scheduler(compiled, {&u, &v});
        scheduler.advance(0, 0.02);
        const auto setup_counts = scheduler.connection_setup_counts();
        if (setup_counts[0] != 1 || setup_counts[1] != 1)
            throw std::runtime_error("Strang exchange rebuilt a compiled FEM connection");
        const auto u_output = std::get<KARMA::PetscFieldBinding>(u.output_field("state", KARMA::FieldVisibility::Committed).binding());
        const auto v_output = std::get<KARMA::PetscFieldBinding>(v.output_field("state", KARMA::FieldVisibility::Committed).binding());
        const auto u_input = std::get<KARMA::PetscFieldBinding>(u.input_field("coupled").binding());
        const auto v_input = std::get<KARMA::PetscFieldBinding>(v.input_field("coupled").binding());
        PetscReal u_norm = distributed_max(u_output.vector), v_norm = distributed_max(v_output.vector),
                  u_input_norm = distributed_max(u_input.vector), v_input_norm = distributed_max(v_input.vector);
        if (!std::isfinite(u_norm) || !std::isfinite(v_norm) || !std::isfinite(u_input_norm) || !std::isfinite(v_input_norm) ||
             std::abs(u_norm - 1.0) < 1.e-10 || std::abs(v_norm - 0.2) < 1.e-10 ||
             std::abs(u_input_norm) < 1.e-14 || std::abs(v_input_norm) < 1.e-14 || scheduler.position().step != 1)
        {
            std::cerr << "Gray-Scott norms: U=" << u_norm << " V=" << v_norm << " U input=" << u_input_norm
                      << " V input=" << v_input_norm << '\n';
            status = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    PetscFinalize();
    return status;
}
