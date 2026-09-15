#include "karma_field.h"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int result = 0;
    try
    {
        KARMA::KARMAMesh mesh(1, MPI_COMM_SELF);
        mesh.generate_box(1, 0, 0);
        mesh.ensure_global_ids();
        moab::Tag tag = nullptr;
        double initial = 1.0;
        if (mesh.moab()->tag_get_handle("TRANSFER_TAG", 1, moab::MB_TYPE_DOUBLE, tag,
                                         moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &initial) != moab::MB_SUCCESS)
            throw std::runtime_error("failed to create commit tag");
        KARMA::KARMAField field(KARMA::MoabTagBinding{&mesh, tag, KARMA::FieldLocation::Vertex, 1});
        std::vector<std::pair<KARMA::FieldEntryKey, KARMA::Real>> updates;
        for (const auto& entry : field.entries()) updates.push_back({entry, 3.0});
        field.commit_moab_values(updates);
        std::vector<double> values(mesh.vertices().size());
        if (mesh.moab()->tag_get_data(tag, mesh.vertices(), values.data()) != moab::MB_SUCCESS)
            throw std::runtime_error("failed to read committed tag");
        for (const double value : values)
            if (value != 3.0) throw std::runtime_error("MOAB commit did not publish values");
        std::swap(updates[0].first, updates[1].first);
        try
        {
            field.commit_moab_values(updates);
            throw std::runtime_error("invalid MOAB commit was accepted");
        }
        catch (const std::logic_error&)
        {
        }
        if (mesh.moab()->tag_get_data(tag, mesh.vertices(), values.data()) != moab::MB_SUCCESS)
            throw std::runtime_error("failed to reread committed tag");
        for (const double value : values)
            if (!std::isfinite(value) || value != 3.0) throw std::runtime_error("failed commit mutated tag");
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    MPI_Finalize();
    return result;
}
