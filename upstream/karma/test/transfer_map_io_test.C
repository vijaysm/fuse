#include "karma_transfer_map_io.h"

#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2) return 1;
    const std::string filename = argv[1];
    try
    {
        KARMA::KARMATransferMap map(KARMA::TransferMapRepresentation::SparseInterpolation, 1);
        KARMA::FieldEntryKey target;
        target.kind = KARMA::FieldEntryKey::Kind::Dof;
        target.dof.words[0] = 11;
        KARMA::FieldEntryKey source;
        source.kind = KARMA::FieldEntryKey::Kind::Dof;
        source.dof.words[0] = 7;
        map.add_row({target, {source}, {1.0}, KARMA::CoverageClass::Full});
        KARMA::save_transfer_map(filename, map, {1, sizeof(KARMA::Real), 1, 17, 23});
        KARMA::TransferMapFileMetadata metadata;
        const auto loaded = KARMA::load_transfer_map(filename, &metadata);
        KARMA::KARMATransferMap factors(KARMA::TransferMapRepresentation::ProjectionFactors, 1);
        factors.add_row({target, {source}, {1.0}, KARMA::CoverageClass::Full});
        KARMA::save_transfer_map(filename + ".factors", factors, {1, sizeof(KARMA::Real), 1, 17, 23});
        (void)KARMA::load_transfer_map(filename + ".factors");
        if (metadata.source_signature != 17 || metadata.target_signature != 23 || loaded.rows().size() != 1)
            return 1;
        std::fstream corrupt(filename, std::ios::in | std::ios::out | std::ios::binary);
        corrupt.seekp(0);
        char byte = 0;
        corrupt.write(&byte, 1);
        corrupt.close();
        try
        {
            (void)KARMA::load_transfer_map(filename);
            return 1;
        }
        catch (const std::logic_error&)
        {
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
