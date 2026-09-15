#include "karma_transfer_map_io.h"

#include <iostream>
#include <map>

int main()
{
    KARMA::KARMATransferMap map(KARMA::TransferMapRepresentation::ProjectionFactors, 1);
    KARMA::FieldEntryKey target;
    target.kind = KARMA::FieldEntryKey::Kind::Dof;
    target.dof.words[0] = 2;
    KARMA::FieldEntryKey source;
    source.kind = KARMA::FieldEntryKey::Kind::Dof;
    source.dof.words[0] = 1;
    map.add_row({target, {source}, {2.0}, KARMA::CoverageClass::Full});
    const std::map<KARMA::FieldEntryKey, KARMA::Real> values{{source, 3.0}};
    const std::vector<KARMA::FieldEntryKey> owned{target};
    const std::vector<KARMA::FieldEntryKey> source_local{source};
    try
    {
        const auto rhs = map.apply_b(values, owned, source_local);
        if (rhs.at(target) != 6.0) return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    try
    {
        const std::vector<KARMA::FieldEntryKey> missing{target};
        (void)map.apply_b(values, owned, missing);
        return 1;
    }
    catch (const std::logic_error&)
    {
    }
    return 0;
}
