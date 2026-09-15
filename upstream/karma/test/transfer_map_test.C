#include "karma_field_coverage.h"

#include <iostream>

int main()
{
    KARMA::FillPolicy fill(2.0, 2);
    fill.validate(2);
    try
    {
        fill.validate(1);
        return 1;
    }
    catch (const std::logic_error&)
    {
    }

    KARMA::KARMATransferMap map(KARMA::TransferMapRepresentation::SparseInterpolation, 1);
    KARMA::FieldEntryKey target;
    KARMA::FieldEntryKey source;
    source.global_id = 1;
    map.add_row({target, {source}, {1.0}, KARMA::CoverageClass::Full});
    map.verify();
    try
    {
        map.add_row({target, {source}, {1.0}, KARMA::CoverageClass::Full});
        return 1;
    }
    catch (const std::logic_error&)
    {
    }
    return 0;
}
