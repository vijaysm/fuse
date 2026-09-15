#include "karma_transfer_map_io.h"

#include "karma_exceptions.h"

#include <bit>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unistd.h>

namespace KARMA
{
namespace
{
constexpr std::uint64_t magic = 0x4b41524d41544d50ULL;

void write_bytes(std::ofstream& output, const void* data, std::size_t size)
{
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!output) KARMA_THROW(InvalidArgument("failed to write transfer map"));
}

void read_bytes(std::ifstream& input, void* data, std::size_t size)
{
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!input) KARMA_THROW(InvalidArgument("transfer map is truncated or unreadable"));
}

std::uint64_t checksum_words(std::span<const std::uint64_t> words)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto word : words)
        for (unsigned byte = 0; byte < sizeof(word); ++byte)
        {
            hash ^= (word >> (8 * byte)) & 0xffU;
            hash *= 1099511628211ULL;
        }
    return hash;
}
}

void save_transfer_map(std::string_view filename, const KARMATransferMap& map, TransferMapFileMetadata metadata)
{
    map.verify();
    if (map.representation() == TransferMapRepresentation::ProjectionFactors && metadata.target_signature == 0)
        KARMA_THROW(InvalidArgument("projection-factor maps require a target mass signature"));
    metadata.components = map.components();
    std::vector<std::uint64_t> words;
    words.push_back(magic);
    words.push_back(metadata.schema_version);
    words.push_back(metadata.scalar_width);
    words.push_back(static_cast<std::uint64_t>(metadata.components));
    words.push_back(metadata.source_signature);
    words.push_back(metadata.target_signature);
    words.push_back(static_cast<std::uint64_t>(map.representation()));
    words.push_back(map.rows().size());
    for (const auto& row : map.rows())
    {
        words.push_back(static_cast<std::uint64_t>(row.target.kind));
        words.push_back(row.target.global_id);
        words.push_back(row.target.component);
        words.insert(words.end(), row.target.dof.begin(), row.target.dof.end());
        words.push_back(static_cast<std::uint64_t>(row.coverage));
        words.push_back(row.source.size());
        for (std::size_t index = 0; index < row.source.size(); ++index)
        {
            words.push_back(static_cast<std::uint64_t>(row.source[index].kind));
            words.push_back(row.source[index].global_id);
            words.push_back(row.source[index].component);
            words.insert(words.end(), row.source[index].dof.begin(), row.source[index].dof.end());
            words.push_back(std::bit_cast<std::uint64_t>(row.weights[index]));
        }
    }
    const std::uint64_t checksum = checksum_words(words);
    const std::filesystem::path destination{std::string(filename)};
    const auto temporary = destination.string() + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) KARMA_THROW(InvalidArgument("failed to open transfer map temporary file"));
    write_bytes(output, words.data(), words.size() * sizeof(std::uint64_t));
    write_bytes(output, &checksum, sizeof(checksum));
    output.flush();
    output.close();
    const int descriptor = ::open(temporary.c_str(), O_RDONLY);
    if (descriptor < 0) KARMA_THROW(InvalidArgument("failed to open transfer map temporary file for sync"));
    if (::fsync(descriptor) != 0)
    {
        ::close(descriptor);
        KARMA_THROW(InvalidArgument("failed to sync transfer map temporary file"));
    }
    ::close(descriptor);
    std::filesystem::rename(temporary, destination);
}

KARMATransferMap load_transfer_map(std::string_view filename, TransferMapFileMetadata* metadata)
{
    std::ifstream input(std::string(filename), std::ios::binary);
    if (!input) KARMA_THROW(InvalidArgument("failed to open transfer map for reading"));
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size < static_cast<std::streamoff>(sizeof(std::uint64_t) * 9))
        KARMA_THROW(InvalidArgument("transfer map file is too small"));
    input.seekg(0);
    const std::size_t word_count = static_cast<std::size_t>(size) / sizeof(std::uint64_t) - 1;
    std::vector<std::uint64_t> words(word_count);
    read_bytes(input, words.data(), words.size() * sizeof(std::uint64_t));
    std::uint64_t stored_checksum = 0;
    read_bytes(input, &stored_checksum, sizeof(stored_checksum));
    if (checksum_words(words) != stored_checksum) KARMA_THROW(InvalidArgument("transfer map checksum mismatch"));
    std::size_t offset = 0;
    const auto next = [&]() {
        if (offset >= words.size()) KARMA_THROW(InvalidArgument("transfer map record is truncated"));
        return words[offset++];
    };
    if (next() != magic) KARMA_THROW(InvalidArgument("transfer map magic is invalid"));
    TransferMapFileMetadata loaded;
    loaded.schema_version = static_cast<std::uint32_t>(next());
    loaded.scalar_width = static_cast<std::uint32_t>(next());
    loaded.components = static_cast<std::uint32_t>(next());
    loaded.source_signature = next();
    loaded.target_signature = next();
    const auto representation = static_cast<TransferMapRepresentation>(next());
    if (loaded.schema_version != 1 || loaded.scalar_width != sizeof(Real) || loaded.components == 0)
        KARMA_THROW(InvalidArgument("transfer map metadata is unsupported"));
    KARMATransferMap map(representation, loaded.components);
    const auto row_count = next();
    for (std::uint64_t row_index = 0; row_index < row_count; ++row_index)
    {
        SparseInterpolationRow row;
        row.target.kind = static_cast<FieldEntryKey::Kind>(next());
        row.target.global_id = next();
        row.target.component = static_cast<unsigned>(next());
        for (auto& word : row.target.dof.words) word = next();
        row.coverage = static_cast<CoverageClass>(next());
        const auto source_count = next();
        if (source_count > std::numeric_limits<std::uint32_t>::max())
            KARMA_THROW(InvalidArgument("transfer map source row is too large"));
        row.source.resize(static_cast<std::size_t>(source_count));
        row.weights.resize(static_cast<std::size_t>(source_count));
        for (std::size_t source_index = 0; source_index < row.source.size(); ++source_index)
        {
            row.source[source_index].kind = static_cast<FieldEntryKey::Kind>(next());
            row.source[source_index].global_id = next();
            row.source[source_index].component = static_cast<unsigned>(next());
            for (auto& word : row.source[source_index].dof.words) word = next();
            row.weights[source_index] = std::bit_cast<Real>(next());
        }
        map.add_row(std::move(row));
    }
    if (offset != words.size()) KARMA_THROW(InvalidArgument("transfer map contains trailing records"));
    map.verify();
    if (metadata) *metadata = loaded;
    return map;
}
}  // namespace KARMA
