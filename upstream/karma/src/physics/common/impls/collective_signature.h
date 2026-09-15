#ifndef KARMA_COLLECTIVE_SIGNATURE_H
#define KARMA_COLLECTIVE_SIGNATURE_H

#include "karma_dof_map.h"
#include "karma_exceptions.h"

#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace KARMA::detail
{
inline void signature_mpi_check( int error, const char* operation )
{
    if( error != MPI_SUCCESS ) KARMA_THROW( InvalidArgument( std::string( "MPI error in " ) + operation ) );
}

inline void collective_signature_error( MPI_Comm communicator, std::string_view local_error,
                                        bool* communicator_usable = nullptr )
{
    const auto check = [&]( int error, const char* operation ) {
        if( error != MPI_SUCCESS && communicator_usable != nullptr ) *communicator_usable = false;
        signature_mpi_check( error, operation );
    };
    int rank = 0, ranks = 1;
    check( MPI_Comm_rank( communicator, &rank ), "MPI_Comm_rank(signature validation)" );
    check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size(signature validation)" );
    const int candidate = local_error.empty() ? ranks : rank;
    int failing_rank = ranks;
    check( MPI_Allreduce( &candidate, &failing_rank, 1, MPI_INT, MPI_MIN, communicator ),
           "MPI_Allreduce(signature validation)" );
    if( failing_rank == ranks ) return;

    int message_size = 0;
    bool local_ready = true;
    if( rank == failing_rank )
        try { message_size = checked_mpi_count( local_error.size() ); }
        catch( ... ) { local_ready = false; }
    const int ready = local_ready ? 1 : 0;
    int all_ready = 0;
    check( MPI_Allreduce( &ready, &all_ready, 1, MPI_INT, MPI_MIN, communicator ),
           "MPI_Allreduce(signature message size)" );
    if( !all_ready ) KARMA_THROW( InvalidArgument( "collective signature error message is too large" ) );

    check( MPI_Bcast( &message_size, 1, MPI_INT, failing_rank, communicator ),
           "MPI_Bcast(signature validation size)" );
    std::string message;
    try
    {
        if( rank == failing_rank ) message.assign( local_error );
        else message.resize( static_cast< std::size_t >( message_size ) );
    }
    catch( ... ) { local_ready = false; }
    const int buffer_ready = local_ready ? 1 : 0;
    all_ready = 0;
    check( MPI_Allreduce( &buffer_ready, &all_ready, 1, MPI_INT, MPI_MIN, communicator ),
           "MPI_Allreduce(signature message allocation)" );
    if( !all_ready ) KARMA_THROW( InvalidArgument( "collective signature error message allocation failed" ) );

    check( MPI_Bcast( message.data(), message_size, MPI_CHAR, failing_rank, communicator ),
           "MPI_Bcast(signature validation message)" );
    KARMA_THROW( InvalidArgument( "rank " + std::to_string( failing_rank ) + ": " + message ) );
}

template < class Preparation >
void prepare_collectively( MPI_Comm communicator, Preparation&& prepare, std::string_view failure )
{
    bool failed = false;
    try { std::forward< Preparation >( prepare )(); }
    catch( ... ) { failed = true; }
    collective_signature_error( communicator, failed ? failure : std::string_view{} );
}

template < class SignatureProducer >
void require_collective_signature( MPI_Comm communicator, std::string_view local_error,
                                   SignatureProducer&& produce_signature,
                                   std::string_view construction_error, std::string_view mismatch_error )
{
    int ranks = 1;
    signature_mpi_check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size(signature gather)" );
    std::string signature;
    std::vector< int > sizes;
    int size = 0;
    bool construction_failed = false;
    try
    {
        signature = std::forward< SignatureProducer >( produce_signature )();
        size = checked_mpi_count( signature.size() );
        sizes.resize( static_cast< std::size_t >( ranks ) );
    }
    catch( ... ) { construction_failed = true; }
    collective_signature_error( communicator, construction_failed ? construction_error : local_error );

    signature_mpi_check( MPI_Allgather( &size, 1, MPI_INT, sizes.data(), 1, MPI_INT, communicator ),
                         "MPI_Allgather(signature sizes)" );
    MpiLayout layout;
    std::string signatures;
    bool layout_failed = false;
    try
    {
        layout = checked_mpi_displacements( sizes, 1 );
        signatures.resize( layout.total_count );
    }
    catch( ... ) { layout_failed = true; }
    collective_signature_error( communicator,
                                layout_failed ? "signature gather layout or allocation failed" : std::string_view{} );

    signature_mpi_check( MPI_Allgatherv( signature.data(), size, MPI_CHAR, signatures.data(), sizes.data(),
                                         layout.displacements.data(), MPI_CHAR, communicator ),
                         "MPI_Allgatherv(signatures)" );
    int failing_rank = ranks;
    for( int i = 1; i < ranks; ++i )
        if( sizes[i] != sizes[0] ||
            signatures.compare( layout.displacements[i], sizes[i], signatures, 0, sizes[0] ) != 0 )
        {
            failing_rank = i;
            break;
        }
    int rank = 0;
    signature_mpi_check( MPI_Comm_rank( communicator, &rank ), "MPI_Comm_rank(signature mismatch)" );
    collective_signature_error( communicator, rank == failing_rank ? mismatch_error : std::string_view{} );
}
}  // namespace KARMA::detail

#endif
