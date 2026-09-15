#include "generic_mass_operator.h"
#include "karma_moab_io.h"
#include "karma_system.h"
#include "petsc_raii.h"

#include <petsc.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
int failures = 0;

void check( bool condition, const char* message )
{
    if( !condition )
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void add_reference_element( KARMA::KARMAMesh& mesh, KARMA::GeometryVariant variant )
{
    const KARMA::KARMAReferenceTopology topology( variant );
    std::vector< double > coordinates;
    for( const auto& point : topology.reference_nodes() )
        for( unsigned d = 0; d < 3; ++d )
            coordinates.push_back( point( d ) );
    moab::Range vertices;
    mesh.moab()->create_vertices( coordinates.data(), static_cast< int >( topology.node_count() ), vertices );
    std::vector< moab::EntityHandle > connectivity( vertices.begin(), vertices.end() );
    moab::EntityHandle element = 0;
    mesh.moab()->create_element( topology.topology(), connectivity.data(), static_cast< int >( connectivity.size() ),
                                 element );
    mesh.assign_ids();
    mesh.update();
}

void check_simplex_mass( unsigned dimension, KARMA::GeometryVariant variant, Real measure )
{
    KARMA::KARMAMesh mesh( dimension, PETSC_COMM_WORLD );
    add_reference_element( mesh, variant );
    KARMA::KARMASystem system( mesh, "simplex-mass" );
    system.add_variable( "u" );
    system.init();
    const auto mass        = KARMA::element_mass_matrix( system, *mesh.elements().begin() );
    const std::size_t size = dimension + 1;
    const Real scale       = measure / ( size * ( size + 1 ) );
    for( std::size_t i = 0; i < size; ++i )
        for( std::size_t j = 0; j < size; ++j )
            check( std::abs( mass( i, j ) - scale * ( i == j ? 2 : 1 ) ) < 1.e-12,
                   "simplex P1 mass has analytic entries" );
}

bool throws_with( auto&& function, std::string_view text )
{
    try
    {
        function();
    }
    catch( const std::logic_error& error )
    {
        return std::string_view( error.what() ).find( text ) != std::string_view::npos;
    }
    return false;
}

struct MassReference
{
    std::vector< std::uint64_t > elements;
    std::vector< KARMA::DofKey > keys;
    std::vector< PetscReal > matrix;
    std::vector< PetscReal > product;
};

void require( bool condition, const char* message )
{
    if( !condition ) throw std::runtime_error( message );
}

std::map< KARMA::DofKey, PetscInt > visible_dof_indices( const KARMA::KARMASystem& system )
{
    std::map< KARMA::DofKey, PetscInt > result;
    for( const auto element : system.mesh().elements() )
    {
        std::vector< PetscInt > indices;
        system.dof_map().dof_indices( element, indices );
        const auto& keys = system.dof_map().dof_keys( element );
        require( keys.size() == indices.size(), "mass keys and indices differ in size" );
        for( std::size_t i = 0; i < keys.size(); ++i )
        {
            const auto [found, inserted] = result.emplace( keys[i], indices[i] );
            require( inserted || found->second == indices[i], "stable mass key has inconsistent index" );
        }
    }
    return result;
}

MassReference read_mass_reference( const char* filename )
{
    std::ifstream input( filename );
    require( input.good(), "open mass reference" );
    MassReference reference;
    std::size_t element_count = 0;
    std::size_t dof_count = 0;
    require( static_cast< bool >( input >> element_count >> dof_count ), "read mass reference header" );
    reference.elements.resize( element_count );
    for( auto& element : reference.elements ) require( static_cast< bool >( input >> element ), "read mass element id" );
    reference.keys.resize( dof_count );
    for( auto& key : reference.keys )
        for( auto& word : key.words ) require( static_cast< bool >( input >> word ), "read mass DoF key" );
    reference.matrix.resize( dof_count * dof_count );
    for( auto& value : reference.matrix )
        require( static_cast< bool >( input >> value ), "read mass matrix value" );
    reference.product.resize( dof_count );
    for( auto& value : reference.product )
        require( static_cast< bool >( input >> value ), "read mass product value" );
    return reference;
}

void write_partitioned_mass_reference( const char* mesh_filename, const char* reference_filename )
{
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMA::KARMAMoabIO( mesh ).read_mesh( mesh_filename );
    mesh.update();
    KARMA::KARMASystem system( mesh, "serial-partitioned-mass-reference" );
    system.add_variable( "u" );
    system.init();
    Mat mass = system.create_matrix();
    KARMA::assemble_mass_matrix( system, mass );
    const auto indices = visible_dof_indices( system );
    require( indices.size() == static_cast< std::size_t >( system.n_dofs() ), "serial mass reference sees every DoF" );
    std::vector< KARMA::DofKey > keys;
    std::vector< PetscInt > serial_indices;
    keys.reserve( indices.size() );
    for( const auto& [key, index] : indices )
    {
        keys.push_back( key );
        serial_indices.push_back( index );
    }
    std::vector< PetscScalar > matrix( keys.size() * keys.size() );
    require( MatGetValues( mass, system.n_dofs(), serial_indices.data(), system.n_dofs(), serial_indices.data(), matrix.data() ) ==
                 PETSC_SUCCESS,
             "read serial mass matrix" );
    Vec input = system.create_vector(), product = system.create_vector();
    for( std::size_t i = 0; i < serial_indices.size(); ++i )
        require( VecSetValue( input, serial_indices[i], static_cast< PetscScalar >( i + 1 ), INSERT_VALUES ) == PETSC_SUCCESS,
                 "set serial mass input" );
    require( VecAssemblyBegin( input ) == PETSC_SUCCESS && VecAssemblyEnd( input ) == PETSC_SUCCESS,
             "assemble serial mass input" );
    require( MatMult( mass, input, product ) == PETSC_SUCCESS, "multiply serial mass matrix" );
    std::vector< PetscScalar > values( keys.size() );
    require( VecGetValues( product, system.n_dofs(), serial_indices.data(), values.data() ) == PETSC_SUCCESS,
             "read serial mass product" );
    std::set< std::uint64_t > elements;
    for( const auto element : system.dof_map().assembly_elements() )
        elements.insert( system.dof_map().canonical_element_ordinal( element ) );
    require( elements.size() == system.dof_map().assembly_elements().size(), "serial mass elements have unique identities" );
    std::ofstream output( reference_filename );
    require( output.good(), "open mass reference output" );
    output << elements.size() << ' ' << keys.size() << '\n';
    for( const auto element : elements ) output << element << '\n';
    for( const auto& key : keys )
    {
        for( const auto word : key.words ) output << word << ' ';
        output << '\n';
    }
    output << std::setprecision( 17 );
    for( const auto value : matrix ) output << PetscRealPart( value ) << '\n';
    for( const auto value : values ) output << PetscRealPart( value ) << '\n';
    VecDestroy( &product );
    VecDestroy( &input );
    MatDestroy( &mass );
}

void verify_partitioned_mass( const char* mesh_filename, const char* reference_filename, bool mutate_cross_coupling = false )
{
    const MassReference reference = read_mass_reference( reference_filename );
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMA::KARMAMoabIO( mesh ).read_mesh(
        mesh_filename, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" );
    mesh.update();
    KARMA::KARMASystem system( mesh, "partitioned-mass" );
    system.add_variable( "u" );
    system.init();
    Mat mass = system.create_matrix();
    KARMA::assemble_mass_matrix( system, mass );
    const auto visible = visible_dof_indices( system );
    std::map< KARMA::DofKey, PetscInt > reference_indices;
    for( std::size_t i = 0; i < reference.keys.size(); ++i )
        require( reference_indices.emplace( reference.keys[i], static_cast< PetscInt >( i ) ).second,
                 "serial mass reference keys are unique" );
    int ranks = 1;
    require( MPI_Comm_size( PETSC_COMM_WORLD, &ranks ) == MPI_SUCCESS, "query mass rank count" );
    std::vector< std::uint64_t > local_words;
    local_words.reserve( visible.size() * ( KARMA::DofKey::width + 1 ) );
    for( const auto& [key, index] : visible )
    {
        local_words.insert( local_words.end(), key.begin(), key.end() );
        local_words.push_back( static_cast< std::uint64_t >( index ) );
    }
    const int local_count = static_cast< int >( local_words.size() );
    std::vector< int > counts( static_cast< std::size_t >( ranks ) );
    require( MPI_Allgather( &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "gather mass key counts" );
    std::vector< int > offsets( counts.size() );
    for( std::size_t i = 1; i < offsets.size(); ++i ) offsets[i] = offsets[i - 1] + counts[i - 1];
    const int total_count = offsets.back() + counts.back();
    std::vector< std::uint64_t > gathered( static_cast< std::size_t >( total_count ) );
    require( MPI_Allgatherv( local_words.empty() ? nullptr : local_words.data(), local_count, MPI_UINT64_T,
                             gathered.empty() ? nullptr : gathered.data(), counts.data(), offsets.data(),
                             MPI_UINT64_T, PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "gather mass keys and indices" );
    std::map< KARMA::DofKey, PetscInt > partitioned_indices;
    constexpr std::size_t record_width = KARMA::DofKey::width + 1;
    require( gathered.size() % record_width == 0, "gathered mass key records are complete" );
    for( std::size_t offset = 0; offset < gathered.size(); offset += record_width )
    {
        KARMA::DofKey key;
        std::copy_n( gathered.begin() + static_cast< std::ptrdiff_t >( offset ), KARMA::DofKey::width, key.begin() );
        const auto [found, inserted] = partitioned_indices.emplace( key, static_cast< PetscInt >( gathered[offset + KARMA::DofKey::width] ) );
        require( inserted || found->second == static_cast< PetscInt >( gathered[offset + KARMA::DofKey::width] ),
                 "partitioned stable mass key maps to one index" );
    }
    check( partitioned_indices.size() == reference_indices.size(), "partitioned mass exposes every serial stable DoF" );
    std::set< PetscInt > partitioned_index_values;
    for( const auto& [key, index] : partitioned_indices )
    {
        (void)key;
        partitioned_index_values.insert( index );
    }
    check( partitioned_index_values.size() == partitioned_indices.size(),
           "partitioned stable mass DoF identities map bijectively to PETSc indices" );
    check( !partitioned_index_values.empty() && *partitioned_index_values.begin() == 0 &&
               *partitioned_index_values.rbegin() + 1 == system.n_dofs() &&
               std::adjacent_find( partitioned_index_values.begin(), partitioned_index_values.end(),
                                   []( PetscInt first, PetscInt second ) { return second != first + 1; } ) ==
                   partitioned_index_values.end(),
           "partitioned mass PETSc indices form the deterministic contiguous owner-major layout" );
    for( const auto& [key, reference_index] : reference_indices )
    {
        (void)reference_index;
        const auto found = partitioned_indices.find( key );
        check( found != partitioned_indices.end(), "partitioned mass preserves every stable DoF identity" );
    }
    std::set< std::uint64_t > local_elements;
    for( const auto element : system.dof_map().assembly_elements() )
        local_elements.insert( system.dof_map().canonical_element_ordinal( element ) );
    const int local_element_count = static_cast< int >( local_elements.size() );
    std::vector< int > element_counts( static_cast< std::size_t >( ranks ) );
    require( MPI_Allgather( &local_element_count, 1, MPI_INT, element_counts.data(), 1, MPI_INT, PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "gather owned mass element counts" );
    std::vector< int > element_offsets( element_counts.size() );
    for( std::size_t i = 1; i < element_offsets.size(); ++i ) element_offsets[i] = element_offsets[i - 1] + element_counts[i - 1];
    std::vector< std::uint64_t > all_elements( static_cast< std::size_t >( element_offsets.back() + element_counts.back() ) );
    const std::vector< std::uint64_t > local_element_values( local_elements.begin(), local_elements.end() );
    require( MPI_Allgatherv( local_element_values.empty() ? nullptr : local_element_values.data(), local_element_count, MPI_UINT64_T, all_elements.data(), element_counts.data(),
                             element_offsets.data(), MPI_UINT64_T, PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "gather owned mass elements" );
    std::sort( all_elements.begin(), all_elements.end() );
    check( std::adjacent_find( all_elements.begin(), all_elements.end() ) == all_elements.end(),
           "old replicated-element MASS traversal would duplicate owned elements" );
    check( all_elements == reference.elements, "every serial element contributes once in partitioned MASS assembly" );
    PetscInt first = 0, last = 0;
    require( MatGetOwnershipRange( mass, &first, &last ) == PETSC_SUCCESS, "get partitioned mass row ownership" );
    check( first == system.dof_map().first_owned_dof() && last == system.dof_map().end_owned_dof(),
           "partitioned mass matrix row ownership follows the DoF owner-major layout" );
    int local_cross_partition_cg = 0;
    for( const auto element : system.dof_map().assembly_elements() )
    {
        std::vector< PetscInt > dofs;
        system.dof_map().dof_indices( element, dofs );
        if( std::any_of( dofs.begin(), dofs.end(), [first, last]( PetscInt dof ) { return dof < first || dof >= last; } ) )
            local_cross_partition_cg = 1;
    }
    int cross_partition_cg = 0;
    require( MPI_Allreduce( &local_cross_partition_cg, &cross_partition_cg, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "reduce cross-partition cG mass evidence" );
    check( cross_partition_cg == 1, "partitioned MASS inserts cG cross-partition rows and columns" );
    std::vector< PetscInt > columns;
    columns.reserve( reference.keys.size() );
    for( const auto& key : reference.keys ) columns.push_back( partitioned_indices.at( key ) );
    PetscInt mutated_row = -1, mutated_column = -1;
    PetscReal mutated_reference_value = 0;
    if( mutate_cross_coupling )
    {
        bool mutated = false;
        for( const auto& [key, row] : partitioned_indices )
        {
            if( row < first || row >= last ) continue;
            const PetscInt reference_row = reference_indices.at( key );
            for( std::size_t column = 0; column < columns.size(); ++column )
                if( columns[column] < first || columns[column] >= last )
                {
                    const PetscReal expected = reference.matrix[static_cast< std::size_t >( reference_row ) * columns.size() + column];
                    if( std::abs( expected ) > 1.e-12 )
                    {
                        require( MatSetValue( mass, row, columns[column], 0, INSERT_VALUES ) == PETSC_SUCCESS,
                                 "mutate partitioned cG mass coupling" );
                        mutated_row = row;
                        mutated_column = columns[column];
                        mutated_reference_value = expected;
                        mutated = true;
                        break;
                    }
                }
            if( mutated ) break;
        }
        int global_mutated = mutated ? 1 : 0;
        require( MPI_Allreduce( MPI_IN_PLACE, &global_mutated, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD ) == MPI_SUCCESS,
                 "reduce mass coupling mutation" );
        require( global_mutated > 0, "fixture has an owned nonzero cG cross-partition mass coupling" );
        require( MatAssemblyBegin( mass, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS &&
                     MatAssemblyEnd( mass, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
                 "assemble mutated partitioned mass matrix" );
    }
    bool local_missing_coupling_detected = false;
    for( const auto& [key, row] : partitioned_indices )
    {
        if( row < first || row >= last ) continue;
        const PetscInt reference_row = reference_indices.at( key );
        std::vector< PetscScalar > values( columns.size() );
        require( MatGetValues( mass, 1, &row, static_cast< PetscInt >( columns.size() ), columns.data(), values.data() ) ==
                     PETSC_SUCCESS,
                 "read owned partitioned mass row" );
        for( std::size_t column = 0; column < values.size(); ++column )
        {
            const PetscReal expected =
                reference.matrix[static_cast< std::size_t >( reference_row ) * columns.size() + column];
            const bool matches = std::abs( PetscRealPart( values[column] ) - expected ) < 1.e-12;
            if( mutate_cross_coupling && row == mutated_row && columns[column] == mutated_column )
                local_missing_coupling_detected = !matches && std::abs( PetscRealPart( values[column] ) ) < 1.e-12 &&
                                                   std::abs( expected - mutated_reference_value ) < 1.e-12;
            else
                check( matches, "partitioned mass matrix equals serial stable-key reference" );
        }
    }
    if( mutate_cross_coupling )
    {
        int missing_coupling_detected = local_missing_coupling_detected ? 1 : 0;
        require( MPI_Allreduce( MPI_IN_PLACE, &missing_coupling_detected, 1, MPI_INT, MPI_SUM,
                                PETSC_COMM_WORLD ) == MPI_SUCCESS,
                 "reduce missing cG mass coupling detection" );
        check( missing_coupling_detected > 0,
               "mutated partitioned mass matrix is missing the selected cG cross-partition coupling" );
    }
    Vec input = system.create_vector(), product = system.create_vector();
    for( const auto& [key, index] : partitioned_indices )
        if( index >= first && index < last )
            require( VecSetValue( input, index, static_cast< PetscScalar >( reference_indices.at( key ) + 1 ), INSERT_VALUES ) ==
                         PETSC_SUCCESS,
                     "set partitioned mass input by stable key" );
    require( VecAssemblyBegin( input ) == PETSC_SUCCESS && VecAssemblyEnd( input ) == PETSC_SUCCESS,
             "assemble partitioned mass input" );
    require( MatMult( mass, input, product ) == PETSC_SUCCESS, "multiply partitioned mass matrix" );
    bool local_product_discrepancy = false;
    for( const auto& [key, index] : partitioned_indices )
    {
        if( index < first || index >= last ) continue;
        PetscScalar value = 0;
        require( VecGetValues( product, 1, &index, &value ) == PETSC_SUCCESS, "read owned partitioned mass product" );
        const bool matches =
            std::abs( PetscRealPart( value ) -
                      reference.product[static_cast< std::size_t >( reference_indices.at( key ) )] ) < 1.e-12;
        if( mutate_cross_coupling && index == mutated_row ) local_product_discrepancy = !matches;
        else check( matches, "partitioned mass product equals serial stable-key reference" );
    }
    if( mutate_cross_coupling )
    {
        int product_discrepancy = local_product_discrepancy ? 1 : 0;
        require( MPI_Allreduce( MPI_IN_PLACE, &product_discrepancy, 1, MPI_INT, MPI_SUM,
                                PETSC_COMM_WORLD ) == MPI_SUCCESS,
                 "reduce mutated mass product discrepancy" );
        check( product_discrepancy > 0,
               "missing cG cross-partition coupling changes the partitioned mass product" );
    }
    VecDestroy( &product );
    VecDestroy( &input );
    MatDestroy( &mass );
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        if( mode == "--write-partitioned-mass-reference" )
        {
            check( argc == 4, "serial MASS reference mode requires mesh and reference paths" );
            if( argc == 4 ) write_partitioned_mass_reference( argv[2], argv[3] );
        }
        else if( mode == "--verify-partitioned-mass" )
        {
            check( argc == 4, "partitioned MASS mode requires mesh and reference paths" );
            if( argc == 4 ) verify_partitioned_mass( argv[2], argv[3] );
        }
        else if( mode == "--verify-partitioned-mass-mutation" )
        {
            check( argc == 4, "partitioned MASS mutation mode requires mesh and reference paths" );
            if( argc == 4 ) verify_partitioned_mass( argv[2], argv[3], true );
        }
        else if( mode.rfind( "--asymmetric-", 0 ) == 0 )
        {
            PetscOptionsClearValue( nullptr, std::string( mode ).c_str() );
            int rank = 0;
            MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
            KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
            mesh.generate_box( 1, 1, 0 );
            KARMA::KARMASystem system( mesh, "asymmetric-mass" );
            system.add_variable( "u" );
            system.init();
            Mat mass = system.create_matrix();
            Vec input = system.create_vector(), output = system.create_vector();
            Mat bad_matrix = nullptr;
            Vec bad_vector = nullptr;
            if( mode.find( "matrix" ) != std::string_view::npos && mode.find( "null" ) == std::string_view::npos )
            {
                const PetscInt extra = mode.find( "size" ) != std::string_view::npos && rank == 1 ? 1 : 0;
                const MPI_Comm communicator = mode.find( "comm" ) != std::string_view::npos ? PETSC_COMM_SELF
                                                                                             : PETSC_COMM_WORLD;
                MatCreateAIJ( communicator, system.n_local_dofs() + extra, system.n_local_dofs() + extra,
                              PETSC_DECIDE, PETSC_DECIDE, 1, nullptr, 0, nullptr, &bad_matrix );
                MatSetUp( bad_matrix );
            }
            if( mode.find( "input" ) != std::string_view::npos || mode.find( "output" ) != std::string_view::npos ||
                mode.find( "rhs" ) != std::string_view::npos || mode.find( "solution" ) != std::string_view::npos )
                if( mode.find( "null" ) == std::string_view::npos )
                {
                    const PetscInt extra = mode.find( "size" ) != std::string_view::npos && rank == 1 ? 1 : 0;
                    const MPI_Comm communicator = mode.find( "comm" ) != std::string_view::npos ? PETSC_COMM_SELF
                                                                                                 : PETSC_COMM_WORLD;
                    VecCreateMPI( communicator, system.n_local_dofs() + extra, PETSC_DECIDE, &bad_vector );
                }
            constexpr PetscReal output_sentinel = 37;
            constexpr PetscReal mass_sentinel = 41;
            check( VecSet( output, output_sentinel ) == PETSC_SUCCESS,
                   "asymmetric mass test initializes the vector sentinel" );
            check( MatAssemblyBegin( mass, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
                   "asymmetric mass test begins initial matrix assembly" );
            check( MatAssemblyEnd( mass, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
                   "asymmetric mass test ends initial matrix assembly" );
            check( MatShift( mass, mass_sentinel ) == PETSC_SUCCESS,
                   "asymmetric mass test initializes the matrix sentinel" );
            PetscReal initial_mass_norm = 0;
            check( MatNorm( mass, NORM_INFINITY, &initial_mass_norm ) == PETSC_SUCCESS &&
                       initial_mass_norm == mass_sentinel,
                   "asymmetric mass test observes the exact matrix sentinel before validation" );
            Mat call_matrix = mass;
            Vec call_input = input, call_output = output;
            const PetscInt first_dof = 0;
            const PetscInt second_dof = system.n_dofs() > 1 ? 1 : 0;
            const PetscInt invalid_dof = system.n_dofs();
            const PetscInt* dirichlet = nullptr;
            PetscInt n_dirichlet = 0;
            int quadrature_degree = 5;
            if( mode == "--asymmetric-dirichlet-count" && rank == 1 )
            {
                dirichlet = &first_dof;
                n_dirichlet = 1;
            }
            if( mode == "--asymmetric-dirichlet-list" )
            {
                dirichlet = rank == 1 ? &second_dof : &first_dof;
                n_dirichlet = 1;
            }
            if( mode == "--asymmetric-quadrature" && rank == 1 ) quadrature_degree = 4;
            if( mode == "--asymmetric-dirichlet-pointer" && rank == 1 ) n_dirichlet = 1;
            if( mode == "--asymmetric-dirichlet-range" && rank == 1 )
            {
                dirichlet = &invalid_dof;
                n_dirichlet = 1;
            }
            if( rank == 1 && mode.find( "matrix" ) != std::string_view::npos ) call_matrix = bad_matrix;
            if( rank == 1 && ( mode.find( "input" ) != std::string_view::npos ||
                              mode.find( "rhs" ) != std::string_view::npos ) )
                call_input = bad_vector;
            if( rank == 1 && ( mode.find( "output" ) != std::string_view::npos ||
                              mode.find( "solution" ) != std::string_view::npos ) )
                call_output = bad_vector;
            const bool matrix_mode = mode.find( "matrix" ) != std::string_view::npos;
            const bool solve_mode = mode.find( "rhs" ) != std::string_view::npos ||
                                    mode.find( "solution" ) != std::string_view::npos;
            check( throws_with( [&] {
                       if( matrix_mode || mode.find( "dirichlet" ) != std::string_view::npos ||
                           mode.find( "quadrature" ) != std::string_view::npos )
                           KARMA::assemble_mass_matrix( system, call_matrix, dirichlet, n_dirichlet,
                                                        quadrature_degree );
                       else if( solve_mode ) KARMA::mass_solve( system, mass, call_input, call_output );
                       else KARMA::mass_multiply( system, mass, call_input, call_output );
                   }, "rank" ),
                   "asymmetric mass object validation fails collectively" );
            PetscReal output_norm = 0, mass_norm = 0;
            const PetscErrorCode output_norm_error = VecNorm( output, NORM_INFINITY, &output_norm );
            const PetscErrorCode mass_norm_error = MatNorm( mass, NORM_INFINITY, &mass_norm );
            check( output_norm_error == PETSC_SUCCESS && mass_norm_error == PETSC_SUCCESS &&
                       output_norm == output_sentinel && mass_norm == mass_sentinel,
                   "asymmetric mass validation leaves exact sentinels unchanged" );
            VecDestroy( &bad_vector );
            MatDestroy( &bad_matrix );
            VecDestroy( &output );
            VecDestroy( &input );
            MatDestroy( &mass );
        }
        else
        {
        check_simplex_mass( 1, KARMA::GeometryVariant::Edge2, 1 );
        check_simplex_mass( 2, KARMA::GeometryVariant::Tri3, 0.5 );
        check_simplex_mass( 3, KARMA::GeometryVariant::Tet4, 1.0 / 6.0 );

        for( const auto [dimension, variant] :
             { std::pair{ 2u, KARMA::GeometryVariant::Tri3 }, std::pair{ 3u, KARMA::GeometryVariant::Tet4 } } )
        {
            KARMA::KARMAMesh simplex( dimension, PETSC_COMM_WORLD );
            add_reference_element( simplex, variant );
            KARMA::KARMASystem p2( simplex, "invalid-lumped-p2" );
            p2.add_variable( "u", KARMA::Order::Second );
            p2.init();
            Mat matrix   = p2.create_matrix();
            Vec rhs      = p2.create_vector();
            Vec solution = p2.create_vector();
            KARMA::assemble_mass_matrix( p2, matrix );
            VecSet( rhs, 1 );
            check( throws_with( [&] { KARMA::mass_solve( p2, matrix, rhs, solution, true ); }, "strictly positive" ),
                   "P2 simplex row-sum lumping rejects zero or negative lumps clearly" );
            VecDestroy( &solution );
            VecDestroy( &rhs );
            MatDestroy( &matrix );
        }

        KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        mesh.generate_box( 1, 1, 0 );
        KARMA::KARMASystem system( mesh, "mass" );
        system.add_variable( "u" );
        system.add_variable( "v" );
        system.init();

        const auto element = *mesh.elements().begin();
        {
            KARMA::KARMASystem p0( mesh, "p0-mass" );
            p0.add_variable( { "p", KARMA::FieldBasisFamily::LagrangeNodal, 0, KARMA::BasisContinuity::dG, 1 } );
            p0.init();
            const auto p0_mass = KARMA::element_mass_matrix( p0, element );
            check( p0_mass.rows() == 1 && std::abs( p0_mass( 0, 0 ) - 1 ) < 1.e-12,
                   "element-local P0 mass equals element measure" );
            Mat p0_matrix   = p0.create_matrix();
            Vec p0_rhs      = p0.create_vector();
            Vec p0_solution = p0.create_vector();
            KARMA::assemble_mass_matrix( p0, p0_matrix );
            VecSet( p0_rhs, 3 );
            KARMA::mass_solve( p0, p0_matrix, p0_rhs, p0_solution, true );
            PetscReal p0_error = 0;
            VecAXPY( p0_solution, -1, p0_rhs );
            VecNorm( p0_solution, NORM_INFINITY, &p0_error );
            check( p0_error < 1.e-12, "positive P0 row sum gives the exact lumped solve" );
            VecDestroy( &p0_solution );
            VecDestroy( &p0_rhs );
            MatDestroy( &p0_matrix );
        }
        {
            KARMA::KARMASystem p2( mesh, "p2-mass" );
            p2.add_variable( "q", KARMA::Order::Second );
            p2.init();
            const auto p2_mass = KARMA::element_mass_matrix( p2, element );
            check( p2_mass.rows() == 9 && p2_mass.native().isApprox( p2_mass.native().transpose(), 1.e-12 ) &&
                       p2_mass.native().selfadjointView< Eigen::Lower >().eigenvalues().minCoeff() > 0,
                   "P2 consistent element mass is symmetric positive definite" );
        }
        {
            KARMA::KARMASystem mixed( mesh, "mixed-mass" );
            mixed.add_variable( { "cell", KARMA::FieldBasisFamily::LagrangeNodal, 0, KARMA::BasisContinuity::dG, 1 } );
            mixed.add_variable( "quadratic", KARMA::Order::Second );
            mixed.init();
            const auto mixed_mass = KARMA::element_mass_matrix( mixed, element );
            check( mixed_mass.rows() == 10 && std::abs( mixed_mass( 0, 0 ) - 1 ) < 1.e-12 &&
                       std::abs( mixed_mass( 0, 1 ) ) < 1.e-12,
                   "mixed P0/P2 layout uses per-variable basis offsets and remains decoupled" );
        }
        const auto local = KARMA::element_mass_matrix( system, element );
        check( local.rows() == 8 && local.cols() == 8, "two P1 fields use variable-major element layout" );
        const Real scale          = 1.0 / 36.0;
        const Real expected[4][4] = { { 4, 2, 1, 2 }, { 2, 4, 2, 1 }, { 1, 2, 4, 2 }, { 2, 1, 2, 4 } };
        for( std::size_t i = 0; i < 8; ++i )
            for( std::size_t j = 0; j < 8; ++j )
            {
                const bool same_field = i / 4 == j / 4;
                const Real value      = same_field ? scale * expected[i % 4][j % 4] : 0;
                check( std::abs( local( i, j ) - value ) < 1.e-12,
                       "unit quad mass has analytic entries and no cross-field coupling" );
            }

        Mat mass = system.create_matrix();
        KARMA::assemble_mass_matrix( system, mass );
        std::vector< PetscInt > element_dofs;
        system.dof_map().dof_indices( element, element_dofs );
        const std::array< PetscInt, 2 > duplicate_dirichlet{ element_dofs.front(), element_dofs.front() };
        KARMA::assemble_mass_matrix( system, mass, duplicate_dirichlet.data(), duplicate_dirichlet.size() );
        PetscInt first_owned = 0, end_owned = 0;
        MatGetOwnershipRange( mass, &first_owned, &end_owned );
        if( duplicate_dirichlet.front() >= first_owned && duplicate_dirichlet.front() < end_owned )
        {
            PetscScalar constrained_diagonal = 0;
            MatGetValues( mass, 1, duplicate_dirichlet.data(), 1, duplicate_dirichlet.data(), &constrained_diagonal );
            check( std::abs( PetscRealPart( constrained_diagonal ) - 1 ) < 1.e-12,
                   "duplicate Dirichlet DoFs are eliminated once" );
        }
        KARMA::assemble_mass_matrix( system, mass );
        PetscReal norm = 0;
        MatNorm( mass, NORM_FROBENIUS, &norm );
        check( norm > 0, "global PETSc mass matrix is assembled" );
        PetscBool symmetric = PETSC_FALSE;
        MatIsSymmetric( mass, 1.e-12, &symmetric );
        check( symmetric == PETSC_TRUE, "global mass matrix is symmetric" );
        Vec diagonal = system.create_vector();
        MatGetDiagonal( mass, diagonal );
        PetscReal minimum = 0;
        VecMin( diagonal, nullptr, &minimum );
        check( minimum > 0, "global mass matrix has positive diagonal" );
        VecDestroy( &diagonal );

        Vec one       = system.create_vector();
        Vec product   = system.create_vector();
        Vec recovered = system.create_vector();
        VecSet( one, 1 );
        if( mesh.n_ranks() == 1 )
        {
            Vec plain = nullptr, wrong_size = nullptr;
            Mat plain_matrix = nullptr, wrong_size_matrix = nullptr;
            VecCreateMPI( PETSC_COMM_WORLD, system.n_local_dofs(), system.n_dofs(), &plain );
            VecCreateMPI( PETSC_COMM_WORLD, system.n_local_dofs() + 1, system.n_dofs() + 1, &wrong_size );
            MatCreateAIJ( PETSC_COMM_WORLD, system.n_local_dofs(), system.n_local_dofs(), system.n_dofs(),
                          system.n_dofs(), 1, nullptr, 0, nullptr, &plain_matrix );
            MatSetUp( plain_matrix );
            MatCreateAIJ( PETSC_COMM_WORLD, system.n_local_dofs() + 1, system.n_local_dofs() + 1,
                          system.n_dofs() + 1, system.n_dofs() + 1, 1, nullptr, 0, nullptr, &wrong_size_matrix );
            MatSetUp( wrong_size_matrix );
            VecSet( product, 37 );
            VecSet( recovered, 43 );
            MatShift( mass, 41 );
            check( throws_with( [&] { KARMA::assemble_mass_matrix( system, nullptr ); }, "null" ),
                   "mass assembly rejects a null matrix" );
            check( throws_with( [&] { KARMA::assemble_mass_matrix( system, wrong_size_matrix ); }, "dimensions" ),
                   "mass assembly rejects wrong matrix dimensions" );
            check( throws_with( [&] { KARMA::assemble_mass_matrix( system, plain_matrix ); }, "mapping" ),
                   "mass assembly rejects a plain same-sized matrix" );
            check( throws_with( [&] { KARMA::assemble_mass_matrix( system, mass, nullptr, 1 ); }, "pointer" ) &&
                       throws_with( [&] { KARMA::assemble_mass_matrix( system, mass, nullptr, -1 ); }, "nonnegative" ) &&
                       throws_with( [&] { KARMA::assemble_mass_matrix( system, mass, nullptr, 0, 0 ); }, "positive" ),
                    "mass assembly validates Dirichlet arguments and quadrature degree" );

            const PetscInt invalid_dof = system.n_dofs();
            check( throws_with( [&] { KARMA::assemble_mass_matrix( system, mass, &invalid_dof, 1 ); }, "range" ),
                   "mass assembly validates Dirichlet ranges" );
            check( throws_with( [&] { KARMA::mass_multiply( system, mass, nullptr, product ); }, "null" ) &&
                       throws_with( [&] { KARMA::mass_multiply( system, mass, wrong_size, product ); }, "size" ) &&
                       throws_with( [&] { KARMA::mass_multiply( system, mass, plain, product ); }, "mapping" ),
                   "mass multiply rejects null, wrong-sized, and plain input vectors" );
            check( throws_with( [&] { KARMA::mass_multiply( system, mass, one, nullptr ); }, "null" ) &&
                       throws_with( [&] { KARMA::mass_multiply( system, mass, one, wrong_size ); }, "size" ) &&
                       throws_with( [&] { KARMA::mass_multiply( system, mass, one, plain ); }, "mapping" ),
                   "mass multiply rejects null, wrong-sized, and plain output vectors" );
            check( throws_with( [&] { KARMA::mass_solve( system, mass, nullptr, recovered ); }, "null" ) &&
                       throws_with( [&] { KARMA::mass_solve( system, mass, wrong_size, recovered ); }, "size" ) &&
                       throws_with( [&] { KARMA::mass_solve( system, mass, plain, recovered ); }, "mapping" ),
                   "mass solve rejects null, wrong-sized, and plain right-hand sides" );
            check( throws_with( [&] { KARMA::mass_solve( system, mass, one, nullptr ); }, "null" ) &&
                       throws_with( [&] { KARMA::mass_solve( system, mass, one, wrong_size ); }, "size" ) &&
                       throws_with( [&] { KARMA::mass_solve( system, mass, one, plain ); }, "mapping" ),
                   "mass solve rejects null, wrong-sized, and plain output vectors" );
            PetscReal retained_product = 0, retained_solution = 0, retained_mass = 0;
            VecNorm( product, NORM_INFINITY, &retained_product );
            VecNorm( recovered, NORM_INFINITY, &retained_solution );
            MatNorm( mass, NORM_INFINITY, &retained_mass );
            check( std::abs( retained_product - 37 ) < 1.e-12 && std::abs( retained_solution - 43 ) < 1.e-12 &&
                       retained_mass > 40,
                   "invalid mass arguments leave matrix and vector outputs unchanged" );
            MatShift( mass, -41 );
            MatDestroy( &wrong_size_matrix );
            MatDestroy( &plain_matrix );
            VecDestroy( &wrong_size );
            VecDestroy( &plain );
        }
        KARMA::mass_multiply( system, mass, one, product );
        PetscScalar sum = 0;
        VecSum( product, &sum );
        check( std::abs( PetscRealPart( sum ) - 2.0 ) < 1.e-12, "mass row sums integrate each field to total measure" );
        KARMA::mass_solve( system, mass, product, recovered );
        VecAXPY( recovered, -1, one );
        VecNorm( recovered, NORM_2, &norm );
        check( norm < 1.e-10, "mass solve inverts mass multiply" );
        KARMA::mass_solve( system, mass, product, recovered, true );
        VecAXPY( recovered, -1, one );
        VecNorm( recovered, NORM_2, &norm );
        check( norm < 1.e-12, "positive P1 row sums give the exact constant-field lumped solve" );
        for( const PetscScalar matrix_scale : { PetscScalar{ 1.e-20 }, PetscScalar{ 1.e20 } } )
        {
            MatScale( mass, matrix_scale );
            VecScale( product, matrix_scale );
            KARMA::mass_solve( system, mass, product, recovered, true );
            VecAXPY( recovered, -1, one );
            VecNorm( recovered, NORM_2, &norm );
            check( norm < 1.e-12, "positive lumped mass validation is invariant under uniform scaling" );
            MatScale( mass, 1 / matrix_scale );
            VecScale( product, 1 / matrix_scale );
        }

        KARMA::PetscOwner< Vec, VecDestroy > owner;
        VecDuplicate( one, owner.put() );
        check( throws_with( [&] { (void)owner.put(); }, "empty" ), "PetscOwner put rejects replacing ownership" );
        VecDuplicate( one, owner.reset_and_put() );
        KARMA::PetscOwner< Vec, VecDestroy > moved( std::move( owner ) );
        check( owner.get() == nullptr && moved.get() != nullptr, "PetscOwner supports reset reuse and move ownership" );

        VecDestroy( &recovered );
        VecDestroy( &product );
        VecDestroy( &one );
        MatDestroy( &mass );
        }
    }
    int global_failures = 0;
    MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    PetscFinalize();
    return global_failures == 0 ? 0 : 1;
}
