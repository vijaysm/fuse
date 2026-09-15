#include "abstract_system_owner.h"
#include "karma_system.h"

#include <petsc.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{
int failures = 0;

void check( bool ok, const std::string& what )
{
    if( !ok )
    {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

template < typename Exception, typename F >
bool throws( F&& fn )
{
    try
    {
        fn();
    }
    catch( const Exception& )
    {
        return true;
    }
    catch( ... )
    {}
    return false;
}

class FakeSystem final : public KARMA::AbstractSystem
{
  public:
    explicit FakeSystem( bool& destroyed ) : _destroyed( destroyed ) {}
    ~FakeSystem() override { _destroyed = true; }

    [[nodiscard]] const std::string& name() const noexcept override { return _name; }
    unsigned add_variable( KARMA::VariableSpec ) override { return 0; }
    [[nodiscard]] unsigned n_variables() const noexcept override { return 0; }
    [[nodiscard]] const KARMA::VariableSpec& variable( unsigned ) const override
    { throw std::logic_error( "unused" ); }
    [[nodiscard]] unsigned variable_number( std::string_view ) const override { return 0; }
    [[nodiscard]] bool has_variable( std::string_view ) const override { return false; }
    void init() override {}
    [[nodiscard]] bool is_initialized() const noexcept override { return false; }
    [[nodiscard]] PetscInt n_dofs() const noexcept override { return 0; }
    [[nodiscard]] PetscInt n_local_dofs() const noexcept override { return 0; }
    [[nodiscard]] Vec create_vector() const override { return nullptr; }
    [[nodiscard]] Mat create_matrix() const override { return nullptr; }
    [[nodiscard]] Vec solution() const noexcept override { return nullptr; }
    [[nodiscard]] Vec current_solution() const noexcept override { return nullptr; }
    [[nodiscard]] Vec old_solution() const noexcept override { return nullptr; }
    [[nodiscard]] Vec local_solution() const noexcept override { return nullptr; }
    [[nodiscard]] Vec temporary_vector() const noexcept override { return nullptr; }

  private:
    bool& _destroyed;
    std::string _name{ "fake" };
};

class TransitionalOwner final : public KARMA::AbstractSystemOwner
{
  public:
    TransitionalOwner() = default;
    using AbstractSystemOwner::set_abstract_system;
};

}  // namespace

int main( int argc, char** argv )
{
    if( const PetscErrorCode error = PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return error;

    check( throws< std::invalid_argument >( [] {
               KARMA::AbstractSystemOwner owner( std::unique_ptr< KARMA::AbstractSystem >{} );
           } ),
           "construction rejects a null system" );

    bool destroyed = false;
    {
        KARMA::AbstractSystemOwner owner( std::make_unique< FakeSystem >( destroyed ) );
        check( owner.has_abstract_system(), "constructed owner has a system" );
        check( owner.abstract_system().name() == "fake", "non-const accessor returns the owned system" );
        const auto& const_owner = owner;
        check( const_owner.abstract_system().name() == "fake", "const accessor returns the owned system" );
    }
    check( destroyed, "owner destroys its abstract system" );

    TransitionalOwner transitional;
    check( !transitional.has_abstract_system(), "legacy owner starts empty" );
    check( throws< std::logic_error >( [&] { (void)transitional.abstract_system(); } ),
           "empty owner rejects access" );
    check( throws< std::invalid_argument >( [&] {
               transitional.set_abstract_system( std::unique_ptr< KARMA::AbstractSystem >{} );
           } ),
           "adoption rejects a null system" );
    transitional.set_abstract_system( std::make_unique< FakeSystem >( destroyed ) );
    check( throws< std::logic_error >( [&] {
               transitional.set_abstract_system( std::make_unique< FakeSystem >( destroyed ) );
           } ),
           "a system can only be adopted once" );

    {
        KARMA::KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        mesh.generate_box( 1, 0, 0 );
        KARMA::AbstractSystemOwner owner( std::make_unique< KARMA::KARMASystem >( mesh, "real" ) );
        check( owner.abstract_system().name() == "real", "owner accepts a real KARMASystem" );
    }

    const PetscErrorCode finalize_error = PetscFinalize();
    return finalize_error ? finalize_error : failures == 0 ? 0 : 1;
}
