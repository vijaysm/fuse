#ifndef __karma_physics_bc_manager_h__
#define __karma_physics_bc_manager_h__

#include "karma_boundary_condition.h"
#include "karma_dense.h"
#include "karma_mesh.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class KARMAParameters;
#ifdef KARMA_HAVE_LIBMESH
#include "elem.h"
#endif
namespace KARMA
{
class KARMASystem;
class KARMAFEBase;
namespace detail
{
struct SerializedDirichlet
{
    PetscInt dof;
    bool time_dependent;
    std::vector< Real > values;
};

[[nodiscard]] std::vector< SerializedDirichlet > decode_dirichlet_conditions( std::string_view bytes );
}  // namespace detail
}  // namespace KARMA

using BoundaryNodes             = std::map< std::string, KARMABoundaryCondition* >;
using BoundaryPair              = std::pair< std::string, KARMABoundaryCondition* >;
using BoundaryNodeIterator      = BoundaryNodes::iterator;
using ConstBoundaryNodeIterator = BoundaryNodes::const_iterator;
using boundary_function         = KARMABoundaryCondition* (*)( const KARMAParameters& );
using modern_boundary_function  = std::unique_ptr< KARMABoundaryCondition > ( * )( const KARMAParameters& );

class KARMAPhysicsBCManager
{
  public:
    explicit KARMAPhysicsBCManager( unsigned variable, bool continuous = true,
                                    std::optional< unsigned > component = std::nullopt );
    KARMAPhysicsBCManager( const KARMAPhysicsBCManager& other );
    KARMAPhysicsBCManager& operator=( const KARMAPhysicsBCManager& other );
    ~KARMAPhysicsBCManager();

    void attach_function( modern_boundary_function function, KARMA::PHYSICS::BCType type );
#ifdef KARMA_HAVE_LIBMESH
    void attach_function( boundary_function function, KARMA::PHYSICS::BCType type );
    void update_boundary_conditions( const Elem* element, KARMAParameters& parameters, bool check_all_sides = true );
#endif
    void refresh( const KARMA::KARMAMesh& mesh, const KARMA::KARMASystem& system, KARMAParameters& parameters );
    void update_boundary_conditions( const KARMA::KARMAMesh& mesh, moab::EntityHandle element,
                                     const KARMA::KARMASystem& system, KARMAParameters& parameters,
                                     bool check_all_sides = false );
    void clear() noexcept;

    bool is_dirichlet_condition( std::string& key ) const;
    bool is_neumann_condition( std::string& key ) const;
    bool is_robin_condition( std::string& key ) const;
    bool add_dirichlet_condition( unsigned node, KARMABoundaryCondition* condition );
    bool add_dirichlet_condition( unsigned component, unsigned side, KARMABoundaryCondition* condition );
    bool add_neumann_condition( unsigned element, unsigned side, KARMABoundaryCondition* condition );
    bool add_robin_condition( unsigned element, unsigned side, KARMABoundaryCondition* condition );
    bool delete_dirichlet_condition( unsigned node );
    bool delete_dirichlet_condition( unsigned component, unsigned side );
    bool delete_neumann_condition( unsigned element, unsigned side );
    bool delete_robin_condition( unsigned element, unsigned side );
    [[nodiscard]] const BoundaryNodes& get_dirichlet_condition() const noexcept;
    [[nodiscard]] const BoundaryNodes& get_neumann_condition() const noexcept;
    [[nodiscard]] const BoundaryNodes& get_robin_condition() const noexcept;
    void print_conditions() const;

    [[nodiscard]] unsigned n_dirichlet_nodes() const noexcept;
    [[nodiscard]] unsigned n_neumann_nodes() const noexcept;
    [[nodiscard]] unsigned n_robin_nodes() const noexcept;
    [[nodiscard]] unsigned variable() const noexcept
    { return _variable; }
    [[nodiscard]] const std::vector< PetscInt >& dirichlet_dofs() const noexcept
    { return _dirichlet_dofs; }
    /** Deterministic byte signature of all boundary data consumed by assembly. */
    [[nodiscard]] std::string assembly_signature( const KARMA::KARMASystem& system ) const;

    [[nodiscard]] KARMABoundaryCondition* get_dirichlet_condition( PetscInt dof ) const;
    [[nodiscard]] KARMABoundaryCondition* get_dirichlet_condition( unsigned element, unsigned side ) const;
    [[nodiscard]] KARMABoundaryCondition* get_neumann_condition( unsigned element, unsigned side ) const;
    [[nodiscard]] KARMABoundaryCondition* get_robin_condition( unsigned element, unsigned side ) const;
    [[nodiscard]] KARMABoundaryCondition* get_dirichlet_condition( KARMA::GlobalId element, unsigned side ) const;
    [[nodiscard]] KARMABoundaryCondition* get_neumann_condition( KARMA::GlobalId element, unsigned side ) const;
    [[nodiscard]] KARMABoundaryCondition* get_robin_condition( KARMA::GlobalId element, unsigned side ) const;

    struct RobinTerms
    {
        KARMA::RealDenseMatrix matrix;
        KARMA::RealDenseVector load;
    };

    [[nodiscard]] static KARMA::RealDenseVector integrate_face_load( const KARMA::KARMAFEBase& fe, Real value );
    [[nodiscard]] static KARMA::RealDenseMatrix integrate_face_mass( const KARMA::KARMAFEBase& fe, Real value );
    /** g is prescribed outward normal flux; returns positive integral phi_i g. */
    [[nodiscard]] static KARMA::RealDenseVector integrate_neumann_load( const KARMA::KARMAFEBase& fe,
                                                                        const KARMABoundaryCondition& condition );
    /** Typed flux form a*(k grad(u).n)+b*u=c returns positive b/a matrix and c/a load. */
    [[nodiscard]] static RobinTerms integrate_robin( const KARMA::KARMAFEBase& fe,
                                                     const KARMABoundaryCondition& condition );

  private:
    using SideKey    = std::pair< KARMA::GlobalId, unsigned >;
    using Conditions = std::map< SideKey, std::unique_ptr< KARMABoundaryCondition > >;

    void swap_conditions( KARMAPhysicsBCManager& other ) noexcept;
    static std::string dirichlet_key( unsigned node );
    static std::string dirichlet_key( unsigned component, unsigned side );
    static std::string neumann_key( unsigned element, unsigned side );
    static std::string robin_key( unsigned element, unsigned side );

    unsigned _variable;
    bool _continuous;
    std::optional< unsigned > _component;
    modern_boundary_function _dirichlet_function = nullptr;
    modern_boundary_function _neumann_function   = nullptr;
    modern_boundary_function _robin_function     = nullptr;
#ifdef KARMA_HAVE_LIBMESH
    boundary_function _legacy_dirichlet_function = nullptr;
    boundary_function _legacy_neumann_function   = nullptr;
    boundary_function _legacy_robin_function     = nullptr;
#endif
    std::map< PetscInt, std::unique_ptr< KARMABoundaryCondition > > _dirichlet;
    Conditions _weak_dirichlet;
    Conditions _neumann;
    Conditions _robin;
    std::vector< PetscInt > _dirichlet_dofs;
    BoundaryNodes _legacy_dirichlet;
    BoundaryNodes _legacy_neumann;
    BoundaryNodes _legacy_robin;
};

#endif
