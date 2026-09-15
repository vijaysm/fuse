#ifndef __abstract_system_owner_h__
#define __abstract_system_owner_h__

#include "karma_system_base.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace KARMA
{
class AbstractSystemOwner
{
  public:
    explicit AbstractSystemOwner( std::unique_ptr< AbstractSystem > system )
    { set_abstract_system( std::move( system ) ); }
    virtual ~AbstractSystemOwner() = default;

    AbstractSystemOwner( const AbstractSystemOwner& )            = delete;
    AbstractSystemOwner& operator=( const AbstractSystemOwner& ) = delete;

    [[nodiscard]] bool has_abstract_system() const noexcept { return _system != nullptr; }

    [[nodiscard]] AbstractSystem& abstract_system()
    {
        if( !_system ) throw std::logic_error( "abstract system has not been adopted" );
        return *_system;
    }

    [[nodiscard]] const AbstractSystem& abstract_system() const
    {
        if( !_system ) throw std::logic_error( "abstract system has not been adopted" );
        return *_system;
    }

  protected:
    AbstractSystemOwner() = default;

    void set_abstract_system( std::unique_ptr< AbstractSystem > system )
    {
        if( !system ) throw std::invalid_argument( "abstract system cannot be null" );
        if( _system ) throw std::logic_error( "abstract system has already been adopted" );
        _system = std::move( system );
    }

  private:
    std::unique_ptr< AbstractSystem > _system;
};
}  // namespace KARMA

#endif
