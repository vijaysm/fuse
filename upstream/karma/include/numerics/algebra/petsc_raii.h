#ifndef __petsc_raii_h__
#define __petsc_raii_h__

#include "karma_exceptions.h"

#include <petscvec.h>

#include <string>
#include <utility>

namespace KARMA
{
namespace detail
{
    inline void petsc_raii_check( PetscErrorCode error, const char* operation )
    {
        if( error ) KARMA_THROW( InvalidArgument( std::string( "PETSc error in " ) + operation ) );
    }
}  // namespace detail

template < class Handle, auto Destroy >
class PetscOwner
{
  public:
    PetscOwner() = default;
    ~PetscOwner() noexcept
    { reset(); }
    PetscOwner( const PetscOwner& )            = delete;
    PetscOwner& operator=( const PetscOwner& ) = delete;
    PetscOwner( PetscOwner&& other ) noexcept : _value( other.release() ) {}
    PetscOwner& operator=( PetscOwner&& other ) noexcept
    {
        if( this != &other )
        {
            reset();
            _value = other.release();
        }
        return *this;
    }

    [[nodiscard]] Handle get() const noexcept
    { return _value; }
    [[nodiscard]] Handle* put()
    {
        if( _value ) KARMA_THROW( InvalidArgument( "PetscOwner::put requires empty ownership" ) );
        return &_value;
    }
    [[nodiscard]] Handle* reset_and_put() noexcept
    {
        reset();
        return &_value;
    }
    void reset() noexcept
    {
        if( _value ) (void)Destroy( &_value );
    }
    [[nodiscard]] Handle release() noexcept
    { return std::exchange( _value, nullptr ); }

  private:
    Handle _value = nullptr;
};

class PetscVecLocalForm
{
  public:
    explicit PetscVecLocalForm( Vec vector ) : _vector( vector )
    { detail::petsc_raii_check( VecGhostGetLocalForm( _vector, &_local ), "VecGhostGetLocalForm" ); }
    ~PetscVecLocalForm() noexcept
    {
        if( _local ) (void)VecGhostRestoreLocalForm( _vector, &_local );
    }
    PetscVecLocalForm( const PetscVecLocalForm& )            = delete;
    PetscVecLocalForm& operator=( const PetscVecLocalForm& ) = delete;
    [[nodiscard]] Vec get() const noexcept
    { return _local; }

  private:
    Vec _vector = nullptr;
    Vec _local  = nullptr;
};

class PetscVecArrayRead
{
  public:
    explicit PetscVecArrayRead( Vec vector ) : _vector( vector )
    {
        detail::petsc_raii_check( VecGetArrayRead( _vector, &_values ), "VecGetArrayRead" );
        _acquired = true;
    }
    ~PetscVecArrayRead() noexcept
    {
        if( _acquired ) (void)VecRestoreArrayRead( _vector, &_values );
    }
    PetscVecArrayRead( const PetscVecArrayRead& )            = delete;
    PetscVecArrayRead& operator=( const PetscVecArrayRead& ) = delete;
    [[nodiscard]] const PetscScalar* get() const noexcept
    { return _values; }
    [[nodiscard]] PetscErrorCode release() noexcept
    {
        if( !_acquired ) return PETSC_SUCCESS;
        const PetscScalar* values = _values;
        const PetscErrorCode error = VecRestoreArrayRead( _vector, &values );
        if( error == PETSC_SUCCESS )
        {
            _values = nullptr;
            _acquired = false;
        }
        return error;
    }

  private:
    Vec _vector                = nullptr;
    const PetscScalar* _values = nullptr;
    bool _acquired             = false;
};

class PetscVecArrayWrite
{
  public:
    explicit PetscVecArrayWrite( Vec vector ) : _vector( vector )
    {
        detail::petsc_raii_check( VecGetArray( _vector, &_values ), "VecGetArray" );
        _acquired = true;
    }
    ~PetscVecArrayWrite() noexcept
    {
        if( _acquired ) (void)VecRestoreArray( _vector, &_values );
    }
    PetscVecArrayWrite( const PetscVecArrayWrite& )            = delete;
    PetscVecArrayWrite& operator=( const PetscVecArrayWrite& ) = delete;
    [[nodiscard]] PetscScalar* get() const noexcept
    { return _values; }
    [[nodiscard]] PetscErrorCode release() noexcept
    {
        if( !_acquired ) return PETSC_SUCCESS;
        PetscScalar* values = _values;
        const PetscErrorCode error = VecRestoreArray( _vector, &values );
        if( error == PETSC_SUCCESS )
        {
            _values = nullptr;
            _acquired = false;
        }
        return error;
    }

  private:
    Vec _vector          = nullptr;
    PetscScalar* _values = nullptr;
    bool _acquired       = false;
};
}  // namespace KARMA

#endif
