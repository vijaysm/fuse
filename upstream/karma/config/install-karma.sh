#!/bin/sh

# make sure the environmental variables point to right locations for
# PETSC_DIR, SLEPC_DIR, LIBMESH_DIR and KARMA_DIR, and whether you want
# optimized or debug version, set PETSC_ARCH accordingly.
# I have grouped everything in the file envKARMA which is sourced now
# but one can also put it in their .bashrc or .cshrc 
 
source ~/karma/envKARMA
echo "KARMA_DIR is   $KARMA_DIR"
echo "PETSC_DIR is   $PETSC_DIR"
echo "SLEPC_DIR is   $SLEPC_DIR"
echo "LIBMESH_DIR is $LIBMESH_DIR"
echo "PETSC_ARCH is  $PETSC_ARCH"
sleep 5

# configure
chmod u+x ./config/configure
./config/configure --enable-petsc --enable-slepc --enable-libMesh --enable-csvparser --enable-tinyxpath --disable-cpptest --enable-shared

# make dependencies
make dependencies

# whether you want Karma to be compiled in debug or opt mode
# the METHOD variable should have been already set, 
# otherwise put it in the envKARMA file as well
echo "METHOD is $METHOD"
sleep 3

# make Karma
make

cd test/ode
make all

exit

