#!/bin/sh

echo ""
echo " ====================="
echo "  --- LIBMESH ---     "
echo " ====================="
echo ""


LIBMESH_CONFIGURE="--enable-everything=yes --disable-shared --with-mpi=$MPI_DIR --enable-petsc --enable-vtk --with-vtk-lib=/software/visit1.12.2/VTK --enable-slepc --disable-complex --disable-parmesh --disable-ghosted --disable-ifem"
#LIBMESH_CONFIGURE="--enable-everything=yes --enable-shared=yes --with-mpi=$MPI_DIR --enable-petsc --disable-vtk --enable-slepc --disable-complex --disable-parmesh --enable-ghosted --disable-ifem"
# LIBMESH_CONFIGURE="--enable-everything=yes --enable-shared=yes --with-mpi=$PETSC_DIR/$PETSC_ARCH --enable-petsc --disable-vtk --enable-slepc --enable-complex=no --enable-ghosted --enable-exceptions"

newinstallation=0
export LIBMESH_INSTALLED=0
## First installation of libmesh: So checkout from the developer version
## In the future, we shall make this checkout a specific revision of libMesh that
## has been well tested with KARMA.
if [ ! -d "libmesh" ]; then 
    echo "! Downloading LibMesh !"
    svn checkout https://libmesh.svn.sourceforge.net/svnroot/libmesh/trunk/libmesh libmesh
    newinstallation=1
fi

cd libmesh
export LIBMESH_DIR=`pwd`

echo "! Updating libMesh to the latest revision !"
sleep 2
svn update

if [ $newinstallation -eq 0  ]; then 
    echo "! Reversing KARMA patch to LibMesh !"
    sleep 2
    patch -p0 -R < $KARMA_DIR/contrib/libmesh_dev.patch
fi

echo "! Applying KARMA patch to LibMesh !"
sleep 2
patch -p0 < $KARMA_DIR/contrib/libmesh_dev.patch

## now configure
chmod u+x configure
echo "! Configuring LibMesh !"
./configure $LIBMESH_CONFIGURE

# now compile
echo "! Compiling LibMesh in Opt mode !"
export METHOD=opt
make all -j8
# again
echo "! Compiling LibMesh in Debug mode !"
export METHOD=dbg
make all -j8
cd ..

export LIBMESH_INSTALLED=1
