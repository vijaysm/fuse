#!/bin/sh

echo ""
echo " ========================"
echo "  --- PETSc --- ${DBG_OPT} mode   "
echo " ========================"
echo ""

# location of openmpi on the machine
export MPI_DIR="/usr/local/openmpi"
export PATH=$PATH:$MPI_DIR/bin/

# name of the PETSc version
PETSC_FILE="petsc-lite-3.1-p5"
PETSC_DOWNLOAD="http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/$PETSC_FILE.tar.gz"
echo "PETSc version from : " $PETSC_DOWNLOAD
echo ""

# configure line: note that DBG_OPT and MPI_DIR are variables used in that line

# JANUS.ANL
#PETSC_CONFIGURE="--with-blas-lapack-dir=/opt/intel/mkl/10.0.2.018 --with-mpi-dir=/software/mpich2-1.2/O-intel-10 --with-mpi-shared=0 --known-mpi-shared=no --FOPTFLAGS=-g --with-shared=0 --COPTFLAGS=-g --download-parmetis=1 --download-superlu_dist=1 --download-hypre=1 --download-blacs=1 --download-scalapack=1 --with-clanguage=C++ --download-plapack=1 --download-mumps=1 --download-umfpack=yes --with-debugging=1 --with-errorchecking=yes --with-batch=1 --host"
# GROVE.NE.TAMU
PETSC_CONFIGURE="--with-blas-lapack-dir=/usr/local/intel/11.0/074/fce/mkl/lib/em64t/ --with-mpi-dir=${MPI_DIR} --with-shared=1 --with-X11=1 --known-mpi-shared=yes --download-parmetis=1 --download-superlu_dist=1 --download-hypre=1 --download-blacs=1 --download-scalapack=1 --with-mpi-shared=1 --with-clanguage=C++ --download-plapack=1 --download-mumps=1 --download-umfpack=yes --with-batch=1 --host --with-errorchecking=yes --with-debugging=${DBG_TXT}"
##PETSC_CONFIGURE="--with-blas-lapack-dir=/usr/local/intel/11.0/074/fce/mkl/lib/em64t/ --with-mpi-dir=${MPI_DIR} --with-shared=1 --with-X11=1 --known-mpi-shared=yes --download-parmetis=1 --download-boost=1 --download-superlu_dist=1 --download-hypre=1 --download-blacs=1 --download-scalapack=1 --with-batch=1 --with-mpi-shared=1 --with-clanguage=C++ --download-plapack=1 --download-mumps=1 --download-sundials=yes --download-blacs=yes --download-spai=yes --download-umfpack=yes --host --with-debugging=yes"
## PETSC_CONFIGURE=" --download-mpich=1 --download-f-blas-lapack=1 --with-shared=1 --with-X11=1 --download-parmetis=1 --download-boost=1 --download-superlu_dist=1 --download-hypre=1 --download-blacs=1 --download-scalapack=1 --with-mpi-shared=1 --with-clanguage=C++ --download-plapack=1 --download-mumps=1 --download-sundials=yes --with-debugging=no"

export PETSC_INSTALLED=0
if [ ! -f 'petsc_download.log' ]; then 
    echo "! Downloading PETSc !"
    wget $PETSC_DOWNLOAD -o petsc_download.log

    PETSC_DOWNLOADED=`grep 'Saving to:' petsc_download.log`
    ## check for errors in the download
    if [ -z "$PETSC_DOWNLOADED" ]; then
        echo "! Aborting PETSc configuration. Download incomplete."
        exit
    fi

    ## now unzip and install petsc 
    gunzip -df "$PETSC_FILE.tar.gz"
    tar -xvf "$PETSC_FILE.tar" 
    rm "$PETSC_FILE.tar"
    PETSC_DIRECTORY=petsc${PETSC_FILE##p*e} 
    #echo $PETSC_DIRECTORY
    rm -rf petsc
    mv "$PETSC_DIRECTORY" petsc 
else
    echo "! PETSc was already downloaded !"
fi 
echo ""

if [ ! -d 'petsc' ]; then 
    echo "! PETSc directory does not exist. Quitting installation !"
    exit
fi
cd petsc

## now configure PETSc
export PETSC_DIR=`pwd`
export PETSC_ARCH=linux-mpi-${DBG_OPT}
export PATH=$PATH:$PETSC_DIR/$PETSC_ARCH/bin
echo "PETSc Arch Variable is ${PETSC_ARCH}"
sleep 3

echo "! Configuring PETSc !"
chmod u+x configure
./configure $PETSC_CONFIGURE
mpiexec -n 1 ./conftest
./reconfigure.py

echo "! Compiling and Linking PETSc !"
## now make PETSc
make all test 

cd ..
export PETSC_INSTALLED=1
