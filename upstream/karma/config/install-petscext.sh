# PETSCEXT_DOWNLOAD="http://jupiter.ethz.ch/~dmay/Research/PetscExt/PetscExt-dev.tar.gz"
PETSCEXT_DOWNLOAD="http://jupiter.ethz.ch/~dmay/Research/PetscExt/PetscExt-3.0.0.tar.gz"
PETSCEXT_CONFIGURE=""

echo "! Downloading PetscExt !"
if [ ! `ls petscext_download.log` ]; then 
	wget $PETSCEXT_DOWNLOAD -o petscext_download.log
	gunzip -df PetscExt-3.0.0.tar.gz
	tar -xf PetscExt-3.0.0.tar
	rm PetscExt-3.0.0.tar
	mv PetscExt-3.0.0 petscext
fi
## now unzip and install petsc-ext
echo "! Configuring and installing PetscExt !"
cd petscext
export PETSCEXT_DIR=`pwd`
chmod u+x config/configure.py
config/configure.py $PETSCEXT_CONFIGURE
# Change Line 237: petscext/src/pc/impls/bschur/pc-bschur.c: s->application_type  = PC_BSCHUR_UPPER;
make all check
cd ..
