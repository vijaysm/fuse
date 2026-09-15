#!/bin/sh

echo ""
echo " ========================"
echo "  --- SLEPc --- ${DBG_OPT} mode   "
echo " ========================"
echo ""

# name of the SLEPc version
SLEPC_FILE="slepc-3.1-p4"
SLEPC_DOWNLOAD="http://www.grycap.upv.es/slepc/download/distrib/$SLEPC_FILE.tgz"
echo "SLEPc version from : " $SLEPC_DOWNLOAD
echo ""

SLEPC_CONFIGURE=""
# SLEPC_CONFIGURE="--with-primme --with-primme-dir=/emchome/vijaysm/soft/PRIMME --with-primme-flags=-lprimme --with-arpack --with-arpack-dir=/emchome/vijaysm/soft/ARPACK --with-arpack-flags=-lparpack_MPI-LINUX,-larpack_LINUX"

export SLEPC_INSTALLED=0
if [ ! -f 'slepc_download.log' ]; then 
    echo "! Downloading SLEPc !"
    wget $SLEPC_DOWNLOAD -o slepc_download.log

    SLEPC_DOWNLOADED=`grep 'Saving to:' slepc_download.log`
    ## check for errors in the download
    if [ -z "$SLEPC_DOWNLOADED" ]; then
        echo "! Aborting SLEPc configuration. Download incomplete."
        exit
    fi

    ## now unzip and install slepc
    gunzip -df "$SLEPC_FILE.tgz"
    tar -xvf "$SLEPC_FILE.tar"
    rm "$SLEPC_FILE.tar"
    mv "$SLEPC_FILE" slepc
else
    echo "! SLEPc was already downloaded !"
fi
echo ""

if [ ! -d 'slepc' ]; then 
    echo "! SLEPc directory does not exist. Quitting installation !"
    exit
fi
cd slepc

## now configure slepc
echo "! Configuring SLEPc !"
export SLEPC_DIR=`pwd`
chmod u+x config/configure.py
config/configure.py $SLEPC_CONFIGURE

echo "! Compiling and Linking SLEPc !"
## make slepc
make all test

cd ..
export SLEPC_INSTALLED=1
