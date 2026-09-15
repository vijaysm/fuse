#!/bin/sh

# variable needed in the libmesh install. this should be the location of trunk
export KARMA_DIR=`pwd`
echo $KARMA_DIR

# since I was running this script piece-by-piece, 
# I have set these variable to avoid re-make the install when
# some components were already installed
# ideally, leave the do_xxxxx values to 1
export do_petsc=1
export do_slepc=1
export do_petscext=0
export do_libmesh=1

# select whether to compil dbg or opt 
export DBG_OPT=dbg

if [ $DBG_OPT == opt ] ; then
    export DBG_TXT=no
else
    export DBG_TXT=yes
fi
echo $DBG_TXT

echo " "
echo " ================================"
echo "! Installing KARMA in ${DBG_OPT} mode !"
echo " ================================"
echo " "
## svn co svn+ssh://ragusalinux/jcrgroup/KARMA/trunk . 
## svn co file:///jcrgroup/KARMA/trunk . 
cd contrib

# petsc 
if [ ${do_petsc} -eq 0 ]; then
    echo "SKIPPING installation of PETSc."
    sleep 2
else
    ## configure and install petsc
    sh ../config/install-petsc.sh
    ## check for errors
    if [ ${PETSC_INSTALLED} -eq 0 ]; then
        echo "PETSc installation FAILED. Stopping install-contrib script"
        exit
    fi
fi

# then slepc
if [ ${do_slepc} -eq 0 ]; then
    echo "SKIPPING installation of SLEPc."
    sleep 2
else
    ## configure and install slepc
    sh ../config/install-slepc.sh
    ## check for errors
    if [ ${SLEPC_INSTALLED} -eq 0 ]; then
        echo "SLEPc installation FAILED. Stopping install-contrib script"
        exit
    fi
fi

# then petsc-ext, at some point later
if [ ${do_petscext} -eq 0 ]; then
    echo "SKIPPING installation of PETSc-Ext."
    sleep 2
else
    ## configure and install petsc-ext
    sh ../config/install-petscext.sh
    ## check for errors
    if [ ${PETSCEXT_INSTALLED} -eq 0 ]; then
        echo "PETSc-Ext installation FAILED. Stopping install-contrib script"
        exit
    fi
fi

# 

# then libmesh
if [ ${do_libmesh} -eq 0 ]; then
    echo "SKIPPING installation of libMesh."
    sleep 2
else
    ## configure and install libMesh
    sh ../config/install-libmesh.sh
	## check for errors
    if [ ${LIBMESH_INSTALLED} -eq 0 ]; then
        echo "libMesh installation FAILED. Stopping install-contrib script"
        exit
    fi
fi


# then do some utilities
make


