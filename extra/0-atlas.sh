#!/bin/bash

INSTALLED_PACKAGES=/root/installed-packages
INSTALLED_PACKAGES_DOWNLOAD_FOLDER=$INSTALLED_PACKAGES/downloads
ATLAS_VERSION=3.10.1
ATLAS_INSTALLATION_FOLDER=$INSTALLED_PACKAGES/ATLAS-$ATLAS_VERSION
ATLAS_FILENAME=atlas$ATLAS_VERSION.tar.bz2
LAPACK_VERSION=3.5.0
LAPACK_FILENAME=lapack-$LAPACK_VERSION.tgz
PREFIX=/usr/local/atlas

mkdir -p $INSTALLED_PACKAGES_DOWNLOAD_FOLDER
cd $INSTALLED_PACKAGES_DOWNLOAD_FOLDER

wget http://ufpr.dl.sourceforge.net/project/math-atlas/Stable/$ATLAS_VERSION/$ATLAS_FILENAME
wget http://www.netlib.org/lapack/$LAPACK_FILENAME

tar -xvjf $ATLAS_FILENAME

# -- pre-requisites
apt-get install gfortran77 build-essential

# -- set CPUs to performance mode
apt-get install cpufrequtils

# In Ubuntu 12.04 use cpufreq-set
/usr/bin/cpufreq-set -g performance 

for i in {1..7};
do sudo cp /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor; done;

for i in {0..7}; do cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor; done;

# -- create Makefile properly
mkdir -p $ATLAS_INSTALLATION_FOLDER
cp -R $INSTALLED_PACKAGES_DOWNLOAD_FOLDER/ATLAS $ATLAS_INSTALLATION_FOLDER
mkdir $ATLAS_INSTALLATION_FOLDER/ATLAS/build
cd $ATLAS_INSTALLATION_FOLDER/ATLAS/build/
../configure -b 64 -D c -DWALL --shared --prefix=$PREFIX --with-netlib-lapack-tarfile=$INSTALLED_PACKAGES_DOWNLOAD_FOLDER/$LAPACK_FILENAME

# -- additional stuff
make build
make check
make ptcheck
make time
make install
