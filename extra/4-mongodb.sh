#!/bin/bash

INSTALLED_PACKAGES=/root/installed-packages
INSTALLED_PACKAGES_DOWNLOAD_FOLDER=$INSTALLED_PACKAGES/downloads
MONGODB_VERSION=2.2.7
MONGODB_INSTALLATION_FOLDER=$INSTALLED_PACKAGES/mongodb-$MONGODB_VERSION
MONGODB_PKG=mongodb-linux-x86_64-$MONGODB_VERSION
MONGODB_FILENAME=$MONGODB_PKG.tgz

mkdir -p $INSTALLED_PACKAGES_DOWNLOAD_FOLDER
cd $INSTALLED_PACKAGES_DOWNLOAD_FOLDER

# -- get tar file
wget http://fastdl.mongodb.org/linux/$MONGODB_FILENAME

# -- unpack it
tar -zxvf $MONGODB_FILENAME

# -- copy extracted files to the appropriate directory
cp -R -n $MONGODB_PKG/ $MONGODB_INSTALLATION_FOLDER

# -- set path
echo export PATH='$'PATH:$MONGODB_INSTALLATION_FOLDER/bin >> /etc/profile

echo "done!"
