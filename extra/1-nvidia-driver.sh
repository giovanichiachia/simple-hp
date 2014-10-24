#!/bin/bash

INSTALLED_PACKAGES=/root/installed-packages
INSTALLED_PACKAGES_DOWNLOAD_FOLDER=$INSTALLED_PACKAGES/downloads
NVIDIA_VERSION=343.22
NVIDIA_FILENAME=NVIDIA-Linux-x86_64-$NVIDIA_VERSION.run

cd $INSTALLED_PACKAGES_DOWNLOAD_FOLDER
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/$NVIDIA_VERSION/$NVIDIA_FILENAME

chmod +rwx $NVIDIA_FILENAME

#Ubuntu 12.04

service lightdm stop

apt-get --purge remove nvidia-current

./$NVIDIA_FILENAME




