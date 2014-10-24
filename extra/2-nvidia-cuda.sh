#!/bin/bash

INSTALLED_PACKAGES=/root/installed-packages
INSTALLED_PACKAGES_DOWNLOAD_FOLDER=$INSTALLED_PACKAGES/downloads
CUDA_FILENAME=cuda_6.5.14_linux_64.run

sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libgl1-mesa-dri libglapi-mesa libglu1-mesa libglu1-mesa-dev

cd $INSTALLED_PACKAGES_DOWNLOAD_FOLDER
wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/$CUDA_FILENAME

chmod +rwx $CUDA_FILENAME

#Ubuntu 12.04

service lightdm stop

apt-get --purge remove nvidia-current

./$CUDA_FILENAME

echo "/usr/local/cuda-6.5/lib" > /etc/ld.so.conf.d/nvidia-cuda.conf
echo "/usr/local/cuda-6.5/lib64" >> /etc/ld.so.conf.d/nvidia-cuda.conf
echo export PATH='$'PATH:/usr/local/cuda-6.5/bin >> /etc/profile
