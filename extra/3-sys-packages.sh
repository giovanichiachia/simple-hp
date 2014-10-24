#!/bin/bash

# -- message passing library
apt-get install libzmq-dev

# -- jpeg decoder
apt-get install libjpeg-dev

apt-get install python-dev python-pip

pip install cython
pip install virtualenv
pip install virtualenvwrapper

#Numpy + ATLAS
export ATLAS=/usr/local/atlas/lib/libtatlas.so

pip install numpy
pip install scipy
