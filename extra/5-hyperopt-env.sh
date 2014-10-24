#!/bin/bash

CUDA_PATH=/usr/local/cuda-6.5

CURRENT_DIR=$(pwd)
cd $HOME

VENV_ROOT=$HOME/VENV
mkdir -p $VENV_ROOT

ENV_NAME=hyperopt-env
cp .bashrc .bashrc.$ENV_NAME.bkp

echo "
export WORKON_HOME=$VENV_ROOT
source /usr/local/bin/virtualenvwrapper.sh" >> .bashrc
source ./.bashrc

# -- create virtual environment
mkvirtualenv --system-site-packages $ENV_NAME

# -- set .bashrc to switch to this environment
echo "workon $ENV_NAME" >> .bashrc
source ./.bashrc

# -- installing requirements in virtual environment
pip install --upgrade ipython
pip install argparse  
pip install SQLAlchemy 
pip install sphinx
pip install nose -I

# -- setuptools 0.7 is bugged, use 0.6 instead
#pip install setuptools 
wget -O /tmp/setuptools.egg https://pypi.python.org/packages/2.7/s/setuptools/setuptools-0.6c11-py2.7.egg
sh /tmp/setuptools.egg

pip install --upgrade PIL

pip install pyzmq
pip install bson
pip install pymongo
pip install networkx
pip install six
pip install coverage

pip install scikit-learn
pip install matplotlib

# -- install theano
pip install theano

source ./.bashrc

# set .theanorc -- you change cpu by gpu if you have one
echo "[blas]
ldflags = -L/usr/local/atlas/lib/ -ltatlas -lgfortran

[cuda]
root=$CUDA_PATH

[global]
floatX=float32
device=cpu" > .theanorc

# -- run theano tests
python -c 'import theano as th; th.test()'

read -p "You shold have obtained 19 (known) failures. If that is the case, press [enter] and go ahead..."

# -- installing hyperopt
mkdir -p $HOME/dev/hp-pkgs
cd $HOME/dev/hp-pkgs

git clone https://github.com/hyperopt/hyperopt.git
(cd hyperopt && python setup.py develop)
(cd hyperopt && nosetests)

read -p "Make sure hyperopt tests are OK at this point. If so, press [enter] and go ahead..."

# -- installing pyautodiff from James' repo
git clone https://github.com/jaberg/pyautodiff.git
(cd pyautodiff && python setup.py develop)
(cd pyautodiff && nosetests)

# -- installing hyperopt-convnet
git clone https://github.com/giovanichiachia/hyperopt-convnet.git
(cd hyperopt-convnet && python setup.py develop)

cd $CURRENT_DIR
