# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD


# -- import here additional dataset classes
from simplehp.data.rndsplits    import PubFig83
from simplehp.data.rndsplits    import CalTech256

# -- import here algorithms for hyperparameter optimization
from hyperopt.rand import suggest as random
from hyperopt.tpe import suggest as tpe

# -- import here learning algorithms for hyperparameter evaluation
from svm import svm_ova_from_splits as svm_ova
from svm import svm_one_vs_world as svm_one_vs_world

# -- add here additional dataset classes
datasets = {'01': PubFig83,
            '02': CalTech256,
           }

hp_algos = {'default': 'random',
            'random': random,
            'tpe': tpe,
            }

learning_algos = {'default': 'svm_ova',
                  'svm_ova': svm_ova,
                  'svm_one_vs_world': svm_one_vs_world,
                  }
