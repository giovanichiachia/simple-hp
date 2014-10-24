# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

from joblib import Parallel, delayed
import signal

import logging

import numpy as np
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

DEFAULT_REGULARIZATION = 1e5
DEFAULT_TRACE_NORMALIZATION = True


def one_svm(ktrain, train_labels, ktest, cat, C):

    lcat = np.zeros(train_labels.size)

    lcat[train_labels != cat] = -1
    lcat[train_labels == cat] = +1

    svm = SVC(kernel='precomputed', C=C, tol=1e-5)
    svm.fit(ktrain, lcat)

    return svm.decision_function(ktest)[:, 0]


def svm_ova_from_kernel(ktrain, train_labels,
                        ktest, test_labels,
                        C=DEFAULT_REGULARIZATION,
                        bkg_categories=None):

    def sighandler_svm(signum, frame):
        logger.info('Caught signal %i while training SVMs in paralell.'
                    % signum)

    signal.signal(signal.SIGTERM, sighandler_svm)

    n_test = ktest.shape[0]

    categories = np.unique(train_labels)

    # -- remove background categories
    if bkg_categories is not None:
        categories = list(set(categories).difference(set(bkg_categories)))

    n_categories = len(categories)

    cat_index = {}
    predictions = np.empty((n_test, n_categories))

    # -- train OVA SVMs in parallel
    predictions = Parallel(n_jobs=-1) (delayed(one_svm) (ktrain,
                                                  train_labels.reshape(-1),
                                                  ktest,
                                                  cat, C)
           for cat in categories)

    predictions = np.array(predictions).T

    # -- iterates over categories
    for icat, cat in enumerate(categories):
        cat_index[cat] = icat

    gt = np.array([cat_index[e]
                        for e in test_labels.reshape(-1)]).astype('int')
    pred = predictions.argmax(axis=1)
    acc = (pred == gt).sum() / float(n_test)

    return acc, predictions, gt


def svm_ova_from_splits(data, labels, splits, 
                        C=DEFAULT_REGULARIZATION,
                        trace_normalization=DEFAULT_TRACE_NORMALIZATION,
                        bkg_categories=None
                        ):
    """
    Splits are expect to be a list with dictionaries containing 'train' and
    'test' keys which, in turn, will contain indexes to samples in <data>.
    """

    n_splits = len(splits)

    # -- cast larrays back to ndarrays
    data = np.array(data)
    labels = np.array(labels)

    # -- kernel matrix is computed only once
    kernel = np.dot(data, data.T)

    def one_fold(train_idxs, test_idxs):

        train_labels = labels[train_idxs]
        test_labels = labels[test_idxs]

        kernel_train = kernel[train_idxs]
        kernel_test = kernel[test_idxs]

        kernel_train = kernel_train[:, train_idxs]
        kernel_test = kernel_test[:, train_idxs]

        if trace_normalization:
            kernel_trace = kernel_train.trace()
            kernel_train = kernel_train / kernel_trace
            kernel_test = kernel_test / kernel_trace

        return svm_ova_from_kernel(kernel_train, train_labels,
                                   kernel_test, test_labels,
                                   C,
                                   bkg_categories)

    acc = np.empty((n_splits,), dtype='float32')
    r_dict = {}

    for s_idx, s in enumerate(splits):
        train_idxs = s['train']
        test_idxs = s['test']

        acc[s_idx], pred, gt = one_fold(train_idxs, test_idxs)

        r_dict['split_%d' % s_idx] = {'acc': acc[s_idx],
                                      'predictions': pred,
                                      'ground_truth': gt}

    return acc.mean(), r_dict


def one_vs_world(kernel, labels, one_label,
                 one_idxs, world_idxs, test_idxs, C):

    train_idxs = world_idxs + one_idxs
    train_labels = labels[train_idxs]

    ktrain = kernel[train_idxs]
    ktrain = ktrain[:,train_idxs]

    ktest = kernel[test_idxs]
    ktest = ktest[:,train_idxs]

    lcat = np.zeros(train_labels.size)

    lcat[train_labels != one_label] = -1
    lcat[train_labels == one_label] = +1

    svm = SVC(kernel='precomputed', C=C, tol=1e-5)
    svm.fit(ktrain, lcat)

    return svm.decision_function(ktest)[:, 0]


def svm_one_vs_world(data, labels, world_idxs, train_idxs, test_idxs,
                     C=DEFAULT_REGULARIZATION):

    def sighandler_svm(signum, frame):
        logger.info('Caught signal %i while training SVMs in paralell.'
                    % signum)

    signal.signal(signal.SIGTERM, sighandler_svm)

    # -- cast larrays back to ndarrays
    data = np.array(data)
    labels = np.array(labels)

    # -- kernel matrix is computed only once
    kernel = np.dot(data, data.T)

    train_labels = labels[train_idxs]
    unq_train_labels = np.unique(train_labels)
    arr_train_idxs = np.array(train_idxs)

    # -- train one versus world SVMs in parallel
    predictions = Parallel(n_jobs=-1) (delayed(one_vs_world) (
                           kernel, labels, one_label,
                           arr_train_idxs[train_labels==one_label].tolist(),
                           world_idxs, test_idxs,
                           C)
           for one_label in unq_train_labels)

    predictions = np.array(predictions)

    return predictions, unq_train_labels
