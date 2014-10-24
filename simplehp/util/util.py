# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import os
import cPickle

import numpy as np
from scipy import misc

import hyperopt
from hyperopt.mongoexp import MongoTrials
from simplehp.data.base import mongo_dbname


class SimpleHpStop(Exception):
    """
    Helper class to catch stopping condition by reaching number of ok trials
    """
    pass


def get_folders_recursively(path, type):
    """
    Helper function to recursively retrieve all folders containing files of
    type <type> in a given path.
    """

    ltype = str.lower(type)
    folders = []

    for root, subFolders, files in os.walk(path):
        for file in files:
            lfile_type = str.lower(file[-len(type):])
            if lfile_type == ltype:
                folders += [os.path.relpath(root, path)]
                break

    return folders


def load_imgs(fnames, out_shape=None, dtype='uint8',
              flatten=True,  minmax_norm=False):

    if minmax_norm:
        assert ('float' in dtype)

    if flatten:
        n_channels = 1
    else:
        n_channels = 3

    if out_shape == None:
        # -- read first image to retrieve output shape
        out_shape = misc.imread(fnames[0], flatten).shape[:2]
        # -- check later if all images in the dataset have the same shape
        img_resize = False
    else:
        assert len(out_shape) == 2
        img_resize = True

    n_imgs = len(fnames)
    img_set = np.empty((n_imgs,) + out_shape + (n_channels,), dtype=dtype)

    for i, fname in enumerate(fnames):

        arr = misc.imread(fname, flatten)

        if img_resize:
            # -- resize image keeping its aspect ratio and best fitting it to
            #    the desired output
            in_shape = arr.shape[:2]
            resize_shape = tuple((np.array(in_shape) /
                max(np.array(in_shape) / np.array(out_shape,
                dtype=np.float32))).round().astype(np.int))

            arr = misc.imresize(arr, resize_shape).astype(dtype)

            # -- pad the channel mean value when necessary
            pad_size = np.array(out_shape) - np.array(arr.shape)
            assert pad_size.min() == 0

            if pad_size.max() > 0:
                pad_size = pad_size / 2.
                pad_size = ((np.ceil(pad_size[0]), np.floor(pad_size[0])),
                            (np.ceil(pad_size[1]), np.floor(pad_size[1])))

                if not flatten:
                    pad_size += ((0,0),)

                img_mean = int(arr.mean().round())
                arr = np.pad(arr, pad_size, 'constant',
                            constant_values=img_mean)

        if flatten:
            arr.shape = arr.shape + (1,)

        assert arr.shape[:2] == out_shape

        if minmax_norm:
            arr -= arr.min()
            arr /= arr.max()

        img_set[i] = arr

    return img_set


def count_ok_trials(trials):

    n_ok_trials = 0

    for trial in trials:
        if trial['result']['status'] == 'ok':
            n_ok_trials += 1

    return n_ok_trials


def save_hp(hp_space, trials, n_startup_jobs, hp_fname):

    save_dict = {'hp_space': hp_space,
                 'trials': trials,
                 'n_startup_jobs': n_startup_jobs,
                }

    cPickle.dump(save_dict, open(hp_fname, 'wb'))


def load_hp(hp_fname, host, port):

    if hp_fname is not None:

        hp = cPickle.load(open(hp_fname, 'r'))
        hp_space = hp['hp_space']
        trials = hp['trials']
        n_startup_jobs = hp['n_startup_jobs']

    elif host is not None:

        trials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, int(port),
            mongo_dbname), exp_key=mongo_dbname, refresh=True)

        # -- retrieve hp_space and n_startup_jobs from trials attachment
        cmd0, cmd1 = trials.miscs[0]['cmd']
        assert cmd0 == 'domain_attachment'
        blob = trials.attachments[cmd1]
        domain = cPickle.loads(blob)
        spec = hyperopt.base.spec_from_misc(trials.miscs[0])

        memo = domain.memo_from_config(spec)
        argdict = hyperopt.pyll.rec_eval(domain.expr, memo=memo)

        hp_space = argdict['hp_space']
        n_startup_jobs = argdict['n_startup_trials']

    else:
        raise ValueError('No Pickle file nor MongoDB host informed')

    return hp_space, trials, n_startup_jobs


def hp_from_spec(spec, hp_space):

    rdict = {}
    for k in spec.keys():
        rdict[k] = hp_space[k.split('@')[-1]][spec[k]]

    return rdict


def readable_hps(trials, hp_space):

    # TODO: still need to figure out a way to do this without the use of
    #       hp_space

    # -- get best hyperparameter set
    try:
        best_spec = hyperopt.base.spec_from_misc(trials.best_trial['misc'])
        best_hps = {'hps': hp_from_spec(best_spec, hp_space),
                    'result': trials.best_trial['result']
                   }
    except Exception, e:
        raise ValueError('problem retrieving best trial: %s' % (e))

    # -- get trial hyperparameter values and results
    readable_trials = {}

    for t in trials.trials:
        spec = hyperopt.base.spec_from_misc(t['misc'])

        readable_trials[t['tid']] = {'hps': hp_from_spec(spec, hp_space),
                                     'result': t['result']
                                    }

    return readable_trials, best_hps


def sample_Fx2(labels, n_folds=5, seed=42):

    # -- initialize folds
    folds = []

    for f in xrange(n_folds):
        for s in (0, 1):
            split = f * 2 + s
            folds += [{}]
            folds[split]['train'] = []
            folds[split]['test'] = []

    rng = np.random.RandomState(seed=seed)
    categories = np.unique(labels)

    for cat in categories:

        cat_idxs = np.where([labels == cat])[1]
        n_samples_cat = cat_idxs.size

        for f in xrange(0, n_folds * 2, 2):

            shuffle = cat_idxs[rng.permutation(n_samples_cat)]
            half = n_samples_cat / 2

            folds[f+0]['train'] += shuffle[:half].tolist()
            folds[f+0]['test'] += shuffle[half:].tolist()

            folds[f+1]['train'] += shuffle[half:].tolist()
            folds[f+1]['test'] += shuffle[:half].tolist()

    # -- cast lists of indexes into array
    for f in xrange(n_folds):
        for s in (0, 1):
            split = f * 2 + s

            folds[f]['train'] = np.array(folds[f]['train'])
            folds[f]['test'] = np.array(folds[f]['test'])

    return folds


def test_sample_Fx2(folds):

    for f in folds:
        train = set(f['train'].reshape(-1).tolist())
        test = set(f['test'].reshape(-1).tolist())

        assert len(train.intersection(test)) == 0


def acc_threshold(neg_scores, pos_scores, T):

    n_scores = float(len(neg_scores) + len(pos_scores))

    n_fa = (neg_scores >= T).sum()
    n_fr = (pos_scores <  T).sum()

    return 1. - ((n_fa + n_fr) / n_scores)


def get_interesting_samples(scores, ground_truth, T, n=1):
    """
    Return the n most confusing positive and negative sample indexes. Positive
    samples have scores >= T and are labeled 1 in ground_truth. Negative
    smaples are labeled 0.
    """
    pos_hit = []
    neg_miss = []
    neg_hit = []
    pos_miss = []

    for idx, (score, gt) in enumerate(zip(scores, ground_truth)):
        if score >= T:
            if gt == 1:
                # -- positive hit
                pos_hit += [idx]
            else:
                # -- negative miss
                neg_miss += [idx]
        else:
            if gt == 0:
                # -- negative hit
                neg_hit += [idx]
            else:
                # -- positive miss
                pos_miss += [idx]

    # -- interesting samples
    scores_aux = np.empty(scores.shape, dtype=scores.dtype)

    scores_aux[:] = np.inf
    scores_aux[pos_hit] = scores[pos_hit]
    idx = min(n, len(pos_hit))
    int_pos_hit = scores_aux.argsort()[:idx]

    scores_aux[:] = np.inf
    scores_aux[neg_miss] = scores[neg_miss]
    idx = min(n, len(neg_miss))
    int_neg_miss = scores_aux.argsort()[:idx]

    scores_aux[:] = -np.inf
    scores_aux[neg_hit] = scores[neg_hit]
    idx = min(n, len(neg_hit))
    if idx == 0:
        idx = -len(scores_aux)
    int_neg_hit = scores_aux.argsort()[-idx:]

    scores_aux[:] = -np.inf
    scores_aux[pos_miss] = scores[pos_miss]
    idx = min(n, len(pos_miss))
    if idx == 0:
        idx = -len(scores_aux)
    int_pos_miss = scores_aux.argsort()[-idx:]

    r_dict = {'pos_hit': int_pos_hit,
              'neg_miss': int_neg_miss,
              'neg_hit': int_neg_hit,
              'pos_miss': int_pos_miss,
              }

    return r_dict


def eer_threshold(neg_scores, pos_scores):
    """
    For now we are assuming that that pos_scores are supposed to be
    greater than neg_scores.
    """

    assert isinstance(neg_scores, list)
    assert isinstance(pos_scores, list)

    w_neg = []
    w_pos = []
    w_all = []

    w_neg = neg_scores
    w_pos = pos_scores
    w_all = neg_scores + pos_scores

    n_neg = len(w_neg)
    n_pos = len(w_pos)

    w_neg.sort()
    w_pos.sort()
    w_all.sort()

    far1 = far2 = 1.0
    frr1 = frr2 = 0.0

    far_dec = 1.0 / len(w_neg)
    frr_inc = 1.0 / len(w_pos)

    s2 = w_all[0]
    p2 = w_all[0] - 1e-3

    i_neg = i_pos = 0

    for i in xrange(len(w_all)):

        s1 = s2
        if i == len(w_all) - 1:
            s2 = w_all[i] + 1e-3
        else:
            s2 = w_all[i+1]

        p1 = p2
        p2 = (s1 + s2) / 2

        far1 = far2
        if i_neg < n_neg and w_neg[i_neg] <= p2:
            far2 -= far_dec
            i_neg += 1

        frr1 = frr2
        if i_pos < n_pos and w_pos[i_pos] <= p2:
            frr2 += frr_inc
            i_pos += 1

        if frr2 >= far2:
            break

    assert frr2 >= far2

    # -- interpolate the two lines and find the crossing point
    slope_far = (far2 - far1) / (p2 - p1)
    slope_frr = (frr2 - frr1) / (p2 - p1)

    T = (frr1 - far1) / (slope_far - slope_frr) + p1

    return T


def get_farfrr(neg_scores, pos_scores, T):

    assert isinstance(neg_scores, list)
    assert isinstance(pos_scores, list)

    far = (np.array(neg_scores) > T).mean()
    frr = (np.array(pos_scores) <= T).mean()

    return far, frr
