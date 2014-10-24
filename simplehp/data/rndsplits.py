# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import os
from glob import glob
import numpy as np

from base import Dataset
from simplehp.util.util import (get_folders_recursively, load_imgs)


class RndSplitDataset(Dataset):
    """
    Interface for datasets whose evaluation protocol consists on randomly
    splitting the samples.
    """

    def __init__(self, path, img_type, img_shape,
                 hp_nsplits, hp_ntrain, hp_ntest,
                 pt_nsplits, pt_ntrain, pt_ntest,
                 bkg_categories, seed=42):

        self.path = path
        self.img_type = img_type
        self.img_shape = img_shape
        self.hp_nsplits = hp_nsplits
        self.hp_ntrain = hp_ntrain
        self.hp_ntest = hp_ntest
        self.pt_nsplits = pt_nsplits
        self.pt_ntrain = pt_ntrain
        self.pt_ntest = pt_ntest
        self.bkg_categories = bkg_categories
        self.rng = np.random.RandomState(seed)


    def __build_meta(self):
        """
        Retrieve dataset metadata, which, in this case, consists of image paths
        and labels. The latter assumed as the directory name where the image
        files are located.
        """

        folders = np.array(sorted(get_folders_recursively(
                           self.path, self.img_type)))

        all_fnames = []
        all_labels = []

        for folder in folders:

            fnames = sorted(glob(os.path.join(self.path, folder,
                                              '*' + self.img_type)))
            for fname in fnames:

                label = (os.path.split(fname)[0]).split('/')[-1]

                all_fnames += [fname]
                all_labels += [label]

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.arange(all_labels.size)

        # -- retain hp_ntrain + hp_ntest images per category for 
        #    hyperoptimization
        categories = np.unique(all_labels)
        hp_samples_cat = self.hp_ntrain + self.hp_ntest
        hp_idxs = []
        for cat in categories:
            cat_idxs = np.argwhere(all_labels==cat)[:,0]
            assert cat_idxs.size >= max(hp_samples_cat,
                                       self.pt_ntrain) + self.pt_ntest
            shuffle = self.rng.permutation(cat_idxs.size)[:hp_samples_cat]
            hp_idxs += sorted(cat_idxs[shuffle])

        # -- exclude hp samples from all_idxs in order to select test samples
        #    for evaluation according to the protocol
        test_idxs_to_sample = np.array(list(set(all_idxs).difference(
                                set(hp_idxs))))
        test_labels_to_sample = all_labels[test_idxs_to_sample]

        pt_splits = []
        for s in xrange(self.pt_nsplits):

            # select test samples for this split
            test_idxs = []
            for cat in categories:
                cat_idxs = np.argwhere(test_labels_to_sample==cat)[:,0]
                assert cat_idxs.size > self.pt_ntest
                shuffle = self.rng.permutation(cat_idxs.size)[:self.pt_ntest]

                if cat not in self.bkg_categories:
                    test_idxs += sorted(test_idxs_to_sample[cat_idxs[shuffle]])

            # -- exclude test samples from all_idxs in order to select train
            #    samples for evaluation according to the protocol
            train_idxs_to_sample = np.array(list(set(all_idxs).difference(
                                                            set(test_idxs))))
            train_labels_to_sample = all_labels[train_idxs_to_sample]

            train_idxs = []
            for cat in categories:
                cat_idxs = np.argwhere(train_labels_to_sample==cat)[:,0]
                assert cat_idxs.size >= self.pt_ntrain
                shuffle = self.rng.permutation(cat_idxs.size)[:self.pt_ntrain]
                train_idxs += sorted(train_idxs_to_sample[cat_idxs[shuffle]])

            pt_splits += [{'train': train_idxs, 'test': test_idxs}]

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'hp_idxs': hp_idxs,
                  'pt_splits': pt_splits,
                  }

        return r_dict


    def __get_meta(self):
        try:
            return self._meta
        except AttributeError:
            self._meta = self.__build_meta()
            return self._meta
    meta = property(__get_meta)


    def __get_imgs(self):

        try:
            return self._imgs
        except AttributeError:
            # -- load all images in memory because dataset is not large
            self._imgs = load_imgs(self.meta['all_fnames'],
                                   out_shape=self.img_shape,
                                   dtype='uint8')

            self._imgs = np.rollaxis(self._imgs, 3, 1)
            self._imgs = np.ascontiguousarray(self._imgs)

            return self._imgs
    imgs = property(__get_imgs)


    def hp_imgs(self):

        return self.imgs[self.meta['hp_idxs']]


    def __build_hp_splits(self):
        """
        Randomly split hyperoptimization samples according to the given number
        of train and test samples for the task.
        """
        hp_labels = self.meta['all_labels'][self.meta['hp_idxs']]

        categories = np.unique(hp_labels)
        hp_samples_cat = self.hp_ntrain + self.hp_ntest
        hp_splits = []

        for s in xrange(self.hp_nsplits):

            hp_train_idxs = []
            hp_test_idxs = []

            for cat in categories:
                cat_idxs = np.argwhere(hp_labels==cat)[:,0]
                assert cat_idxs.size == hp_samples_cat
                shuffle = self.rng.permutation(cat_idxs.size)
                hp_train_idxs += sorted(cat_idxs[shuffle[:self.hp_ntrain]])

                if cat not in self.bkg_categories:
                    hp_test_idxs += sorted(cat_idxs[shuffle[self.hp_ntrain:]])

            hp_splits += [{'train': hp_train_idxs, 'test': hp_test_idxs}]

        return hp_splits


    def __get_hp_splits(self):
        try:
            return self._hp_splits
        except AttributeError:
            self._hp_splits = self.__build_hp_splits()
            return self._hp_splits
    hp_splits = property(__get_hp_splits)


    def hp_eval(self, algo, feat_set):

        hp_labels = self.meta['all_labels'][self.meta['hp_idxs']]

        # -- normalize features
        f_mean = feat_set.mean(axis=0)
        f_std = feat_set.std(axis=0, ddof=1)
        f_std[f_std==0.] = 1.

        feat_set -= f_mean
        feat_set /= f_std

        acc, r_dict = algo(feat_set, hp_labels, self.hp_splits,
                           bkg_categories=self.bkg_categories)

        return {'loss': 1. - acc}


    def protocol_imgs(self):

        return self.imgs


    def protocol_eval(self, algo, feat_set):

        all_labels = self.meta['all_labels']
        pt_splits = self.meta['pt_splits']

        acc = np.empty((self.pt_nsplits,), dtype='float32')

        # -- we have to normalize for each split independently
        for s_idx, s in enumerate(pt_splits):

            train_idxs = s['train']
            test_idxs = s['test']

            split_idxs = train_idxs + test_idxs
            ntrain = len(train_idxs)
            ntest = len(test_idxs)

            train_idxs = list(np.arange(ntrain))
            test_idxs = list(np.arange(ntest)+ntrain)

            feat_set_split = feat_set[split_idxs]
            label_split = all_labels[split_idxs]

            # -- normalize features
            f_mean = feat_set_split[train_idxs].mean(axis=0)
            f_std = feat_set_split[train_idxs].std(axis=0, ddof=1)
            f_std[f_std==0.] = 1.

            feat_set_split -= f_mean
            feat_set_split /= f_std

            protocol_split = [{'train': train_idxs,
                               'test': test_idxs,
                              }]

            s_acc, r_dict = algo(feat_set_split, label_split, protocol_split,
                                 bkg_categories=self.bkg_categories)

            assert s_acc.size == 1

            acc[s_idx] = s_acc

        r_dict = {'loss': 1. - acc.mean(),
                  'acc': acc.mean(),
                  }

        return r_dict


def PubFig83(path, img_type='jpg', img_shape=None):
    """
    Expecting data as downloaded from 
    https://www.dropbox.com/s/0ez5p9bpjxobrfv/pubfig83-aligned.tar.bz2
    """
    return RndSplitDataset(path, img_type, img_shape,
                           hp_nsplits=4, hp_ntrain=20, hp_ntest=20,
                           pt_nsplits=10, pt_ntrain=90, pt_ntest=10,
                           bkg_categories=[None,])


def CalTech256(path, img_type='jpg', img_shape=(351,351)):
    """
    Expecting data as downloaded from 
    http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
    """
    return RndSplitDataset(path, img_type, img_shape,
                           hp_nsplits=3, hp_ntrain=20, hp_ntest=10,
                           pt_nsplits=10, pt_ntrain=30, pt_ntest=15,
                           bkg_categories=['257.clutter'])
