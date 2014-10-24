# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# Inspired in https://github.com/jaberg/skdata/blob/master/skdata/base.py
#
# License: BSD


class Dataset(object):
    """
    Base class for datasets on which convolutional network hyperparameters
    are intended to be learned.
    """

    def hp_imgs(self):
        """
        Return the set of raw images to be processed by a candidate CNN
        in a given hyperoptimization iteration. The expected return should
        be a 4-D uint8 channel-major numpy.array with indexes containing
        [img, channel, row, col].
        """

        raise NotImplementedError('implement me')


    def hp_eval(self, algo, feat_set):
        """
        Given a learning algorithm, and the extracted feature set, returns the
        loss to be used by the the hyperparameter optimization algorithm.
        """

        raise NotImplementedError('implement me')


    def protocol_imgs(self):
        """
        Return the set of raw images to be processed for performance
        computation according to the official dataset protocol. The expected
        return should be a 4-D uint8 channel-major numpy.array with indexes
        containing [img, channel, row, col].
        """

        raise NotImplementedError('implement me')


    def protocol_eval(self, algo, feat_set):
        """
        Given a learning algorithm, and the extracted feature set, returns
        performance metrics (in a dictionary) according to the dataset
        official evaluation protocol.
        """

        raise NotImplementedError('implement me')

mongo_dbname = 'simple-hp'
