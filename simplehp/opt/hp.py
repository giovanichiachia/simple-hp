# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

from time import time

import logging
import StringIO
import traceback

import numpy as np
import functools

import hyperopt
from hyperopt import hp, fmin, Trials
from hyperopt.mongoexp import MongoTrials

from hyperopt import pyll
from hyperopt.pyll import scope, rec_eval

from hpconvnet import pyll_slm, pyll_slm_sandbox
from hpconvnet.slm import call_catching_pipeline_errors, USLM_Exception

from simplehp.util.util import count_ok_trials, save_hp, SimpleHpStop
from simplehp.data.base import mongo_dbname

logger = logging.getLogger(__name__)

batchsize = 2
batched_lmap_speed_thresh = {'seconds': 2.0, 'elements': 8}
min_features = 1000
max_features = 30000
max_intermediate_size = 600000

max_evals=1000000 # -- hypothetical maximum number of jobs to execute
checkpoint_every=100

pyll.scope.import_(globals(),
    # -- from pyll
    'partial',
    'callpipe1',
    #
    # -- misc. from ./pyll_scope.py
    'pyll_theano_batched_lmap',
    #
    # -- filterbank allocators from ./pyll_scope.py
    'get_fb', # -- sandbox
    #
    # -- pipeline elements  (./pyll.scope.py)
    'slm_affine_image_warp',
    'slm_img_uint8_to_float32',
    'slm_lnorm',
    'slm_fbcorr_chmaj',
    'slm_lpool_rectlin', # -- sandbox
    #
    # -- renamed symbols
    **{
    # NEW NAME:         ORIG NAME
    's_int':           'int',
    's_float':         'float',
    'pyll_getattr':    'getattr',
    })


@pyll.scope.define
def input_imgs(imgs):
    return imgs

# -- hack to avoid using skdata larray
class lmap_ndarray(object):
    def __init__(self, array_to_map):

        assert isinstance(array_to_map, np.ndarray)

        self._mapped_array = array_to_map

    def __len__(self):
        return len(self._mapped_array)

    @property
    def shape(self):
        return self._mapped_array.shape

    @property
    def ndim(self):
        return self._mapped_array.ndim

    @property
    def dtype(self):
        return self._mapped_array.dtype

    def __getitem__(self, idx):
        return self._mapped_array[idx]

    def __array__(self):
        return self._mapped_array


def make_trials(host, port, exp_key, refresh=True):

    if (host, port) == (None, None):
        trials = Trials()
    else:
        trials = MongoTrials(
                'mongo://%s:%d/%s/jobs' % (host, int(port), mongo_dbname),
                exp_key=exp_key,
                refresh=refresh)
    return trials


def layer_space(layer, channels, hp_space):

    p = hp_space
    ln = []
    op=0

    if layer > 0:

        n_filters = hp.choice('l%d_%d_filt@n_filters' % (layer, op),
                              p['n_filters'])
        size = hp.choice('l%d_%d_filt@rf_size' % (layer, op), p['rf_size'])

        filterbank = get_fb(n_filters=n_filters, size=size, channels=channels)

        sl_filter = partial(slm_fbcorr_chmaj, kerns=filterbank)

        op += 1

        sl_pool = partial(slm_lpool_rectlin,
            ker_size=hp.choice('l%d_%d_pool@rf_size' % (layer, op),
                p['rf_size']),
            order=hp.choice('l%d_%d@pool_order' % (layer, op),
                p['pool_order']),
            stride=hp.choice('l%d_%d@pool_stride' % (layer, op),
                p['pool_stride'])
            )

        op += 1
        ln += [sl_filter, sl_pool]

    sl_norm = partial(slm_lnorm,
        ker_size=hp.choice('l%d_%d_norm@rf_size' % (layer, op), p['rf_size']),
        stretch=1.0,
        threshold=1e-5,
        )

    ln = hp.choice('_l%d@do_norm' % (layer), [ln + [sl_norm], ln])

    return ln


def build_model_space(chmajor_in_shape,
    hp_space,
    batchsize,
    max_features,
    batched_lmap_speed_thresh,
    max_intermediate_size=max_intermediate_size,
    ):

    Xh, Xw = chmajor_in_shape[2:]

    # -- make sure it is channel-major
    assert Xw > 3, chmajor_in_shape

    max_axis_size = hp.choice('_max_axis', hp_space['_max_axis'])
    scaling_factor = s_float(max_axis_size) / max(Xh, Xw)

    resizing = partial(slm_affine_image_warp,
                       rot=0,
                       shear=0,
                       scale=[scaling_factor, scaling_factor],
                       trans=[0, 0],
                       oshape=[s_int(Xh * scaling_factor),
                               s_int(Xw * scaling_factor)])

    pipeline =  [slm_img_uint8_to_float32, resizing]

    # -- fake input matrix just to get pipeline properly compiled
    foo = np.ones((batchsize,) + chmajor_in_shape[1:], dtype='uint8')
    Xcm = lmap_ndarray(foo)

    Xcm = pyll_theano_batched_lmap(
        partial(callpipe1, pipeline),
        Xcm,
        batchsize=batchsize,
        print_progress_every=10,
        speed_thresh=batched_lmap_speed_thresh,
        x_dtype='uint8',
        )[:]

    max_layers = len(hp_space['_depth']) + 1
    depth_choice = []

    # -- models from one to three layers
    for l in xrange(max_layers):

        n_channels_pipe = pyll_getattr(Xcm, 'shape')[1]
        layer_pipe = layer_space(l, n_channels_pipe, hp_space)
        pipeline = pipeline + layer_pipe

        # -- model has to have at least two layers including l0
        if l > 0:
            depth_choice.append(pipeline)

        if l < max_layers-1:
            abort_on_rows_larger_than = max_intermediate_size
        else:
            abort_on_rows_larger_than = max_features

        Xcm = pyll_theano_batched_lmap(
            partial(callpipe1, layer_pipe),
            Xcm,
            batchsize=batchsize,
            print_progress_every=10,
            speed_thresh=batched_lmap_speed_thresh,
            abort_on_rows_larger_than=abort_on_rows_larger_than,
            )[:]

    pipeline = hp.choice('_depth', depth_choice)

    return pipeline


# def build_search_space(fn_imgs, fn_eval, learning_algo, hp_space,
def build_search_space(dataset_info, learning_algo, hp_space,
    n_startup_trials=1000, n_ok_trials=2000,
    batchsize=batchsize,
    min_features=min_features,
    max_features=max_features,
    batched_lmap_speed_thresh=batched_lmap_speed_thresh,
    checkpoint_fname='./checkpoint.pkl',
    ):

    fn_imgs = getattr(dataset_info['data_obj'], dataset_info['fn_imgs'])
    model_space = build_model_space(fn_imgs().shape,
                                    hp_space,
                                    batchsize,
                                    max_features,
                                    batched_lmap_speed_thresh)

    search_space = {
            'dataset_info': dataset_info,
            'learning_algo': learning_algo,
            'hp_space': hp_space,
            'pipeline': model_space,
            'n_startup_trials': n_startup_trials,
            'n_ok_trials': n_ok_trials,
            'batchsize': batchsize,
            'min_features': min_features,
            'max_features': max_features,
            'checkpoint_fname': checkpoint_fname,
            'batched_lmap_speed_thresh': batched_lmap_speed_thresh,
            'ctrl': hyperopt.Domain.pyll_ctrl,
            }

    return search_space


@hyperopt.fmin_pass_expr_memo_ctrl
def objective(expr, memo, ctrl):

    def rdict_error(e):

        sio = StringIO.StringIO()
        traceback.print_exc(None, sio)
        tb = sio.getvalue()

        return {'loss': float(1.0),
                'status': hyperopt.STATUS_FAIL,
                'failure': {
                    'type': str(type(e)),
                    'exc': repr(e),
                    'tb': tb,
                }}

    def exception_thrower():

        argdict = rec_eval(expr, memo=memo, print_node_on_error=False)

        dataset_info = argdict['dataset_info']
        learning_algo = argdict['learning_algo']
        hp_space = argdict['hp_space']
        pipeline = argdict['pipeline']
        n_startup_trials = argdict['n_startup_trials']
        n_ok_trials = argdict['n_ok_trials']
        batchsize = argdict['batchsize']
        min_features = argdict['min_features']
        max_features = argdict['max_features']
        checkpoint_fname = argdict['checkpoint_fname']
        batched_lmap_speed_thresh = argdict['batched_lmap_speed_thresh']
        ctrl = argdict['ctrl']

        tid = ctrl.current_trial['tid']

        # -- checkpoint
        if isinstance(ctrl.trials, Trials):
            if tid > 0 and tid % checkpoint_every == 0:
                save_hp(hp_space, ctrl.trials, n_startup_trials,
                        checkpoint_fname)

        # -- retrieve trials from database
        if isinstance(ctrl.trials, MongoTrials):
            ctrl.trials.refresh()

        # -- check and signal stopping to optimizer
        current_ok_trials = count_ok_trials(ctrl.trials)
        if current_ok_trials >= n_ok_trials:
            raise SimpleHpStop('number of ok trials reached - '
                               'stopping process with %d ok trials out of '
                               '%d trials.' % (
                               current_ok_trials, tid),
                               ctrl.trials)

        # -- feature extraction
        slm_t0 = time()

        fn_imgs = getattr(dataset_info['data_obj'], dataset_info['fn_imgs'])
        imgs = fn_imgs()

        limgs = lmap_ndarray(imgs)

        X = pyll_theano_batched_lmap(
            partial(callpipe1, pipeline),
            limgs,
            batchsize=batchsize,
            print_progress_every=10,
            speed_thresh=batched_lmap_speed_thresh,
            abort_on_rows_larger_than=max_features,
            x_dtype='uint8',
            )[:]

        feat_set = rec_eval(X, print_node_on_error=False)
        slm_time = time() - slm_t0

        # -- classification
        eval_t0 = time()

        # -- feat_set in 2-D
        feat_shape = feat_set.shape
        feat_set.shape = feat_set.shape[0], -1

        assert feat_set.shape[1] >= min_features, 'min_features not satisfied'

        fn_eval = getattr(dataset_info['data_obj'], dataset_info['fn_eval'])
        r_dict = fn_eval(learning_algo, feat_set)
        eval_time = time() - eval_t0

        r_dict['status'] = hyperopt.STATUS_OK
        r_dict['feat_shape'] = feat_shape
        r_dict['slm_time'] = slm_time
        r_dict['eval_time'] = eval_time

        return r_dict

    try:
        rdict = call_catching_pipeline_errors(exception_thrower)
    except USLM_Exception, e:
        exc, rdict = e.args
        logger.info('trial failed: %s: %s' % (type(e), exc))
    except SimpleHpStop, e:
        raise
    except Exception, e:
        rdict = rdict_error(e)
        logger.info('trial failed: Unknown: %s' % (e))

    return rdict


def simple_hp(dataset_info, hp_algo, learning_algo, hp_space,
              n_startup_trials, n_ok_trials, checkpoint_fname,
              host=None, port=None):

    search_space = build_search_space(dataset_info,
                                      learning_algo,
                                      hp_space,
                                      n_startup_trials,
                                      n_ok_trials,
                                      checkpoint_fname=checkpoint_fname)

    if 'tpe' in hp_algo.__globals__['__name__']:
        hp_algo=functools.partial(
                    hp_algo,
                    n_startup_jobs=n_startup_trials)

    trials = make_trials(host, port, exp_key=mongo_dbname)

    # -- minimize the objective over the space
    fmin(fn=objective,
         space=search_space,
         algo=hp_algo,
         max_evals=max_evals,
         trials=trials,
         rstate=np.random.RandomState(seed=63))

    return trials
