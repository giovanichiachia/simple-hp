# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import os
import optparse
import inspect

from simplehp.opt.base import datasets, hp_algos, learning_algos
from simplehp.opt.hp import simple_hp
from simplehp.util.util import save_hp, SimpleHpStop
import hp_spaces

n_startup_trials = 400 # -- random trials before optimization => ~50 ok trials
n_ok_trials = 2000
opt_modes = ['serial', 'async']


def hp_dataset(dataset_class, dataset_path,
               hp_space, hp_algo, learning_algo,
               n_startup_trials, n_ok_trials, output_path,
               opt_mode, host, port):

    data_obj = dataset_class(dataset_path)
    # -- force images do be loaded and be kept in memory
    foo = data_obj.hp_imgs()


    # -- add dataset's max axis as a choice in _max_axis
    hp_space['_max_axis'] += [max(data_obj.hp_imgs().shape[-2:])]

    try:
        if opt_mode == 'serial':
            # -- set host and port in order to run simple_hp locally
            host = None
            port = None
            checkpoint_fname = os.path.join(output_path, 'checkpoint.pkl')
        else:
            checkpoint_fname = '/tmp/checkpoint.pkl'

        dataset_info = {'data_obj': data_obj,
                        'fn_imgs': 'hp_imgs',
                        'fn_eval': 'hp_eval'}

        trials = simple_hp(dataset_info, hp_algo, learning_algo, hp_space,
                   n_startup_trials, n_ok_trials, checkpoint_fname,
                   host, port)

    except SimpleHpStop, e:
        msg, trials = e.args
        print msg

    if opt_mode == 'serial':
        # -- save complete hyperparameter optimization trials in pickle format
        hp_fname = os.path.join(output_path, 'hp.pkl')
        save_hp(hp_space, trials, n_startup_trials, hp_fname)

    print 'done!'


def get_optparser():

    dataset_options = ''
    for k in sorted(datasets.keys()):
      dataset_options +=  ("     %s - %s \n" % (k, datasets[k].__name__))

    usage = ("usage: %prog <DATASET> <DATASET_PATH>\n\n"
             "DATASET is an integer corresponding to the following supported "
             "datasets:\n" + dataset_options
            )

    parser = optparse.OptionParser(usage=usage)

    hp_algo_default = hp_algos['default']
    hp_algos.pop('default', None)
    hp_algo_opts = ' OPTIONS=%s' % (hp_algos.keys())

    learn_algo_default = learning_algos['default']
    learning_algos.pop('default', None)
    learn_algo_opts = ' OPTIONS=%s' % (learning_algos.keys())

    opt_modes_str = ' OPTIONS=%s' % (opt_modes)

    parser.add_option("--opt_mode", "-M",
                      default=opt_modes[0],
                      type="str",
                      metavar="STR",
                      help="[DEFAULT='%default']" + opt_modes_str)

    parser.add_option("--output_path", "-O",
                      default=None,
                      type="str",
                      metavar="STR",
                      help=("Required when opt_mode='serial' "
                            "[DEFAULT='%default']"))

    parser.add_option("--hp_space", "-S",
                      default='default',
                      type="str",
                      metavar="STR",
                      help="[DEFAULT='%default']")

    parser.add_option("--hyperopt_algo", "-H",
                      default=hp_algo_default,
                      type="str",
                      metavar="STR",
                      help="[DEFAULT='%default']" + hp_algo_opts)

    parser.add_option("--learning_algo", "-A",
                      default=learn_algo_default,
                      type="str",
                      metavar="STR",
                      help="[DEFAULT='%default']" + learn_algo_opts)

    parser.add_option("--n_startup_trials", "-T",
                      default=n_startup_trials,
                      type="int",
                      metavar="INT",
                      help="[DEFAULT='%default']")

    parser.add_option("--n_ok_trials", "-N",
                      default=n_ok_trials,
                      type="int",
                      metavar="INT",
                      help="[DEFAULT='%default']")

    parser.add_option("--host", "-W",
                      default='localhost',
                      type="str",
                      metavar="STR",
                      help=("Only considered when optimization mode='async' "
                            "[DEFAULT='%default']"))

    parser.add_option("--port", "-P",
                      default=10921,
                      type="int",
                      metavar="INT",
                      help=("Only considered when optimization mode='async' "
                            "[DEFAULT='%default']"))

    return parser


def main():
    parser = get_optparser()
    opts, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
    else:
        try:
            dataset_class = datasets[args[0]]
        except KeyError:
            raise ValueError('invalid dataset option')

        dataset_path = args[1]

        opt_mode = opts.opt_mode
        assert opt_mode in opt_modes, 'invalid optimization mode'

        output_path = opts.output_path
        if opt_mode == 'serial':
            assert output_path is not None, ("output_path must be informed in "
                "'serial' optimization mode")

        try:
            hp_space = eval('hp_spaces.%s' % opts.hp_space)
        except KeyError:
            raise ValueError('problem importing optimization space')

        try:
            hp_algo = hp_algos[opts.hyperopt_algo]
            assert inspect.isfunction(hp_algo)
        except KeyError:
            raise ValueError('invalid hyperparameter optimization algorithm')

        try:
            learning_algo = learning_algos[opts.learning_algo]
            assert inspect.isfunction(learning_algo)
        except KeyError:
            raise ValueError('invalid learning algorithm')

        n_startup_trials = opts.n_startup_trials
        n_ok_trials = opts.n_ok_trials

        host = opts.host
        port = opts.port

        hp_dataset(dataset_class, dataset_path,
                   hp_space, hp_algo, learning_algo,
                   n_startup_trials, n_ok_trials, output_path,
                   opt_mode, host, port)

if __name__ == "__main__":
    main()
