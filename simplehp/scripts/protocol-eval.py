# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import os
import optparse
import pprint

from scipy import misc

import hyperopt

from simplehp.opt.base import datasets, learning_algos
from simplehp.opt.hp import objective, build_search_space
from simplehp.util.util import load_hp

speed_thresh = {'seconds': 2.0, 'elements': 4}

def interesting_samples(int_dict):

    output_path = './int_samples'

    # -- make sure that the output_path exists, otherwise, create it
    if not os.path.exists('./int_samples'):
        os.makedirs(output_path)

    print '\ninterestig samples:'

    for int_key in int_dict.keys():
        int_output_path = os.path.join(output_path, int_key)
        if not os.path.exists(int_output_path):
            os.makedirs(int_output_path)

        # -- save files
        for fname, img in zip(int_dict[int_key]['fnames'],
                              int_dict[int_key]['imgs']):

            dest_fname = os.path.split(fname)[-1]
            dest_fname = os.path.join(int_output_path, dest_fname)

            print '%s: %s' % (int_key, fname)
            misc.imsave(dest_fname, img)

    return


def protocol_eval(dataset, dataset_path, hp_fname, host, port,
                  learning_algo, int_samples):

    data_obj = dataset(dataset_path)

    # -- load hp file
    hp_space, trials, _ = load_hp(hp_fname, host, port)

    # -- best trial
    try:
        best_trial = trials.best_trial
    except Exception, e:
        raise ValueError('problem retrieving best trial: %s' % (e))

    dataset_info = {'data_obj': data_obj,
                    'fn_imgs': 'protocol_imgs',
                    'fn_eval': 'protocol_eval'}

    search_space = build_search_space(dataset_info,
                                      learning_algo,
                                      hp_space=hp_space,
                                      n_ok_trials=1000000,
                                      batched_lmap_speed_thresh=speed_thresh)

    ctrl = hyperopt.Ctrl(trials=trials, current_trial=best_trial)
    domain = hyperopt.Domain(objective, search_space)

    best_hps = hyperopt.base.spec_from_misc(best_trial['misc'])

    r_dict = domain.evaluate(best_hps, ctrl, attach_attachments=True)

    if r_dict['status'] == 'ok':
        print '\nperformance according to dataset protocol:\n'
        for key in r_dict:
            if key == 'int_samples':
                if int_samples:
                    interesting_samples(r_dict['int_samples'])
            else:
                print key, pprint.pformat(r_dict[key])
    else:
        print '\n', r_dict['failure']['tb']
    print 'done!'


def get_optparser():

    dataset_options = ''
    for k in sorted(datasets.keys()):
      dataset_options +=  ("     %s - %s \n" % (k, datasets[k].__name__))

    usage = ("usage: %prog <DATASET> <DATASET_PATH>\n\n"
             "+ DATASET is an integer corresponding to the following supported "
             "datasets:\n" + dataset_options + '\n'
             "+ HP_FNAME is the pkl file containing the result of a previous "
             "hyperparameter optimization."
            )

    parser = optparse.OptionParser(usage=usage)

    learn_algo_opts = ' OPTIONS=%s' % (learning_algos.keys())

    parser.add_option("--hp_fname", "-F",
                      default=None,
                      type="str",
                      metavar="STR",
                      help=("Pickle file created by optimization in serial"
                            "mode. [DEFAULT='%default']"
                            )
                        )

    parser.add_option("--host", "-H",
                      default=None,
                      type="str",
                      metavar="STR",
                      help=("Host serving MongoDB database created by "
                            "optimization running in asynchronous, parallel "
                            "mode. [DEFAULT='%default']"
                            )
                        )

    parser.add_option("--port", "-P",
                      default=10921,
                      type="int",
                      metavar="INT",
                      help=("MongoDB port at host serving the database. "
                            "[DEFAULT='%default']"
                            )
                        )

    parser.add_option("--learning_algo", "-A",
                      default='svm',
                      type="str",
                      metavar="STR",
                      help="[DEFAULT='%default']" + learn_algo_opts)

    parser.add_option("--int_samples", "-S",
                      default=False,
                      action="store_true",
                      help="output interesting misses and hits in current " \
                           "path [DEFAULT='%default']")


    return parser


def main():
    parser = get_optparser()
    opts, args = parser.parse_args()

    if len(args) != 2 or (opts.hp_fname is None and opts.host is None):
        parser.print_help()
    else:
        try:
            dataset = datasets[args[0]]
        except KeyError:
            raise ValueError('invalid dataset option')

        dataset_path = args[1]

        hp_fname = opts.hp_fname
        host = opts.host
        port = opts.port

        try:
            learning_algo = learning_algos[opts.learning_algo]
        except KeyError:
            raise ValueError('invalid learning algorithm')

        int_samples = opts.int_samples

        protocol_eval(dataset, dataset_path, hp_fname, host, port,
                      learning_algo, int_samples)

if __name__ == "__main__":
    main()
