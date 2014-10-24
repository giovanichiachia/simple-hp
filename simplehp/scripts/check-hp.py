# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import optparse
import json

import numpy as np

from simplehp.util.util import load_hp, readable_hps

# -- default hyperparameter aggregation list
hp_agg_list = '_depth,_max_axis'


def check_hp(hp_fname, host, port, hp_agg_list):

    hp_space, trials, n_startup_trials = load_hp(hp_fname, host, port)


    def nested_aggregation(agg_dict, param_nest, trial, nest_idx=0):

        if nest_idx == len(param_nest):
            return
        else:
            agg_key = param_nest[nest_idx]

            try:
                trial_key  = trial['hps'][agg_key]
            except KeyError:
                return

            trial_loss = trial['result']['loss']

            try:
                agg_dict[agg_key][trial_key]['count'] += 1

                if trial_loss < agg_dict[agg_key][trial_key]['best']:
                    agg_dict[agg_key][trial_key]['best'] = trial_loss

            except KeyError:
                if not agg_dict.has_key(agg_key):
                    # -- new aggregation level
                    agg_dict[agg_key] = {}

                agg_dict[agg_key][trial_key] = {'count':  1,
                                                'best': trial_loss}

            nested_aggregation(agg_dict[agg_key][trial_key], param_nest,
                               trial, nest_idx + 1)

    # -- accumulators
    rdict = {"hp_space": hp_space,
             "n_startup_trials": n_startup_trials,
             "ok": {
                "count": 0,
                "best":  None,
                "agg": {},
                },
             "random_trials": {
                "count": 0,
                "ok": {},
                "fail": {},
                "best":  None,
                },
             "tpe_trials": {
                "count": 0,
                "best":  None,
                "ok": {},
                "fail": {},
                },
            }

    best_rnd = 1.0
    best_tpe = 1.0

    # -- transform
    readable_trials, best_hps = readable_hps(trials, hp_space)

    keys = np.array(readable_trials.keys()).astype('int')
    keys.sort()

    for i_trial, k in enumerate(keys):

        trial = readable_trials[k]

        status = trial['result']['status']

        if status in ('ok', 'fail'):

            loss = trial['result']['loss']

            if i_trial < n_startup_trials:

                rnd = rdict['random_trials']
                rnd['count'] += 1

                nested_aggregation(rnd[status], hp_agg_list, trial)

                if loss < best_rnd:
                    rnd['best'] = trial
                    best_rnd = loss

            else:
                tpe = rdict['tpe_trials']
                tpe['count'] += 1

                nested_aggregation(tpe[status], hp_agg_list, trial)

                if loss < best_tpe:
                    tpe['best'] = trial
                    best_tpe = loss

            if status == 'ok':

                ok = rdict['ok']
                ok['count'] += 1
                nested_aggregation(ok['agg'], hp_agg_list, trial)

    rdict['ok']['best'] = best_hps

    print json.dumps(rdict, indent=4)

    print 'done!'


def get_optparser():

    usage = ("Usage: %prog\n\n"
             "Helper program to retrieve information from hyperparameter "
             "optimization.\n"
             "Information is retrieved either from Pickle files created by "
             "optimization \n"
             "running in serial mode or from MongoDB databases created by "
             "optimization \n"
             "running in asynchronous, parallel mode."
            )


    parser = optparse.OptionParser(usage=usage)

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

    parser.add_option("--hp_agg_list", "-A",
                      default=hp_agg_list,
                      type="str",
                      metavar="STR",
                      help=("List of hyperparameters separated by comma to "
                            "perform nested aggregation. [DEFAULT='%default']"
                            )
                        )

    return parser


def main():
    parser = get_optparser()
    opts, args = parser.parse_args()

    if len(args) != 0 or (opts.hp_fname is None and opts.host is None):
        parser.print_help()
    else:
        
        hp_fname = opts.hp_fname
        host = opts.host
        port = opts.port

        hp_agg_list = opts.hp_agg_list.split(',')

        check_hp(hp_fname, host, port, hp_agg_list)

if __name__ == "__main__":
    main()
