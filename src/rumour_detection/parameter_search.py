#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pickle
import os
from hyperopt import fmin, tpe, hp, Trials
import numpy as np
import outer

save_path = outer.save_path


def parameter_search(ntrials, objective_function, fname):
    ## Grid search

    search_space = {'num_dense_layers': hp.choice('nlayers', [1, 2]),
                    'num_dense_units': hp.choice('num_dense', [300, 400,
                                                               500, 600]),
                    'num_epochs': hp.choice('num_epochs', [50]),
                    'num_lstm_units': hp.choice('num_lstm_units', [100, 200,
                                                                   300]),
                    'num_lstm_layers': hp.choice('num_lstm_layers', [1, 2]),
                    'learn_rate': hp.choice('learn_rate', [1e-4, 1e-3]),
                    'batchsize': hp.choice('batchsize', [32]),
                    'l2reg': hp.choice('l2reg', [1e-3, 1e-4])

                    }



    trials = Trials()


    best = fmin(objective_function,
                space=search_space,
                algo=tpe.suggest,
                max_evals=ntrials,
                trials=trials)

    params = trials.best_trial['result']['Params']
    print(params)

    directory = save_path
    os.makedirs(directory, exist_ok=True)
    print(directory)


    f = open(os.path.join(directory, 'trials_' + fname + '.txt'), "wb")
    pickle.dump(trials, f)
    f.close()

    filename = os.path.join(directory, 'bestparams_' + fname + '.txt')
    f = open(filename, "wb")
    pickle.dump(params, f)
    f.close()

    return params


