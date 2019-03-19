import pickle

def load_output(output_file):
    """
    Read output
    :return:
    """
    with open(output_file, 'rb') as f:
        output = pickle.load(f)
        pp(output)
        f.close()
    # pp(output['TaskC'])
    # pp(output['Params'])

def save_params():
    """
    Save parameters in .txt format
    :return:
    """
    params_file = 'output/pheme/bestparams_MTL2_detection_PHEME5.txt'
    params_file = 'output/boston9000+randomsearch/outputMTL2_detection_augmented-9000.txt'
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
        pp(params)
        params['num_dense_layers']= 2
        params['num_lstm_layers'] = 1

    # params = {'batchsize': 32,
    #  'l2reg': 0.001,
    #  'learn_rate': 0.001,
    #  'num_dense_layers': 2,
    #  'num_dense_units': 400,
    #  'num_epochs': 50,
    #  'num_lstm_layers': 1,
    #  'num_lstm_units': 300}
    #
    # params = {'batchsize': 32,
    #  'l2reg': 0.08470831444286872,
    #  'learn_rate': 0.1884857857509769,
    #  'num_dense_layers': 2,
    #  'num_dense_units': 446,
    #  'num_epochs': 50,
    #  'num_lstm_layers': 4,
    #  'num_lstm_units': 194}

    with open(params_file, 'wb') as f:
        pickle.dump(params, f)
        f.close()

    # with open(params_file, 'rb') as f:
    #     x = pickle.load(f)
    #     f.close()
    # pp(x)

# output_file = './output/pheme+boston7000/outputMTL2_detection_augmented-8500.pkl'
# output_file = './output/d/outputMTL2_detection_PHEME5.pkl'
# output_file = './output/outputMTL2_detection_augmented-9000_pretrained.pkl'
output_file = './output/pheme5+randomsearch/trials_MTL2_detection_PHEME5-randomsearch.txt'
load_output(output_file)
