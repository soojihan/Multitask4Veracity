import pickle
import os
from pprint import pprint as pp
from glob import glob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, accuracy_score
def check_input():
    np.set_printoptions(threshold=np.nan)
    pd.set_option('display.expand_frame_repr', False)

    # data_path = './saved_data/saved_data_MTL2_detection/putinmissing/'
    # data_path = './saved_data/source-tweets/boston-9000-0.3/'
    # data_path = glob(os.path.join('..', 'saved_data/source-tweets-context/*'))
    # data_path = glob(os.path.join('..', 'saved_data/source-tweets-augmented-context-top25/*'))
    data_path = glob(os.path.join('..', 'saved_data/source-tweets-augmented-context-top25/*'))
    for d in data_path:
        print(d)
        x = np.load(os.path.join(d, 'rnr_labels.npy'))
        # print(type(x))
        print(len(x))
        print(x.shape)
        # print(type(x[0]))
        # x = np.load(os.path.join(data_path, 'train_array.npy'))
        x = np.load(os.path.join(d, 'train_array.npy'))
        # print(type(x))
        print(len(x))
        print(x.shape)
        # print(type(x[0]))
        print("")
        # raise SystemExit
# check_input()

def load_output(output_file):
    """
    Read output
    :return:
    """
    with open(output_file, 'rb') as f:
        output = pickle.load(f)
        f.close()
    pp(output.keys())
    pp(output['TaskC'])
    pp(output['Params'])
    # print(output['attachments']['allfolds'])
    folds = output['attachments']['allfolds']

    print(len(folds))
    folds_name = ['charliehebdo', 'germanwings-crash' ,
                  'ferguson', 'ottawashooting', 'sydneysiege','boston']
    # folds_name = ['charliehebdo', 'germanwings-crash',
    #               'ferguson', 'ottawashooting', 'sydneysiege']

    for i, name in zip(range(len(folds)), folds_name):

        target_folds = folds[i]
        label = target_folds['Task C']['Label']
        label = [1 if x == True else 0 for x in label ]
        pred = target_folds['Task C']['Prediction']
        P,R,F, _ = precision_recall_fscore_support(
            label,
            pred,
        average='binary')
        acc=accuracy_score(label, pred)
        print(name)
        # print(P, R ,F)
        print("P: {:0.4f}, F: {:0.4f}, R: {:0.4f}, acc: {:0.4f}".format(P, F, R, acc))
        print("")

    # print(len(output['attachments']['allfolds']))
    # print(len(output['attachments']['Task C']['ID']))

def save_params():
    """
    Save parameters in .txt format
    :return:
    """
    # params_file = 'output/pheme/bestparams_MTL2_detection_PHEME5.txt'
    # params_file = os.path.join('..', 'output/outputMTL2_detection_aug9000-trial20.txt')
    params_file = os.path.join('..', 'output/pheme5top25+trial50/bestparams_MTL2_detection_PHEME5-context-top25.txt')
    params_file = os.path.join('..', 'output/pheme5top25+trial20/bestparams_pheme5-context-top25-trial20.txt')
    # with open(params_file, 'rb') as f:
    #     params = pickle.load(f)
    #     pp(params)
    #     params['num_dense_layers']= 2
    #     params['num_lstm_layers'] = 1


    # params = {'batchsize': 32,
    #  'l2reg': 0.08470831444286872,
    #  'learn_rate': 0.1884857857509769,
    #  'num_dense_layers': 2,
    #  'num_dense_units': 446,
    #  'num_epochs': 50,
    #  'num_lstm_layers': 4,
    #  'num_lstm_units': 194}

    # with open(params_file, 'wb') as f:
    #     pickle.dump(params, f)
    #     f.close()
    #
    with open(params_file, 'rb') as f:
        x = pickle.load(f)
        f.close()
    pp(x)

#
# output_file = os.path.join('..', 'output/outputMTL2_detection_augmented-9000_randomsearch.pkl')
output_file = os.path.join('..', 'output/aug9000top25+trial50/outputMTL2_detection_augmented-9000-context-top25.pkl')
# output_file = os.path.join('..', 'output/aug9000top25+trial20/outputaug9000-context-top25-trial20.pkl')
# output_file = os.path.join('..', 'output/pheme5top25+trial50/outputMTL2_detection_PHEME5-context-top25.pkl')
# output_file = os.path.join('..', 'output/pheme5top25+trial20/outputpheme5-context-top25-trial20.pkl')
load_output(output_file)

# save_params()
