import pickle
import os
from pprint import pprint as pp
from glob import glob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, accuracy_score

np.set_printoptions(threshold=np.nan)
pd.set_option('display.expand_frame_repr', False)



def check_input():
    data_path = glob(os.path.join('..', 'saved_data/source-tweets-augmented-context-top25/*'))
    for d in data_path:
        x = np.load(os.path.join(d, 'rnr_labels.npy'))
        print(len(x))
        print(x.shape)
        x = np.load(os.path.join(d, 'train_array.npy'))
        print(len(x))
        print(x.shape)
        print("")

def load_output(output_file: str):
    """
    Read rumour detection results
    :return:
    """
    with open(output_file, 'rb') as f:
        output = pickle.load(f)
        f.close()
    pp(output.keys())
    pp(output['TaskC'])
    pp(output['Params'])

    if 'attachment' in output:
        folds = output['attachments']['allfolds']
        folds_name = ['charliehebdo', 'germanwings' ,
                      'ferguson', 'ottawashooting', 'sydneysiege', 'boston']
        # folds_name = ['charliehebdo', 'germanwings',
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
            print("Event: ", name)
            print("P: {:0.4f}, F: {:0.4f}, R: {:0.4f}, acc: {:0.4f}".format(P, F, R, acc))
            print("")


# output_file = os.path.join('..','..', 'output', 'pheme5-aug-shuffle-twitter1516-2', 'output_pheme5-aug-shuffle-twitter1516.pkl')
print(output_file)
load_output(output_file)

# save_params()
