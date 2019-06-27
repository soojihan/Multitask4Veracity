import pickle
import os
from pprint import pprint as pp
from glob import glob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
np.set_printoptions(threshold=np.nan)
pd.set_option('display.expand_frame_repr', False)



def check_input():
    # data_path = glob(os.path.join('..', 'saved_data/source-tweets-augmented-context-top25/*'))
    # data_path = glob(os.path.join('..', '..', 'data/saved_data/saved_data_MTL2_detection/charliehebdo'))
    data_path = glob(os.path.join('..', '..', 'data/recollect/aug/ferguson'))
    for d in data_path:
        x = np.load(os.path.join(d, 'rnr_labels.npy'))
        print(len(x))
        print(x.shape)
        print(Counter(x))
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

    if 'attachments' in output:
        folds = output['attachments']['allfolds']
        folds_name = ['charliehebdo', 'germanwings' ,
                      'ferguson', 'ottawashooting', 'sydneysiege', 'bostonbombings']
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
            print("P: {:0.3f}, F: {:0.3f}, R: {:0.3f}, acc: {:0.3f}".format(P, F, R, acc))
            print("")

## -balance: 2:1
## -equal: 1:1

# output_file = os.path.join('..','..', 'output', 'pheme5-aug-shuffle-twitter1516-2', 'output_pheme5-aug-shuffle-twitter1516.pkl')
# output_file = os.path.join('..','..', 'output', 'recollect', 'output', 'pheme', 'output_pheme.pkl') # 0.5230
# output_file = os.path.join('..','..', 'output', 'output-asonam', 'pheme5-aug-boston-fergusoncorrect', 'output_pheme5-aug-boston-correctferguson.pkl')
# output_file = os.path.join('..','..', 'output', 'output-asonam', 'pheme5-30', 'output_pheme5.pkl')
# output_file = os.path.join('..','..', 'output', 'out-dated', 'pheme5top25+trial20', 'outputpheme5-context-top25-trial20-2.pkl') # 0.5285 --> asonam initial

# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306', 'output_aug-balance(4400124).pkl') # 0.5317
# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306', 'output_aug-balance(4401441).pkl') # 0.5174
# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306', 'output_aug-iter50(4392576).pkl') # 0.5177
# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306', 'output_aug-param.pkl') # 0.5479
# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306', 'output_aug.pkl') # 0.5101

# output_file = os.path.join('..','..', 'output', 'output-final', 'pheme-early', 'output_pheme.pkl') # 0.5456
# output_file = os.path.join('..','..', 'output', 'output-final',  'pheme-early-balance', 'output_pheme.pkl') # 0.6132
# output_file = os.path.join('..','..', 'output', 'output-final', 'pheme-early-balance', 'output_pheme-2(4407882).pkl') # 0.6098

output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306-boston', 'output_aug-boston-1-balance.pkl') # 0.5671
# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306-boston', 'output_aug-boston-1-iter50(4392577).pkl') # 0.4862
# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306-boston', 'output_aug-boston-1-param.pkl') #  0.3406

# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306-boston-2', 'output_aug-boston-2(4385416).pkl') # 0.5175
# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306-boston-2', 'output_aug-boston-2-balance.pkl') # 0.5362
# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306-boston-2', 'output_aug-boston-2-iter50(4392673).pkl') # 0.5328
# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-1306-boston-2', 'output_aug-boston-2-param.pkl') # 0.4518


# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-equal', 'output_aug-equal-20.pkl') # 0.6299
# output_file = os.path.join('..','..', 'output', 'output-final', 'pheme-equal', 'output_pheme-equal-20.pkl') #  0.6891
# output_file = os.path.join('..','..', 'output', 'output-final', 'aug-equal-boston-1', 'output_aug-equal-boston-1-20.pkl') # 0.6613

# output_file = os.path.join('..','..', 'output', 'output-test', 'pheme', 'output_pheme-test.pkl') # 0.5791


# output_file = os.path.join('..','..', 'output', 'output-final-first-try(original pheme as test)', 'aug-1306', 'output_aug.pkl') # 0.5101
# output_file = os.path.join('..','..', 'output', 'output-final-first-try(original pheme as test)', 'aug-1306-boston', 'output_aug-boston-1.pkl') # 0.5169
# print(output_file)
print(output_file)
load_output(output_file)

# save_params()
# check_input()