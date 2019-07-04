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


def load_output(output_file: str):
    """
    Read rumour detection results
    :return:
    """
    with open(output_file, 'rb') as f:
        output = pickle.load(f)
        f.close()
    print("Parameters")
    pp(output['Params'])
    print("")
    for metric, value in output['TaskC']['Macro'].items():
        print("{} : {:0.3f}".format(metric, value))
    print("Accuracy: {:0.3f}".format(output['TaskC']['accuracy']))



event = 'ottawashooting'
output_file = os.path.join('..','..', 'output', '2806', 'pheme', '{}'.format(event), 'output_ottawashooting-model_0.516.pkl')
print(output_file)
load_output(output_file)
