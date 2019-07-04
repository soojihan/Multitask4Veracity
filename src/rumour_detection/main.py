import pickle
# os.environ["THEANO_FLAGS"]="floatX=float32"
# if using theano then use flag to set floatX=float32
from optparse import OptionParser
#
from sys import platform
from datetime import datetime
import time

# from application import objective_MTL2_detection_CV9


parser = OptionParser()
parser.add_option(
    '--search', dest='psearch', default=False,
    help='Whether parameter search should be done: default=%default')
parser.add_option('--ntrials', dest='ntrials', default=10,
                  help='Number of trials: default=%default')
parser.add_option(
    '--model', dest='model', default='mtl2stance',
    help='Which model to use. Can be one of the following:  mtl2stance, mtl2detect, mtl3; default=%default')
parser.add_option(
    '--data', dest='data', default='RumEval',
    help='Which dataset to use: RumEval(train, dev, test) or PHEME5 or PHEME9 (leave one event out cross-validation): default=%default')
parser.add_option(
    '--fname', dest='fname', default=False,
    help='filename')
parser.add_option(
    '--train_path', dest='train_path', default=False,
    help='path to train file')
parser.add_option(
    '--heldout_path', dest='heldout_path', default=False,
    help='path to heldout file')
parser.add_option(
    '--test_path', dest='test_path', default=False,
    help='path to test file')
parser.add_option(
    '--save_path', dest='save_path', default=False,
    help='path to save results')
parser.add_option(
    '--params_file', dest='params_file', default=False,
    help='file path to parameters')

(options, args) = parser.parse_args()
psearch = options.psearch
ntrials = int(options.ntrials)
data = options.data
model = options.model
fname = options.fname
# fname = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
train_path = options.train_path
heldout_path = options.heldout_path
test_path = options.test_path
save_path = options.save_path
params_file = options.params_file


def main():
    from detection import eval_MTL2_detection_CV
    from detection import objective_MTL2_detection_CV5
    from parameter_search import parameter_search

    output = []
    print(data)

    if model == 'mtl2detect':
        if data == 'pheme5':
            if psearch:
                print("parameter search.....")
                params = parameter_search(ntrials,
                                          objective_MTL2_detection_CV5,
                                          fname
                                          )
            else:
                print(params_file)
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
            print(params)
            output = eval_MTL2_detection_CV(params, data,
                                            fname)

        else:
            print ("Check dataset name")


    else:
        print ('Check model name')

    return output
#%%

if __name__ == '__main__':
    output = main()