"""
Run outer.py
python outer.py

outer.py has the following options:
python outer.py --model='mtl2stance' --data='RumEval' --search=True --ntrials=10 --params="output/bestparams.txt"

--model - which task to train, stance or veracity
--data - which dataset to use
--search  - boolean, controls whether parameter search should be performed
--ntrials - if --search is True then this controls how many different
            parameter combinations should be assessed
--params - specifies filepath to file with parameters if --search is false
-h, --help - explains the command line

If performing parameter search, then execution will take long time
depending on number of trials, size and number of layers in parameter space.
Use of GPU is highly recommended.
If running with default parametes then search won't be performed
"""
import pickle
# os.environ["THEANO_FLAGS"]="floatX=float32"
# if using theano then use flag to set floatX=float32
from optparse import OptionParser
#
from sys import platform
# from application import objective_MTL2_detection_CV9

# train_path = "/Users/suzie/Desktop/PhD/Projects/Multitask4Veracity/data/Multitask-saved/pheme/charliehebdo/train"
# heldout_path = "/Users/suzie/Desktop/PhD/Projects/Multitask4Veracity/data/Multitask-saved/pheme/charliehebdo/heldout"
# test_path = "/Users/suzie/Desktop/PhD/Projects/Multitask4Veracity/data/Multitask-saved/pheme/charliehebdo/test"

if platform == 'linux':
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
        '--save path', dest='save_path', default=False,
        help='path to save results')

    (options, args) = parser.parse_args()
    psearch = options.psearch
    ntrials = int(options.ntrials)
    data = options.data
    model = options.model
    fname = options.fname
    train_path = options.train_path
    heldout_path = options.heldout_path
    test_path = options.test_path
    save_path = options.save_path

elif platform == 'darwin':
    psearch = True
    ntrials = 1
    data = 'pheme5'
    model = 'mtl2detect'
    fname = 'baseline-test'
    train_path = "/Users/suzie/Desktop/PhD/Projects/Multitask4Veracity/data/Multitask-saved/pheme/charliehebdo/heldout"
    heldout_path = "/Users/suzie/Desktop/PhD/Projects/Multitask4Veracity/data/Multitask-saved/pheme/charliehebdo/heldout"
    test_path = "/Users/suzie/Desktop/PhD/Projects/Multitask4Veracity/data/Multitask-saved/pheme/charliehebdo/heldout"

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
                params_file = '/mnt/fastdata/acp16sh/Multitask4Veracity/output-final/backup/aug-1306/bestparams_aug-param.txt'
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