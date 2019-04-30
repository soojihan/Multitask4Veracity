import pickle
file = '/Users/suzie/Desktop/PhD/Projects/Multitask4Veracity/output/out-dated/aug9000top25+trial50/\
bestparams_MTL2_detection_augmented-9000-context-top25.txt'

# with open(file, 'rb') as f:
#     d = pickle.load(f)
#
# print(d.keys())


# params = {'batchsize': 32, 'l2reg': 0.001, 'learn_rate': 0.0001, 'num_dense_layers': 2, 'num_dense_units': 400, 'num_epochs': 50, 'num_lstm_layers': 2, 'num_lstm_units': 200}
# print(type(params))
#
# import os
# file = '/Users/suzie/Desktop/PhD/Projects/Multitask4Veracity/output/output-asonam/pheme5_params.txt'
# with open(file, 'wb') as f:
#     pickle.dump(params, f)

# import numpy as np
# import os
# data = os.path.join('..', '..', 'data', 'saved_data_twitter1516/twitter15/rnr_labels.npy')
# x = np.load(data)
# print(x)