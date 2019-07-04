import os
from keras.models import Model, load_model

from keras.layers import Input, LSTM, Dense, Masking, Dropout
from keras import regularizers
import numpy as np
import pickle
from hyperopt import STATUS_OK
from keras.utils.np_utils import to_categorical
from branch2treelabels import branch2treelabels
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.preprocessing.sequence import pad_sequences

from copy import deepcopy
from itertools import compress
import time
import outer

global train_path
global holdout_path

train_path = outer.train_path
holdout_path = outer.holdout_path
test_path = outer.test_path
save_path = outer.save_path

def build_model(params, num_features):
    num_lstm_units = int(params['num_lstm_units'])
    num_lstm_layers = int(params['num_lstm_layers'])
    num_dense_layers = int(params['num_dense_layers'])
    num_dense_units = int(params['num_dense_units'])
    l2reg = params['l2reg']

    inputs_abc = Input(shape=(None, num_features))
    mask_abc = Masking(mask_value=0.)(inputs_abc)
    lstm_abc = LSTM(num_lstm_units, return_sequences=True)(mask_abc)
    for nl in range(num_lstm_layers - 1):
        lstm_abc2 = LSTM(num_lstm_units, return_sequences=True)(lstm_abc)
        lstm_abc = lstm_abc2

    lstm_c = LSTM(num_lstm_units, return_sequences=False)(lstm_abc)
    hidden1_c = Dense(num_dense_units)(lstm_c)
    for nl in range(num_dense_layers - 1):
        hidden2_c = Dense(num_dense_units)(hidden1_c)
        hidden1_c = hidden2_c
    dropout_c = Dropout(0.5)(hidden1_c)
    softmax_c = Dense(2, activation='softmax',
                      activity_regularizer=regularizers.l2(l2reg),
                      name='softmaxc')(dropout_c)
    model = Model(inputs=inputs_abc, outputs=[softmax_c])

    model.compile(optimizer='adam',
                  loss={
                        'softmaxc': 'binary_crossentropy'},
                  loss_weights={'softmaxc': 0.5},
                  metrics=['accuracy'])

    return model


def training(params, x_train,  y_trainC):
    num_epochs = params['num_epochs']
    batchsize = params['batchsize']

    # maskB = np.any(y_trainB, axis=1) + 0.00001

    num_features = np.shape(x_train)[2]
    model = build_model(params, num_features)

    model.fit(x_train, {"softmaxc": y_trainC},
              sample_weight={"softmaxc": None},
              epochs=num_epochs, batch_size=batchsize, verbose=0)

    return model

import glob
from pprint import pprint as pp

def objective_MTL2_detection_CV5(params):
    max_branch_len = 25

    print("Train ", train_path)
    print("Holdout ", holdout_path)
    print("")
    x_train = np.load(os.path.join(train_path, 'train_array.npy'))
    y_train = np.load(os.path.join(train_path,'rnr_labels.npy' ))
    x_train = pad_sequences(x_train, maxlen=max_branch_len,
                  dtype='float32', padding='post',
                  truncating='post', value=0.)
    x_train = np.asarray(x_train)
    print("X train shape ", x_train.shape)
    y_train = np.asarray(y_train)
    print("y train ", y_train.shape)
    y_train = to_categorical(y_train, num_classes=2)
    print("")
    x_holdout = np.load(os.path.join(holdout_path, 'train_array.npy'))
    print("X holdout shape ", x_holdout.shape)
    print("")
    y_holdout = np.load(os.path.join(holdout_path, 'rnr_labels.npy'))
    ids_holdout = np.load(os.path.join(holdout_path, 'ids.npy'))
    start =time.time()
    model = training(params, x_train, y_train)
    end=time.time()
    print("** Elapsed time ", end-start)
    print("")
    pred_probabilities = model.predict(x_holdout, verbose=0)

    Y_pred = np.argmax(pred_probabilities, axis=1)

    trees, tree_prediction, tree_label = branch2treelabels(ids_holdout,
                                                              y_holdout,
                                                              Y_pred)

    mactest_F = f1_score(tree_label,
                           tree_prediction,
                           average='binary')

    output = {
        'loss':  (1 - mactest_F),
        'Params': params,
        'status': STATUS_OK,
    }

    return output


def eval_MTL2_detection_CV(params, data, fname):
    max_branch_len = 25
    x_train = np.load(os.path.join(train_path, 'train_array.npy'))
    y_train = np.load(os.path.join(train_path, 'rnr_labels.npy'))
    x_train = pad_sequences(x_train, maxlen=max_branch_len,
                  dtype='float32', padding='post',
                  truncating='post', value=0.)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    y_train = to_categorical(y_train, num_classes=2)

    x_test = np.load(os.path.join(test_path, 'train_array.npy'))
    y_test = np.load(os.path.join(test_path, 'rnr_labels.npy'))
    ids_test = np.load(os.path.join(test_path, 'ids.npy'))


    model = training(params, x_train, y_train)
    model.save(os.path.join(save_path, 'model_' + fname + '.h5'))
    del model
    model = load_model(os.path.join(save_path, 'model_' + fname + '.h5'))

    pred_probabilities = model.predict(x_test, verbose=0)

    Y_pred = np.argmax(pred_probabilities, axis=1)

    trees, tree_prediction, tree_label = branch2treelabels(ids_test,
                                                              y_test,
                                                              Y_pred)

    perfold_result = {
                      'Task C': {'ID': trees, 'Label': tree_label,
                                 'Prediction': tree_prediction}
                      }


    Cmactest_P, Cmactest_R, Cmactest_F, _ = precision_recall_fscore_support(
        tree_label,
        tree_prediction,
        average='binary')

    Cacc = accuracy_score(tree_label, tree_prediction)

    output = {
        'Params': params,

        'TaskC': {
            'accuracy': Cacc,
            'Macro': {'Macro_Precision': Cmactest_P,
                      'Macro_Recall': Cmactest_R,
                      'Macro_F_score': Cmactest_F}
        },
        'attachments': {

            'Task C': {'ID': trees,
                       'Label': tree_label,
                       'Prediction': tree_prediction
                       },
            'allfolds': perfold_result

        }
    }
    print("-- output")

    directory = save_path

    print(directory)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'output_' + fname + '.pkl'), 'wb') as outfile:
        pickle.dump(output, outfile)

    return output

