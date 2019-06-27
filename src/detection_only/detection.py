import os
from keras.models import Model
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
from data_loader import *
# %%

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
    # path = 'saved_data/saved_data_MTL2_detection'
    # path = "/mnt/fastdata/acp16sh/Multitask4Veracity/data/recollect/pheme"
    # path = "/mnt/fastdata/acp16sh/Multitask4Veracity/data/recollect/aug-balance"
    # path = "/mnt/fastdata/acp16sh/Multitask4Veracity/data/final/pheme-early"
    path = "/mnt/fastdata/acp16sh/Multitask4Veracity/data/final/pheme-equal-balance"
    print(path)

    train = [
             'ferguson',
             'ottawashooting',
             'sydneysiege',
             ]

    # train = [
    #          'charliehebdo',
    #          'germanwings',
    #          'sydneysiege',
    #          ]
    #
    test = 'charliehebdo'  # 'germanwings-crash'
    # test = 'ferguson'  # 'germanwings-crash'

    max_branch_len = 25
    x_train = []

    yb_train = []
    yc_train = []
    print("Train ", train)
    print("Test ", test)
    for t in train:
        temp_x_train = np.load(os.path.join(path, t, 'train_array.npy'))

        temp_yc_train = np.load(os.path.join(path, t, 'rnr_labels.npy'))

        temp_x_train = pad_sequences(temp_x_train, maxlen=max_branch_len,
                                     dtype='float32', padding='post',
                                     truncating='post', value=0.)
        print("temp_x_train ", temp_x_train.shape)
        x_train.extend(temp_x_train)
        yc_train.extend(temp_yc_train)

    x_train = np.asarray(x_train)
    print("X train shape ", x_train.shape)
    yc_train = np.asarray(yc_train)
    print("y train ", yc_train.shape)
    yc_train = to_categorical(yc_train, num_classes=2)
    x_test = np.load(os.path.join(path, test, 'train_array.npy'))
    print("X test shape ", x_test.shape)
    yc_test = np.load(os.path.join(path, test, 'rnr_labels.npy'))
    ids_testBC = np.load(os.path.join(path, test, 'ids.npy'))
    start =time.time()
    model = training(params, x_train, yc_train)
    end=time.time()
    print("** Elapsed time ", end-start)
    print("")
    pred_probabilities_c = model.predict(x_test, verbose=0)

    Y_pred_c = np.argmax(pred_probabilities_c, axis=1)

    treesC, tree_predictionC, tree_labelC = branch2treelabels(ids_testBC,
                                                              yc_test,
                                                              Y_pred_c)

    mactest_F_c = f1_score(tree_labelC,
                           tree_predictionC,
                           average='binary')

    output = {
        'loss':  (1 - mactest_F_c),
        'Params': params,
        'status': STATUS_OK,
    }

    return output


def eval_MTL2_detection_CV(params, data, fname):
    # path = 'saved_data/saved_data_MTL2_detection'
    # path ="/mnt/fastdata/acp16sh/Multitask4Veracity/data/recollect/pheme"
    # path ="/mnt/fastdata/acp16sh/Multitask4Veracity/data/recollect/aug-balance"
    # path ="/mnt/fastdata/acp16sh/Multitask4Veracity/data/final/pheme-early"
    # path = "/mnt/fastdata/acp16sh/Multitask4Veracity/data/final/backup/pheme-early(imbalance)"
    path = "/mnt/fastdata/acp16sh/Multitask4Veracity/data/final/pheme-equal-balance"
    print("path ", path)
    folds = ['charliehebdo', 'germanwings', 'ferguson',
             'ottawashooting', 'sydneysiege']

    print("******* Folds ")
    print(folds)
    print("")
    allfolds = []

    cv_ids_c = []
    cv_prediction_c = []
    cv_label_c = []

    for number in range(len(folds)):

        test = folds[number]
        print("test ", test)
        train = deepcopy(folds)
        del train[number]

        max_branch_len = 25
        x_train = []
        yc_train = []

        for t in train:
            temp_x_train = np.load(os.path.join(path, t, 'train_array.npy'))
            temp_yc_train = np.load(os.path.join(path, t, 'rnr_labels.npy'))

            temp_x_train = pad_sequences(temp_x_train, maxlen=max_branch_len,
                                         dtype='float32', padding='post',
                                         truncating='post', value=0.)
            x_train.extend(temp_x_train)
            yc_train.extend(temp_yc_train)

        x_train = np.asarray(x_train)
        yc_train = np.asarray(yc_train)
        yc_train = to_categorical(yc_train, num_classes=2)

        x_test = np.load(os.path.join(test_path, test, 'train_array.npy'))
        yc_test = np.load(os.path.join(test_path, test, 'rnr_labels.npy'))
        ids_testBC = np.load(os.path.join(test_path, test, 'ids.npy'))

        model = training(params, x_train, yc_train)

        pred_probabilities_c = model.predict(x_test, verbose=0)

        Y_pred_c = np.argmax(pred_probabilities_c, axis=1)

        treesC, tree_predictionC, tree_labelC = branch2treelabels(ids_testBC,
                                                                  yc_test,
                                                                  Y_pred_c)

        perfold_result = {
                          'Task C': {'ID': treesC, 'Label': tree_labelC,
                                     'Prediction': tree_predictionC}
                          }
        # %%
        cv_ids_c.extend(treesC)
        cv_prediction_c.extend(tree_predictionC)
        cv_label_c.extend(tree_labelC)

        allfolds.append(perfold_result)

    Cmactest_P, Cmactest_R, Cmactest_F, _ = precision_recall_fscore_support(
        cv_label_c,
        cv_prediction_c,
        average='binary')
    Cmictest_P, Cmictest_R, Cmictest_F, _ = precision_recall_fscore_support(
        cv_label_c,
        cv_prediction_c,
        average='binary')
    Ctest_P, Ctest_R, Ctest_F, _ = precision_recall_fscore_support(
        cv_label_c,
        cv_prediction_c)
    Cacc = accuracy_score(cv_label_c, cv_prediction_c)



    output = {
        'Params': params,

        'TaskC': {
            'accuracy': Cacc,
            'Macro': {'Macro_Precision': Cmactest_P,
                      'Macro_Recall': Cmactest_R,
                      'Macro_F_score': Cmactest_F},
            'Micro': {'Micro_Precision': Cmictest_P,
                      'Micro_Recall': Cmictest_R,
                      'Micro_F_score': Cmictest_F},
            'Per_class': {'Pclass_Precision': Ctest_P,
                          'Pclass_Recall': Ctest_R,
                          'Pclass_F_score': Ctest_F}
        },
        'attachments': {

            'Task C': {'ID': cv_ids_c,
                       'Label': cv_label_c,
                       'Prediction': cv_prediction_c
                       },
            'allfolds': allfolds

        }
    }
    # directory = "/mnt/fastdata/acp16sh/Multitask4Veracity/output/pheme"
    directory = "/mnt/fastdata/acp16sh/Multitask4Veracity/output-final/pheme-equal"
    # directory = "/mnt/fastdata/acp16sh/Multitask4Veracity/output-test/pheme"

    print(directory)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'output_' + fname + '.pkl'), 'wb') as outfile:
        pickle.dump(output, outfile)

    return output

