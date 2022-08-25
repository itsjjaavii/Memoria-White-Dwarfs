# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import os
import pandas as pd
import numpy as np
from tensorflow import keras
import pickle
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

def sort_ohe(ohe_dict):
    sorted_keys = []

    for k in range(12):
        key = list(12 * '0')
        key[k] = '1'
        np.asarray(key, dtype=float)
        for id, ohe_id in ohe_dict.items():
            if (ohe_id == np.asarray(key, dtype=float)).all():
                sorted_keys.append(id)
    
    return sorted_keys

def ohe_decoder(class_series, numpy_labels):
    list_of_classes = list(class_series.unique())
    ohe_dict = dict()

    while list_of_classes:
        for index, elem in class_series.iteritems():
            if elem in list_of_classes:
                ohe_dict[elem] = numpy_labels[index]
                list_of_classes.remove(elem)

    return ohe_dict

def main():
    """ Runs data processing scripts to turn processed data into
        training, test and validation set, as well as bulding
        the features from the .dat spectrum files.
    """
    logger = logging.getLogger(__name__)
    logger.info('Preparing training, test and validation sets')

    # Current Project directory
    project_dir = Path(__file__).resolve().parents[2]
    
    # Load validation set 

    valid_spectrum_matrix = np.load(os.path.join(project_dir,'data\processed\lpv1_valid_spectrum_matrix.npy'))
    valid_label_matrix = np.load(os.path.join(project_dir,'data\processed\lpv1_valid_label_matrix.npy'))

    # Get one hot encoding from one set, in this case, the training set.

    train_csv_path = os.path.join(project_dir, r'data\processed\train_set.csv')
    train_set = pd.read_csv (train_csv_path)
    train_label_matrix = np.load(os.path.join(project_dir,'data\processed\lpv1_train_label_matrix.npy'))

    ohe_dict = ohe_decoder(train_set['classID'], train_label_matrix)

    # Load keras saved models

    path_to_cnn = os.path.join(project_dir, r'models\CNN_lpv1')
    cnn_model = keras.models.load_model(path_to_cnn)
    path_to_cnn = os.path.join(project_dir, r'models\ANN_lpv1')
    ann_model = keras.models.load_model(path_to_cnn)

    history_CNN = pickle.load(open(os.path.join(project_dir, 'models\CNN_lpv1_trainHistory'), "rb"))
    history_ANN = pickle.load(open(os.path.join(project_dir, 'models\ANN_lpv1_trainHistory'), "rb"))

    # Visualize confusion matrix


    valid_sample_output = cnn_model.predict(valid_spectrum_matrix, batch_size=60)
    predict_class = np.argmax(valid_sample_output, axis=1)
    actual_class = np.argmax(valid_label_matrix, axis=1)
    cf_matrix = confusion_matrix(actual_class, predict_class)

    star_class = sort_ohe(ohe_dict)

    sns.set(rc = {'figure.figsize':(15,11)})
    ax = sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Star Class')
    ax.set_ylabel('Actual Star Category ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(star_class)
    ax.yaxis.set_ticklabels(star_class)

    ## Display the visualization of the Confusion Matrix.
    plt.show()

# Name guarding for executing file a a script.

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()