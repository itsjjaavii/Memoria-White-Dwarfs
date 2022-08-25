# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import os
import numpy as np
import pickle
from imblearn.over_sampling import RandomOverSampler
import keras
from tensorflow import keras
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten


def main():
    """ Runs data processing scripts to turn processed data into
        training, test and validation set, as well as bulding
        the features from the .dat spectrum files.
    """
    logger = logging.getLogger(__name__)
    logger.info('Preparing training, test and validation sets')

    # Current Project directory
    project_dir = Path(__file__).resolve().parents[2]
    
    train_spectrum_matrix = np.load(os.path.join(project_dir, 'data\processed\lpv1_train_spectrum_matrix.npy'))
    train_label_matrix = np.load(os.path.join(project_dir,'data\processed\lpv1_train_label_matrix.npy'))

    valid_spectrum_matrix = np.load(os.path.join(project_dir,'data\processed\lpv1_valid_spectrum_matrix.npy'))
    valid_label_matrix = np.load(os.path.join(project_dir,'data\processed\lpv1_valid_label_matrix.npy'))

    # Over sampling the imbalanced dataset

    under_sampler = RandomOverSampler(random_state=40, sampling_strategy='minority')
    X_over, Y_over = under_sampler.fit_resample(train_spectrum_matrix, train_label_matrix)

    model = keras.Sequential([Dense(4200, input_dim = 4200, activation='relu', kernel_initializer='uniform'),
                    Dense(32, activation='relu', kernel_initializer='uniform'),
                    Dense(64, activation='relu', kernel_initializer='uniform'),
                    Dense(12, activation='softmax', kernel_initializer='uniform')])

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_over, Y_over, batch_size=32, epochs=50 , validation_data=(valid_spectrum_matrix, valid_label_matrix))
    model.save(os.path.join(project_dir, 'models\ANN_lpv1'))
    with open(os.path.join(project_dir, 'models\ANN_lpv1_trainHistory'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model = keras.Sequential([Conv1D(filters=128, kernel_size=4, activation='relu', strides=2, input_shape=(X_over.shape[1],1)),
                    MaxPooling1D(pool_size=2),
                    Conv1D(filters=64, kernel_size=4, activation='relu', strides=2),
                    MaxPooling1D(pool_size=2),
                    Conv1D(filters=32, kernel_size=4, activation='relu', strides=2),
                    MaxPooling1D(pool_size=2),
                    Conv1D(filters=16, kernel_size=4, activation='relu', strides=2),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(64, activation='relu', kernel_initializer='uniform'),
                    Dense(32, activation='relu', kernel_initializer='uniform'),
                    Dense(12, activation='softmax', kernel_initializer='uniform')])

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_over, Y_over, batch_size=32, epochs=40 , validation_data=(valid_spectrum_matrix, valid_label_matrix))
    model.save(os.path.join(project_dir, 'models\CNN_lpv1'))
    with open(os.path.join(project_dir, 'models\CNN_lpv1_trainHistory'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# Name guarding for executing file a a script.

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()