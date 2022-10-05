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
import statistics
import joblib

def preprocessing(file_path :str, sdss_data_path:str, first_bin=48, last_bin=-400):

    full_path = os.path.join(sdss_data_path, file_path)
    print(full_path)
    sed = np.loadtxt(full_path, unpack = True)
    wavelength = sed[0,first_bin:last_bin]
    flux = sed[1,first_bin:last_bin]
    norm_magnitud = statistics.mean(flux[2050:2100])
    if(np.isnan(flux).any() or np.isnan(wavelength).any() or norm_magnitud == 0):
        print("can't process file ".format(file_path))
    else:
        flux = np.divide(flux, statistics.mean(flux[2050:2100]))
    
    return flux

class wd_predictor:
    def __init__(self, model_path, dom_path, dat_file_path):
        self.model_path = model_path
        self.dom_path = dom_path
        self.dat_file_path = dat_file_path
    
    def predict_class(self, filepath, get_cam_info=False):
        flux = preprocessing(file_path=filepath, sdss_data_path=self.dat_file_path, first_bin=298, last_bin=-600)
        model = keras.models.load_model(self.model_path)
        prediction = model.predict(flux.reshape(1,-1,1))
        prediction = prediction.reshape((12,))
        # Calculate Class Activation Map information if required.
        if get_cam_info == True:
            spectrum_output = model.output[:,3]
            last_conv_layer = model.get_layer('conv1d_3').output
            return prediction, last_conv_layer, spectrum_output
        else:
            return prediction
    
    def predict_dom(self, filepath):
        flux = preprocessing(file_path=filepath, sdss_data_path=self.dat_file_path)
        model = joblib.load(self.dom_path)
        prediction = model.predict(flux.reshape(1,-1,1))
        prediction = prediction.reshape((12,))
        return prediction



def main():
    logger = logging.getLogger(__name__)
    logger.info('Preparing training, test and validation sets')

    # Current Project directory
    project_dir = Path(__file__).resolve().parents[2]
    
    train_spectrum_matrix = np.load(os.path.join(project_dir, 'data\processed\lpv1_train_spectrum_matrix.npy'))
    train_label_matrix = np.load(os.path.join(project_dir,'data\processed\lpv1_train_label_matrix.npy'))

    valid_spectrum_matrix = np.load(os.path.join(project_dir,'data\processed\lpv1_valid_spectrum_matrix.npy'))
    valid_label_matrix = np.load(os.path.join(project_dir,'data\processed\lpv1_valid_label_matrix.npy'))

    # Over sampling the imbalanced dataset

    
# Name guarding for executing file a a script.

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()