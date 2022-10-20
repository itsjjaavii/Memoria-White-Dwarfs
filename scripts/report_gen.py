# -*- coding: utf-8 -*-
from pathlib import Path
import os
from xml import dom
import pandas as pd
import numpy as np
import keras
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel
import statistics
import matplotlib.pyplot as plt
import joblib
import logging


# The next lines allows to import modules from the src code.

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.visualization import visualize as vsz
from src.models import predict_model as pdm


# Define relative paths to project and current directory.

project_dir =  str(Path(__file__).resolve().parents[1])
current_dir =  str(Path(__file__).resolve().parents[0])

# Load desired model for prediction and class map activation

model = keras.models.load_model(os.path.join(project_dir, r'models\best_model_originalUpdate.h5'))
dom_rf_model = joblib.load(os.path.join(project_dir, r'models\rf_best_model.joblib'))

# Load ohe dictionary used for training, currently done manually. 

ohe_dict = {'WDA': np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
            'WDZ': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
            'WDB': np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
            'WDO': np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]),
            'WD+MS': np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
            'WD': np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
            'sdX': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
            'WDH': np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
            'WDELM': np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]),
            'WDC': np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
            'CV': np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
            'WDQ': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])}

star_classes = vsz.sort_ohe(ohe_dict)

def extraction(fits_file):
    hdu  = fits.open(fits_file)
    data = hdu['COADD'].data
    w = 10**data['loglam']
    f = data['flux']*1e-17
    return w,f

def load_spectrum_data(file_path: str):
    if file_path.endswith('fits'):
        wavelength, flux = extraction(file_path)
    elif file_path.endswith('.dat'):
        sed = np.loadtxt(file_path, unpack = True)
        wavelength = sed[0,:]
        flux = sed[1,:]
    else:
        print("what format is that?")
        return None
    
    return wavelength, flux

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

base_wavelenght, _ = load_spectrum_data(os.path.join(project_dir, 'calibration_dat_file.dat'))
base_wavelenght = base_wavelenght[298:-600] # this is the wavelenght axis the model was trained on.


save_path = os.path.join(project_dir, r'results\SDSS_dataset_results')
sdss_data_path = os.path.join(project_dir, r'data\raw\sdss_dat_files')

#### filepath iteration ###

results_df = pd.DataFrame(columns=['folder', 'filename', 'preds'])

counter = 0
logger = logging.getLogger(__name__)

scanned_counter = 0

for folder in os.listdir(sdss_data_path):
    try:
        os.mkdir(os.path.join(save_path, folder))
    except:
        print("Failed to create folder. Perhaps the folder already exists, or there's a permissions issue.")

    folder_path = os.path.join(sdss_data_path, folder)

    logger.info('Going trough elements from {}' .format(folder_path))

    for file in os.listdir(folder_path):
        scanned_counter += 1
        if file.endswith('.dat'):
            w, f = load_spectrum_data(os.path.join(folder_path, file))
            # Interpolation

            if (np.isnan(f).any()):
                nans, x= nan_helper(f)
                f[nans] = np.interp(x(nans), x(~nans), f[~nans])
            if (np.isnan(w).any()):
                nans, x= nan_helper(w)
                w[nans] = np.interp(x(nans), x(~nans), w[~nans])
            if(np.isnan(w).all() or np.isnan(f).all()):
                logger.info('Foun all nan file {}'.format(file))
                continue

            flux = np.interp(base_wavelenght, w, f)
            # normalization
            norm_magnitud = statistics.mean(flux[2050:2100])

            if statistics.mean(flux[2050:2100]) == 0:
                logger.info('Foun all 0 file {}'.format(file))
                continue
            else:
                flux = np.divide(flux, statistics.mean(flux[2050:2100]))

            # model prediction
            dom_pred = dom_rf_model.predict(flux.reshape(1, -1) )
            prediction = model.predict(flux.reshape(1,-1,1))
            preds = model.predict(flux.reshape(1,-1,1)).reshape((12,))
            predicted_class_str = star_classes[int(np.where(preds == max(preds))[0])]

            # decide if generate a full report or not.

            if ((bool(dom_pred) == True) or (predicted_class_str in ['WDH', 'WDZ', 'WDQ'])):
                predicted_class = prediction.argmax(1)
                cam_wda = pdm.GradCAM(model, int(predicted_class))
                heatmap = cam_wda.compute_heatmap(flux.reshape(1,-1,1))
                vsz.png_report_generator(flux, base_wavelenght, model, heatmap, dom_pred, star_classes, filepath=os.path.join(folder, file), save_path=save_path)
                counter += 1
                logger.info('Foun and interesting one! file {}, type {}, object number {}'.format(file, predicted_class_str, counter))

            results_df = results_df.append({'folder' : folder, 'filename' : file, 'preds' : predicted_class_str}, ignore_index=True)


results_df.to_csv(os.path.join(save_path, 'results.csv') , index=False)
print('scanned {} elements in total'.format(scanned_counter))