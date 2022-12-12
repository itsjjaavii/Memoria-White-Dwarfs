# -*- coding: utf-8 -*-
from multiprocessing.dummy.connection import families
from pathlib import Path
import os
import pandas as pd
import numpy as np
import keras
import numpy as np
import statistics
import matplotlib.pyplot as plt
import joblib
import logging
import configparser
import argparse

# # Automatic spectral classification report generator.
#
# Version 0, written by Jose Rojas 10/2022.
#
# DESCRIPTION: Performs analysis on folder espectra, or using a dataframe that provides full or relative paths to all espectral files.
#
# REQUIREMENTS: This notebooks requires python libraries as well as the src code files from the Spectral Classification ML github project.

# As long as this script file is called from the "scripts" subdirectory, or any other
# subdirectory at the same level, the following lines define relative paths to project and current directory.

project_dir =  str(Path(__file__).resolve().parents[1])
current_dir =  str(Path(__file__).resolve().parents[0])

# The next lines allows to import modules from the src code.

import sys
sys.path.append(project_dir)
from src.visualization import visualize as vsz
from src.models import predict_model as pdm
from src.features import build_features as bfs

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


def load_spectrum_data(file_path: str):
    if file_path.endswith('fits'):
        wavelength, flux = bfs.extraction(file_path)
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

# Loads a calibration file wich contains the base wavelenght that was used for training.

base_wavelenght, _ = load_spectrum_data(os.path.join(project_dir, 'calibration_dat_file.dat'))
base_wavelenght = base_wavelenght[298:-600] # this is the wavelenght axis the model was trained on.

# Define a saving path for the analysis results. This includes both the csv summary, the png report files and the written summary.

save_path = os.path.join(project_dir, r'results\SDSS_dataset_results')
sdss_data_path = os.path.join(project_dir, r'data\raw\sdss_dat_files')

#### directory iteration  routine: finds all files named .dat and .fits inside a given directory. ###

def dir_input_iteration_routine(save_path, data_path, model, dom_rf_model, classes_of_interest='all', filter_by_domain_detector=False):
    counter=0
    logger = logging.getLogger(__name__)

    results_df = pd.DataFrame(columns=['filename', 'preds']) # for storing prediciton results into csv file.

    # making a variable having the index till which
    # data_path string has the top directory and a path separator
    src_prefix = len(data_path) + len(os.path.sep)

    # replicate folder structure in save_path 
    # and analyze all .dat and .fits files in data_path

    for root, dirnames, filenames in os.walk(data_path):

        # Recursively rebuild source directory tree in destination folder save_path
        for dir in dirnames:
            dirpath = os.path.join(save_path, root[src_prefix:], dir)
            try:
                os.mkdir(dirpath)
            except:
                pass
    
        # Current folder relative to the source data path.
        current_folder = os.path.relpath(path=root, start=data_path)
        
        for file in filenames:
            if file.endswith(('.dat', '.fits')):
                w, f = load_spectrum_data(os.path.join(root, file))

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
                not_norm_flux = flux
                # normalization
                norm_magnitud = statistics.mean(flux[2050:2100])

                if norm_magnitud == 0:
                    logger.info('Foun all 0 file {}'.format(file))
                    continue
                else:
                    flux = np.divide(flux, norm_magnitud)

                # model prediction

                dom_pred = dom_rf_model.predict(flux.reshape(1, -1) )
                prediction = model.predict(flux.reshape(1,-1,1))
                preds = model.predict(flux.reshape(1,-1,1)).reshape((12,))
                predicted_class_str = star_classes[int(np.where(preds == max(preds))[0])]

                # deciding if a png file is generated or not, based on classes of interest and domain detector result, or disregarding these conditions.

                type_condition = (predicted_class_str in classes_of_interest) if isinstance(classes_of_interest, list) else True

                if (((bool(dom_pred) == True) or not(filter_by_domain_detector)) and type_condition):
                    predicted_class = prediction.argmax(1)
                    cam_wda = pdm.GradCAM(model, int(predicted_class))
                    heatmap = cam_wda.compute_heatmap(flux.reshape(1,-1,1))
                    vsz.png_report_generator(not_norm_flux, flux, base_wavelenght, model, heatmap, dom_pred, star_classes, filepath=os.path.join(root, file), png_save_path=os.path.join(save_path, current_folder, file[:-4].replace('.','_')))
                    logger.info('Found and interesting one! file {}, type {}'.format(os.path.join(current_folder, file), predicted_class_str))
                    counter+=1

                results_df = pd.concat([results_df, pd.DataFrame.from_records([{'filename' : current_folder+file, 'preds' : predicted_class_str}])], ignore_index=True)
            elif file.endswith(('.png')):
                os.remove(os.path.join(root, file))
                logger.info('removed {} file'.format(file))

    results_df.to_csv(os.path.join(save_path, 'results.csv') , index=False)
    print('analyzed {} elements'.format(counter))
    

    #### directory iteration  routine: finds all files named .dat and .fits inside a given directory. ###

def csv_input_iteration_routine(save_path, data_path, model, dom_rf_model, classes_of_interest='all', filter_by_domain_detector=False, relative_path=None, normalize='mean'):
    counter=0
    logger = logging.getLogger(__name__)

    # Read dataframe containing filenames
    df = pd.read_csv(data_path)
    filenames = df.loc[:,'filename']

    # Check if dataframe has a column containing a "classID" label
    if "classID" in df.columns:
        classID = df.loc[:,'classID']

    # Create dataframe for storing results
    if "classID" in df.columns:
        results_df = pd.DataFrame(columns=['filename', 'preds', 'classID']) # for storing prediciton results into csv file
    else:
        results_df = pd.DataFrame(columns=['filename', 'preds']) # for storing prediciton results into csv file.
        
    for class_index, file in enumerate(filenames):
        if file.endswith(('.dat', '.fits')):
            w, f = load_spectrum_data(os.path.join(relative_path, file) if relative_path != None else file)

            # Interpolation

            if (np.isnan(f).any()):
                nans, x= nan_helper(f)
                f[nans] = np.interp(x(nans), x(~nans), f[~nans])
                logger.info('file {} has Nan inputs on flux, interpolated and proceeded anyway.'.format(file))
            if (np.isnan(w).any()):
                nans, x= nan_helper(w)
                w[nans] = np.interp(x(nans), x(~nans), w[~nans])
                logger.info('file {} has Nan inputs on wavelenght, interpolated and proceeded anyway.'.format(file))
            if(np.isnan(w).all() or np.isnan(f).all()):
                logger.info('Found all nan file {}. Not analyzing.'.format(file))
                continue

            flux = np.interp(base_wavelenght, w, f)
            not_norm_flux = flux
            # normalization       
              
            flux_mean = statistics.mean(flux)
            if(np.isclose(flux_mean, 0, rtol=0, atol=1e-30, equal_nan=False)):
                logger.info('Found file with zero flux {}. Not analyzing.'.format(file))
                continue
            
            if normalize == 'mean':
                flux = np.divide(flux, flux_mean)
            elif normalize == 'norm_v1':
                flux = np.divide(flux, statistics.mean(flux[2050:2100])) # normalize around a fixed wavelenght vecinity.

            # model prediction

            dom_pred = dom_rf_model.predict(flux.reshape(1, -1) )
            prediction = model.predict(flux.reshape(1,-1,1))
            preds = model.predict(flux.reshape(1,-1,1)).reshape((12,))
            predicted_class_str = star_classes[int(np.where(preds == max(preds))[0])]

            # deciding if a png file is generated or not, based on classes of interest and domain detector result, or disregarding these conditions.

            type_condition = (predicted_class_str in classes_of_interest) if isinstance(classes_of_interest, list) else True

            if (((bool(dom_pred) == True) or not(filter_by_domain_detector)) and type_condition):
                predicted_class = prediction.argmax(1)
                cam_wda = pdm.GradCAM(model, int(predicted_class))
                heatmap = cam_wda.compute_heatmap(flux.reshape(1,-1,1))

                # make folder if necessary
                os.makedirs(os.path.join(save_path, os.path.dirname(file)), exist_ok=True)

                vsz.png_report_generator(not_norm_flux, flux, base_wavelenght, model, heatmap, dom_pred, star_classes, filepath=file, png_save_path=os.path.join(save_path, file[:-4].replace('.','_')))
                if "classID" in df.columns:
                    logger.info('Found and interesting one! file: {}, type: {}, original prediciton: {}'.format(file, predicted_class_str, classID[class_index]))
                else:
                    logger.info('Found and interesting one! file: {}, type: {}'.format(file, predicted_class_str))
                counter+=1

            if "classID" in df.columns:
                results_df = pd.concat([results_df, pd.DataFrame.from_records([{'filename' : file, 'wd_prediction' : predicted_class_str, 'dom_prediciton' : bool(dom_pred),'classID' : classID[class_index]}])], ignore_index=True)
            else:
                results_df = pd.concat([results_df, pd.DataFrame.from_records([{'filename' : file, 'wd_prediction' : predicted_class_str, 'dom_prediciton' : bool(dom_pred)}])], ignore_index=True)
            results_df["dom_prediciton"]=results_df["dom_prediciton"].astype(bool)
    results_df.to_csv(os.path.join(save_path, 'results.csv') , index=False)
    print('analyzed {} elements'.format(counter))


def main():
    logger = logging.getLogger(__name__)
    logger.info('Starting basic spectrum processing.')

    # script arg is path to config file

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config_file", type=str)
    args = parser.parse_args()
    cf_path = args.config_file
    
    # Load configuration file attributes.
    configParser = configparser.RawConfigParser()   
    configFilePath = os.path.join(project_dir, cf_path)
    configParser.read(configFilePath)

    class_model_path = configParser.get('models', 'class_model')
    dom_rf_model_path = configParser.get('models', 'dom_model')
    data_path = configParser.get('paths', 'data_path')
    save_path = configParser.get('paths', 'save_path')

    try:
        relative_path = configParser.get('paths', 'relative_path')
    except:
        pass

    # Load desired model for prediction and class map activation

    model = keras.models.load_model(os.path.join(project_dir, class_model_path))
    dom_rf_model = joblib.load(os.path.join(project_dir, dom_rf_model_path))

    # Call data iteration functions.
    
    if os.path.isdir(data_path):
        dir_input_iteration_routine(save_path, data_path, model, dom_rf_model)
    elif data_path.endswith('.csv'):
        csv_input_iteration_routine(save_path, data_path, model, dom_rf_model, relative_path=relative_path)
    else:
        logger.info("Can't understant input format, it's not a csv nor a directory.")

# Name guarding for executing file a a script.

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, filename="scan_log_file.log", filemode='a', force=True)
    logger = logging.getLogger(__name__)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Calling main() function.')
    main()