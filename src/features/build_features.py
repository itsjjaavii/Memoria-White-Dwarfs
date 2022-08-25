# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib
import statistics
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel


import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.visualization import visualize as vsz



def make_stratified_sets(df, train_set_size: float=0.6, validation_test_split: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Make stratiffied train, test and validation sets, from a dataframe. 

    Parameters
    ----------
    :param df: Dataframe to split.
    :param train_set_size: The size (0-1) of the train set, out of the total dataframe.
    :param validation_test_split: The size (0-1) of the test set, out of the remaining data after making the trian set. 
    The validation set is consists of whats left.

    Returns
    -------
    :returns: returns two numpy arrays, one for the spectrum data and one for the one hot encoded label data, 
    and the filename of spectrums with nan data which will be removed from the datasets.
    """ 
    split = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_set_size), random_state=42)
    for train_index, test_index in split.split(df, df["classID"]):
        train_set = df.loc[train_index]
        test_valid_set = df.loc[test_index]


    split2 = StratifiedShuffleSplit(n_splits=1, test_size=validation_test_split, random_state=42)
    for test_index, valid_index in split2.split(test_valid_set, test_valid_set["classID"]):
        test_set = test_valid_set.iloc[test_index]
        valid_set = test_valid_set.iloc[valid_index]
    
    return train_set, test_set, valid_set

# We want to load a matrix from the training data, given the data size this wil be acceptable.

def data_processing_and_loading(sdss_data_path, block_df):
    """
    The following function takes a dataframe consisting of two columns: filename of .dat
    file and class ID, and returns a numpy array consisting of the spectrum data matrix, 
    processing the dataframe from top to bottom, so the numpy array elements are added in this order.

    This functions also does some flux data pre-processing, trimming the edges of the signal 
    and normalizing flux magnitud to 1 around 5847 [Angstrom].

    The dat files are assumed to share a common wavelenght vector. 

    Parameters
    ----------
    :sdss_data_path: path to folders where each subfolder contains .dat files.
    :block_df: dataframe with two columns, filenames and class id.
    :stars_enconding: dictionary with one hot encoding. 

    Returns
    -------
    :returns: returns two numpy arrays, one for the spectrum data and one for the one hot encoded label data, 
    and the filename of spectrums with nan data which will be removed from the datasets.
    """ 
    to_remove = list()
    spectrum_matrix = np.empty((0,4200), float)

    for _, row in block_df.iterrows():
        file_path = os.path.join(sdss_data_path, row['filename'])
        sed = np.loadtxt(file_path, unpack = True)
        wavelength = sed[0,48:-400]
        flux = sed[1,48:-400]
        norm_magnitud = statistics.mean(flux[2050:2100])
        if(np.isnan(flux).any() or np.isnan(wavelength).any() or norm_magnitud == 0):
            # print('Nan data found for file {}'.format(file_path))
            to_remove.append(row['filename'])
        else:
            flux = np.divide(flux, statistics.mean(flux[2050:2100]))
            spectrum_matrix = np.append(spectrum_matrix, np.transpose(flux[:, None]), axis=0)
    for element in to_remove:
        block_df.drop(block_df.index[block_df['filename'] == element], inplace = True)
    return spectrum_matrix

def basic_spectrum_pipeline(prefix, csv_relative_path):
    """ 
    This function takes,

    Parameters
    ----------
    :param prefix:
    :param csv_relative_path: Path of the matched csv file dataset w 

    Returns
    -------
    :returns: returns two numpy arrays, one for the spectrum data and one for the one hot encoded label data, 
    and the filename of spectrums with nan data which will be removed from the datasets.
    """ 
    logger = logging.getLogger(__name__)
    logger.info('Preparing training, test and validation sets')

    # Current Project directory
    project_dir = Path(__file__).resolve().parents[2]

    csv_path = os.path.join(project_dir, csv_relative_path)
    df = pd.read_csv (csv_path)

    # Make stratified training, test and validation sets

    train_set, test_set, valid_set = make_stratified_sets(df, train_set_size=0.6, validation_test_split=0.5)
    logger.info('Stratified training, test and validation sets done')

    # Creating spectrum numpy arrays from train and test sets
    # Spectrums with NaN values will be removed from the imput dataframes (e.g, test_set dataframe)

    sdss_data_path = os.path.join(project_dir, r'data\raw\sdss_dat_files')
    # print(test_set.shape)
    logger.info('Processing test set spectrum data')
    test_spectrum_matrix = data_processing_and_loading(sdss_data_path, test_set)
    # print(test_set.shape)
    # print(train_set.shape)
    logger.info('Processing training set spectrum data')
    train_spectrum_matrix = data_processing_and_loading(sdss_data_path, train_set)
    # print(train_set.shape)
    logger.info('Processing validation set spectrum data')
    valid_spectrum_matrix = data_processing_and_loading(sdss_data_path, valid_set)

    # One hot encoding categorical data

    ohe = OneHotEncoder(sparse=False)
    train_ohe_labels = ohe.fit_transform(train_set[['classID']])
    test_ohe_labels = ohe.fit_transform(test_set[['classID']])
    valid_ohe_labels = ohe.fit_transform(valid_set[['classID']])

     # Saving spectrums to processes data

    save_folder = os.path.join(project_dir, r'data\processed')
    np.save(os.path.join(save_folder, prefix+'_train_spectrum_matrix.npy'), train_spectrum_matrix)
    np.save(os.path.join(save_folder, prefix+'_test_spectrum_matrix.npy'), test_spectrum_matrix)
    np.save(os.path.join(save_folder, prefix+'_valid_spectrum_matrix.npy'), valid_spectrum_matrix)
    np.save(os.path.join(save_folder, prefix+'_train_label_matrix.npy'), train_ohe_labels)
    np.save(os.path.join(save_folder, prefix+'_test_label_matrix.npy'), test_ohe_labels)
    np.save(os.path.join(save_folder, prefix+'_valid_label_matrix.npy'), valid_ohe_labels)

    file_path = os.path.join(save_folder, r'train_set.csv')
    train_set.to_csv(file_path, index=False)
    file_path = os.path.join(save_folder, r'test_set.csv')
    test_set.to_csv(file_path, index=False)
    file_path = os.path.join(save_folder, r'valid_set.csv')
    valid_set.to_csv(file_path, index=False)

    # save one hot encoder
    # joblib.dump(ohe, os.path.join(save_folder, 'lpv1_ohe.pkl'))

    logger.info('Processed spectrum data and one hot encoded data saved to data/processed')

#----------------------------------------------------------#
#---- External data (Boris' data) processing functions ----#
#----------------------------------------------------------#

def zero_pad_str(final_len, my_str):
    """
    Appends zeros to the left side of str until reaching desired lenght
    """
    return (final_len - len(my_str)) * '0' + my_str if final_len > len(my_str) else my_str

def extraction(fitfile):
    hdu  = fits.open(fitfile)
    data = hdu['COADD'].data
    w = 10**data['loglam']
    f = data['flux']*1e-17
    return w,f

def make_external_datasets():
    # Loading crossmatch dataframe and make stratiffied training, test and validation sets.

    project_dir = Path(__file__).resolve().parents[2]
    crossmatch_path = os.path.join(project_dir, r'data\external\crossmatch')
    cm_info = pd.DataFrame()

    for filename in os.listdir(crossmatch_path):
        if not (filename.endswith('ipynb') or filename.endswith('da')):
            csv_path = os.path.join(crossmatch_path, filename)
            star_set = pd.read_csv(csv_path, usecols=['PLATEID','MJDID','FIBERID','THING_ID', 'source_id'])
            star_set['classID'] = filename.split('_')[-1]
            cm_info = cm_info.append(star_set)
    cm_info.reset_index(drop=True, inplace=True)
    cm_info = cm_info.astype({"PLATEID": 'int64', "MJDID": 'int64', "FIBERID" : 'int64', "THING_ID" : 'int64'})

    ext_train_set, ext_test_set, ext_valid_set = make_stratified_sets(cm_info, train_set_size=0.6, validation_test_split=0.5)

    return ext_train_set, ext_test_set, ext_valid_set

def ext_data_processing_and_loading(ext_data_path, block_df, base_wavelenght, ohe_dict, ext_data_dict):
    """
    The following function takes a dataframe consisting of two columns: filename of .dat
    file and class ID, and returns numpy arrays consisting of the spectrum data matrix.

    This functions also does some flux data pre-processing, trimming the edges of the signal 
    and normalizing flux magnitud to 1 around 5847 [A] 

    Parameters
    ----------
    :sdss_data_path: path to folders where each subfolder contains .dat files.
    :block_df: dataframe with two columns, filenames and class id.
    :stars_enconding: dictionary with one hot encoding. 

    Returns
    -------
    :return: returns two numpy arrays, one for the spectrum data and one for the one hot encoded label data, 
    and the filename of spectrums with nan data which will be removed from the datasets.
    """ 
    to_remove = list()
    spectrum_matrix = np.empty((0,4200), float)
    label_ohe_matrix = np.empty((0,12), float)

    for _, row in block_df.iterrows(): 
        relative_path = 'boss' + row['classID'] + '\\' + 'spec-' + zero_pad_str(5, str(row['PLATEID'])) + '-' + str(row['MJDID']) + '-' + zero_pad_str(4, str(row['FIBERID'])) + '.fits'
        fits_path = os.path.join(ext_data_path, relative_path)
        w,f = extraction(fits_path)
        # w, smoothed = convolution(w, f)

        ########
        flux = np.interp(base_wavelenght, w, f)
        wavelength = base_wavelenght
        norm_magnitud = statistics.mean(flux[2050:2100])
        if(np.isnan(flux).any() or np.isnan(wavelength).any() or norm_magnitud == 0):
            # print('Nan data found for file {}'.format(file_path))
            to_remove.append(row['THING_ID'])
        else:
            flux = np.divide(flux, statistics.mean(flux[2050:2100]))
            spectrum_matrix = np.append(spectrum_matrix, np.transpose(flux[:, None]), axis=0)
            label_ohe_matrix = np.append(label_ohe_matrix, ohe_dict[ext_data_dict[row['classID']]].reshape(1, 12), axis=0)
    for element in to_remove:
        block_df.drop(block_df.index[block_df['THING_ID'] == element], inplace = True)
    
    return spectrum_matrix, label_ohe_matrix

def external_spectrum_processingv1(prefix: str='ext') -> None:
    project_dir = Path(__file__).resolve().parents[2]

    # Decode Ohe used for already processed numpy train set
    train_csv_path = os.path.join(project_dir, r'data\processed\train_set.csv')
    train_set = pd.read_csv (train_csv_path)

    # For now, predefined label matrix, need ti change in future.

    train_label_matrix = np.load(os.path.join(project_dir, r'data\processed\lpv1_train_label_matrix.npy'))

    train_csv_path = os.path.join(project_dir, r'data\processed\train_set.csv')
    train_set = pd.read_csv (train_csv_path)
    ohe_dict = vsz.ohe_decoder(train_set['classID'], train_label_matrix)

    ext_train_set, ext_test_set, ext_valid_set = make_external_datasets()
    
    # Calibration file for base wavelenght, need to improve selection of file in the future.

    dat_path = os.path.join(project_dir, r'data\raw\sdss_dat_files\cb_uvex3\SDSSJ133448.14+374547.9_15299-59369-0406_4602877123.dat')
    sed = np.loadtxt(dat_path, unpack = True)
    base_wavelenght = sed[0,48:-400]

    # Call external pre processing function
    ext_data_dict = { 'cv' : 'CV',  'da' : 'WDA', 'da+ms' : 'WD+MS', 'db+ms' : 'WD+MS', 'db' : 'WDB', 'dq' : 'WDQ', 'dz' : 'WDZ', 'wd' : 'WD'}
    ext_data_path = os.path.join(project_dir, r'data\external\fits_spectrums')
    ext_train_spectrum_matrix, ext_train_label_matrix = ext_data_processing_and_loading(ext_data_path, ext_train_set, base_wavelenght, ohe_dict, ext_data_dict)
    ext_valid_spectrum_matrix, ext_valid_label_matrix = ext_data_processing_and_loading(ext_data_path, ext_valid_set, base_wavelenght, ohe_dict, ext_data_dict)
    ext_test_spectrum_matrix, ext_test_label_matrix = ext_data_processing_and_loading(ext_data_path, ext_test_set, base_wavelenght, ohe_dict, ext_data_dict)

     # Saving spectrums to processes data

    save_folder = os.path.join(project_dir, r'data\processed')
    np.save(os.path.join(save_folder, prefix+'_train_spectrum_matrix.npy'), ext_train_spectrum_matrix)
    np.save(os.path.join(save_folder, prefix+'_test_spectrum_matrix.npy'), ext_test_spectrum_matrix)
    np.save(os.path.join(save_folder, prefix+'_valid_spectrum_matrix.npy'), ext_valid_spectrum_matrix)
    np.save(os.path.join(save_folder, prefix+'_train_label_matrix.npy'), ext_train_label_matrix)
    np.save(os.path.join(save_folder, prefix+'_test_label_matrix.npy'), ext_test_label_matrix)
    np.save(os.path.join(save_folder, prefix+'_valid_label_matrix.npy'), ext_valid_label_matrix)

    file_path = os.path.join(save_folder, prefix + '_train_set.csv')
    ext_train_set.to_csv(file_path, index=False)
    file_path = os.path.join(save_folder, prefix + '_test_set.csv')
    ext_test_set.to_csv(file_path, index=False)
    file_path = os.path.join(save_folder, prefix + '_valid_set.csv')
    ext_valid_set.to_csv(file_path, index=False)

def main():
    logger = logging.getLogger(__name__)
    logger.info('Starting basic spectrum processing.')
    #basic_spectrum_pipeline(prefix='lpv1', csv_relative_path=r'data\processed\lpv1_match_labels.csv')
    external_spectrum_processingv1()


# Name guarding for executing file a a script.

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()