# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import os
from xmlrpc.client import Boolean
import pandas as pd



def load_labels(folder_path: str, columns: list) -> pd.DataFrame:
    """ Simple function to load all label csv files from some folder and load them into
    a dataframe containing the desired columns.

    Parameters
    ----------
    :param folder_path: Folder that contains the label csv files.
    :param columns: List of strings consisting of the desired column names to be added to the dataframe.

    Returns
    -------
    :returns: dataframe consisting of the raw labels.
    """
    raw_labels_df = pd.DataFrame()
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_path = os.path.join(folder_path, filename)
            df = pd.read_csv (csv_path, usecols=columns)
            raw_labels_df = raw_labels_df.append(df, ignore_index=True)
    return raw_labels_df

def clean_labels(raw_labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    A simple function to do basic cleaning on the dataframe of loaded labels, 
    this is fixing some wrong labels and replace the "no comment" over data quality by an "OK" comment.
    This changes are explored over the jupyter notebook analysis.

    Parameters
    ----------
    :param raw_labels_df: Dataframe of raw label information from the spectrocopic labeling campaign.
    
    Returns
    -------
    :returns: A filtered version of the dataframe
    """ 
    # Re-labels some apparently mislabeled items.

    raw_labels_df.loc[raw_labels_df['Classification'] == 'WELM', 'Classification'] = 'WDELM'
    raw_labels_df.loc[raw_labels_df['Classification'] == 'DA', 'Classification'] = 'WDA'

    # Replaces the data quality abscense of commentary with an 'OK'
    raw_labels_df['Data Quality'] = raw_labels_df['Data Quality'].fillna('OK')
    raw_labels_df.loc[raw_labels_df['Data Quality'] == ' ', 'Data Quality'] = 'OK'

    return raw_labels_df

def label_preprocess_v1(labels_df: pd.DataFrame, data_augmentation: Boolean=False) -> pd.DataFrame:
    """
    This function implements a specific policy for label selection. As an object, identified by it's target id,
    can have many evaluations (from the same measurement or from different measurements taken, for example, in diferent MJDs)
    we can define a policy to choose which will be considered the correct label.

    To assign a label to an object, this policy takes the mode of the labels among those which don't have any data quality issues,
    and assigns this mode as the true label of the object. This simple policy also considers the first observation encountered in
    the dataframe as the representative of the object, discarding the others.

    If the function is called whith the parameter data_augmentation=True, then this function returns all observations on the dataframe 
    that had the mode "true label", discarding both the observations with data quality issues and the mislabeled ones, as they may
    have some confusing features (but this may not be true, for example, the labeler may have simply been mistaken, so this data may
    still be useful, and is important to try different label processing policies.) 
    .

    Parameters
    ----------
    :param labels_df: Dataframe of cleaned labels (as delivered by the function clean_labels)
    :param data_augmentation: Set to True to perform basic data augmentation as decribed above.
    Returns
    -------
    :returns: A dataframe were each object has an unique label assign using the polivy described.
    """ 
    # Removing observations with data quality issues:

    labels_df.drop(labels_df[labels_df['Data Quality'] != 'OK'].index, inplace = True)

    # Selecting labels for each target id:

    parsed_df = pd.DataFrame()

    for object_id in labels_df['Target ID'].unique():
        mode_class = labels_df.loc[labels_df['Target ID'] == object_id]['Classification'].mode().to_list()[0]
        if data_augmentation:
            parsed_df = parsed_df.append(labels_df.loc[(labels_df['Classification'] == mode_class) & (labels_df['Target ID'] == object_id)])
        else:
            parsed_df = parsed_df.append(labels_df.loc[(labels_df['Classification'] == mode_class) & (labels_df['Target ID'] == object_id)].iloc[0])

    # For some reason my pd dataframe was loosing format, so I included this line.

    parsed_df = parsed_df.astype({"DB ID": 'int64', "MJD": 'int64', "Target ID" : 'int64'})

    # Remove non interesting classes, note that this is done after a label consensus is reached.

    parsed_df.drop(parsed_df.index[parsed_df['Classification'] == 'DUNNO'], inplace = True)
    parsed_df.drop(parsed_df.index[parsed_df['Classification'] == 'UNCLASS'], inplace = True)
    parsed_df.drop(parsed_df.index[parsed_df['Classification'] == 'STAR'], inplace = True)
    parsed_df.drop(parsed_df.index[parsed_df['Classification'] == 'EXGAL'], inplace = True)

    return parsed_df

def match_dat_files(label_df: pd.DataFrame, sdss_datfiles_path: str) -> pd.DataFrame:
    """
    This function takes an already pre-processed label dataframe (with the final labels to use for ML)
    and delivers the available dat files from a folder. It's assumed the sdss_datfiles_path contains a
    series of subfolders, and the dat files are inside this subfolders, and no level further will be scanned.

    Parameters
    ----------
    :param label_df: An already pre-processed label dataframe (with the final labels to use for ML)
    :param sdss_dat_files: The complete path to a folder, where each subfolder holds .dat files too match with the df info.
    
    Returns
    -------
    :returns: A dataframe with the spectrums filename information found amog the dat files.
    """ 

    # First we create a dictionary out of the files, with the relevant infoormation (MJD, Target id)

    folder_list = os.listdir(sdss_datfiles_path)
    file_dict = dict()
    for folder in folder_list:
        file_list = os.listdir(os.path.join(sdss_datfiles_path, folder))
        for filename in file_list:
            if filename.endswith('.dat'):
                file_dict[filename.split('-')[-2] + '_' + filename.split('_')[-1]] = os.path.join(folder, filename)

    spectrum_df = pd.DataFrame(columns=['filename', 'classID'])

    # Now we search the dictionary keys to populate the spectum dataframe.

    for _, row in label_df.iterrows():
        filekey = str(row['MJD']) + '_' + str(row['Target ID']) + '.dat'
        if filekey in file_dict:
            spectrum_df = spectrum_df.append({'filename' : file_dict[filekey], 'classID' : row['Classification']}, ignore_index=True)
    return spectrum_df

def simple_processing_pipeline(prefix: str='lpv1', data_augmentation: Boolean=False) -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Current Project directory
    project_dir = Path(__file__).resolve().parents[2]

    # Loading dataframe with desired columns:
    columns = ['File Name', 'MJD', 'Target ID', 'DB ID' ,'Classification', 'Data Quality']
    labeldata_path = os.path.join(project_dir, r'data\raw\label_data')
    logger.info('Label data Loaded')

    # Performing basic cleaning operations on the data and saving it to interim data folder.
    sod = load_labels(labeldata_path, columns)
    clean_labels_df = clean_labels(sod)
    interim_path = os.path.join(project_dir, r'data\interim\clean_labels.csv')
    clean_labels_df.to_csv(interim_path, index=False)
    logger.info('Label cleaning operations done')

    # Preprocessing interim data to make final label dataframe.

    logger.info('Preprocessing label data')
    final_label_df = label_preprocess_v1(clean_labels_df, data_augmentation)
    processed_label_path = os.path.join(project_dir, ('data\\processed\\'+ prefix +'_labels.csv'))
    final_label_df.to_csv(processed_label_path, index=False)
    logger.info('Label data saved to processed folder')

    # Matching desired labels to available dat files:
    logger.info('Matching processed labels to available .dat files')
    sdss_datfiles_path = os.path.join(project_dir,  r'data\raw\sdss_dat_files')
    spectrum_match_df = match_dat_files(final_label_df, sdss_datfiles_path)
    match_dat_path = os.path.join(project_dir, ('data\\processed\\'+prefix+'_match_labels.csv'))
    spectrum_match_df.to_csv(match_dat_path, index=False)

    # Finishing message

    msg = ('with basic data augmentation.' if data_augmentation else 'without basic data augmentation.')
    logger.info('Finished basic preprocessing ' + msg)

#----------------------------------------------------------#
#---- External data (Boris' data) processing functions ----#
#----------------------------------------------------------#


def main():
    logger = logging.getLogger(__name__)
    logger.info('Starting basic preprocessing without data augmentation.')
    
    # The prefix will be appended to the generated datasets names, saved in the processes folder.

    simple_processing_pipeline(prefix='lpv1', data_augmentation=False)
    logger.info('Starting basic preprocessing with data augmentation.')
    simple_processing_pipeline(prefix='lpv1da', data_augmentation=True)


# Name guarding for executing file a a script.

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
