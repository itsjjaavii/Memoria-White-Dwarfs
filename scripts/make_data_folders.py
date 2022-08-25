import os
from pathlib import Path


def try_make_folder(base_dir, folder):
    try:
        os.mkdir(os.path.join(base_dir, folder))
    except:
        print('Failed to create folder {}. Permissions maybe? or does some folder already exist?'.format(folder))

script_dir =  Path(os.path.dirname(os.path.realpath(__file__))).parent


folders = ['data', 'data/raw', 'data/external', 'data/interim', 'data/processed', 'data/raw/label_data', 'data/raw/sdss_dat_files']

for folder in folders:
    try_make_folder(script_dir, folder)
