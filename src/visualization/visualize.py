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
from matplotlib.gridspec import GridSpec
import textwrap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
        for index, elem in class_series.items():
            if elem in list_of_classes:
                ohe_dict[elem] = numpy_labels[index]
                list_of_classes.remove(elem)

    return ohe_dict

# Inspection tool code

line_design = {
     
    'Balmer' : {
        'color' : 'g',
        'style' : 'solid'
    } ,
    'HeI' : {
        'color' : 'm',
        'style' : 'solid'
    } ,
    'HeII' : {
        'color' : 'm',
        'style' : 'dashed'
    },
    'OI' : {
        'color' : 'b',
        'style' : 'solid'
    } ,
    'Na I'  : {
        'color' : 'yellow',
        'style' : 'solid'
    } ,
    'Mg I'  : {
        'color' : 'brown',
        'style' : 'solid'
    } ,
    'Mg II' : {
        'color' : 'brown',
        'style' : 'dashed'
    } ,
    'Al I'  : {
        'color' : 'grey',
        'style' : 'solid'
    } ,
    'Si II' : {
        'color' : 'cyan',
        'style' : 'solid'
    } ,
    'Ca II' : {
        'color' : 'pink',
        'style' : 'solid'
    } ,
}

line_dicts = {

    'OI' :  {
    '1'   : 7771.94,
    '2'   : 7774.17,
    '3'   : 7775.39,
    },
    'Na I'  : {
    '1'   : 5889.95,
    '2'   : 5895.92,
    },
    'Mg I'  : {
    '1'   : 3829.36,
    '2'   : 3832.30,
    '3'   : 5167.32,
    '4'   : 5172.68,
    '5'   : 5183.6,
    },
    'Mg II' : {
    '1'   : 3838.29,
    '2'   : 4384.64,
    '3'   : 4390.56,
    '4'   : 4481.33,
    },
    'Al I'  : {
    '1'   : 3944.01,
    },
    'Si II' : {
    '1'   : 3853.66,
    '2'   : 3856.02,
    '3'   : 3862.6,
    '4'   : 4128.07,
    '5'   : 4130.89,
    '6'   : 5055.98
    },
    'Ca II' : {
    #'1'   : 3179.33,
    #'2'   : 3181.28,
    #'3'   : 3736.90,
    '4'   : 3933.66,
    '5'   : 3968.47,
    },


    'Balmer' : {
    'alpha'   : 6564.5377,
    'beta'    : 4861.3615,
    'gamma'   : 4340.462,
    'delta'   : 4101.74,
    },

    'HeI' : {
    #'a' : 3706.054,
    #'b' : 3733.921,
    #'c' : 3820.694,
    #'d' : 3868.596,
    'e' : 3889.752,
    'f' : 3965.852,
    #'g' : 4010.404,
    'h' : 4025.107,
    'i' : 4027.338,
    #'j' : 4121.982,
    'k' : 4144.929,
    #'l' : 4170.146,
    'm' : 4389.163,
    #'n' : 4438.795,
    'o' : 4472.755,
    'p' : 4714.519,
    'q' : 4923.304,
    'r' : 5017.079,
    's' : 5049.147,
    't' : 5877.329,
    'u' : 6679.994,
    'v' : 7067.124,
    'w' : 7283.357,
    #'x' : 3188.622,
    #'y' : 2945.961
    },

     'HeII' : {
    'a' : 4686.992,
    'b' : 5413.024,
    #'c' : 4860.677,
    'd' : 4542.863,
    #'e' : 4339.89,
    #'f' : 4201.014,
    #'g' : 4101.197,
    #'h' : 4026.738,
    #'i' : 1640.41,
    #'j' : 1215.13,
    #'k' : 1085.12,
    #'l' : 1025.44
    },
}

def png_report_generator(not_norm_flux, flux, base_wavelenght, model, heatmap, dom_pred, star_classes, filepath, png_save_path):

    fig = plt.figure(figsize=(20,13)) 

    gs = GridSpec(2, 3, figure=fig)
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    # spectrum, heatmap, activation ant text axes

    spax = fig.add_subplot(gs[0, 0:2])
    hmax = fig.add_subplot(gs[1, 0:2])
    acax = fig.add_subplot(gs[0, 2])
    tax = fig.add_subplot(gs[1, 2])

    # Spectrum plot

    # Vertical line plots 

    for element, elem_dict in line_dicts.items():
        for line_pos in elem_dict.values():
            spax.axvline(x=line_pos, ymin=0, ymax=1, color=line_design[element]['color'],  linestyle=line_design[element]['style'])

    spax.set_title('Flux plot (normalized)')
    spax.plot(base_wavelenght, not_norm_flux, color='black', linewidth=0.5)
    spax.get_tightbbox(renderer=fig.canvas.get_renderer())
    spax.set_ylabel('$F_\lambda$', fontsize=12)

    

    # Heatmap Plot

    hmax.set_title('Class Activation Map for predicted class.')
    sc = hmax.scatter(base_wavelenght, flux, c=heatmap, s=6, cmap = 'inferno')
    hmax.set_ylabel('$F_\lambda$ (Normalized)', fontsize=12)
    cbaxes = inset_axes(hmax, width="15%", height="3%", loc=1) 
    fig.colorbar(sc, cax=cbaxes, orientation='horizontal')

    ### Activation plot

    acax.set_title('Network output')

    preds = model.predict(flux.reshape(1,-1,1)).reshape((12,))

    ### Draw predictions 
    acax.hlines(y=star_classes, xmin=0, xmax=preds, color='#DDA0DD', alpha=0.2, linewidth=5)
    acax.set_xlabel('Class Score', fontsize=12, color = '#333F4B')
    acax.set_ylabel('')

    # add little dots
    acax.plot(preds, star_classes, "o", markersize=5, color='#A020F0', alpha=0.6)

    # add label for top 3 predictions

    try:
        for value in sorted(preds, reverse=True)[:3]:
            acax.text(value - 0.02, int(np.where(preds == value)[0][0]) - 0.5, "{:.2f}".format(value), fontsize=9, color = '#333F4B')
    except:
        pass

    acax.text(-0.1, 11.5, 'Class', fontsize=12, color = '#333F4B')

    # change the style of the axis spines
    acax.spines['top'].set_visible(False)
    acax.spines['right'].set_visible(False)
    acax.spines['left'].set_bounds((0, len(star_classes)-1))
    acax.set_xlim(-0.01,max(preds)+0.01)

    # add some space between the axis and the plot
    acax.spines['left'].set_position(('outward', 8))
    acax.spines['bottom'].set_position(('outward', 5))

    ### Text box axis

    value = """The domain detector thinks this spectrum {} a white dwarf. \nThe neural network predicts this is most likely a {} sub-type, with a score of {:.2f}. \nfilepath={}
    """.format('is' if bool(dom_pred) else "isn't", star_classes[int(np.where(preds == max(preds))[0])], max(preds), str(filepath))

    # Wrap this text.
    wrapper = textwrap.TextWrapper(width=55, replace_whitespace=False)
    
    word_list = []  
    for paragraph in value.split('\n'):
        word_list += wrapper.wrap(text=paragraph) + [' ']
    # Print each line.
    Delta = 0.05
    for (i, element) in enumerate(word_list):
        tax.text(-0.05, 0.9 - i * Delta, element , fontsize=13 ,  transform=tax.transAxes)

    tax.set_axis_off()

    spax.autoscale(tight=True)
    hmax.autoscale(tight=True)

    plt.savefig(png_save_path, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

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