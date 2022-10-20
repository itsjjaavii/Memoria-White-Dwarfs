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

def png_report_generator(flux, base_wavelenght, model, heatmap, dom_pred, star_classes, filepath, save_path):
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

    spax.set_title('Flux plot (normalized)')
    spax.plot(base_wavelenght, flux, color='black', linewidth=0.5)
    spax.get_tightbbox(renderer=fig.canvas.get_renderer())
    spax.set_ylabel('$F_\lambda$ (Normalized)', fontsize=12)

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

    plt.savefig(os.path.join(save_path, filepath[:-4].replace('.','_')), bbox_inches='tight')
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