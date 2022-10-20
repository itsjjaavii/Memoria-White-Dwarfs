# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import os
import numpy as np
import keras
from tensorflow import keras
import statistics
import joblib
import cv2
import tensorflow as tf

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

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()


    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 3D output
            if len(layer.output_shape) == 3:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 3D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, flux, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 3D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the flux tensor to a float-32 data type, pass the
            # flux through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(flux, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (flux.shape[2], flux.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

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