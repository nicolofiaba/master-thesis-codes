## Training a U-Net autoencoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Reshape, concatenate, Dropout, Rescaling, LeakyReLU
from tensorflow.keras.models import Model
from astropy.io import fits
import matplotlib.pyplot as plt
from gelsa import visu
import matplotlib.image as mpimg
import glob
import os
import argparse

# ---- Parse command-line arguments ----
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
args = parser.parse_args()

# ---- GPU configuration ----
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"Using GPU {args.gpu}")
  except RuntimeError as e:
    print(e)

# ---------------------------------------------------------------------------------
hdu_rect = fits.open("/scratch/astro/nicolo.fiaba/training_sets/labels_RGS180_4.fits", memmap=True)
hdu_rect_masked = fits.open("/scratch/astro/nicolo.fiaba/training_sets/train_RGS180_4.fits", memmap=True)

img_data = np.array([hdu.data for hdu in hdu_rect[1:]])
img_data_masked = np.array([hdu.data for hdu in hdu_rect_masked[1:]])

# Setting a maximum flux threshold to 16000. All pixels brighter than that are set to 16000.
max_threshold = 16000
img_data[img_data > max_threshold] = max_threshold
img_data_masked[img_data_masked > max_threshold] = max_threshold

# Add noise
sigma = np.sqrt(800)
size = (len(img_data), 512, 512)

noise = np.random.normal(loc=0, scale=sigma, size=size)
img_noise = img_data + noise
img_noise = np.array(img_noise, dtype=np.float32)

img_noise_masked = img_data_masked + noise
img_noise_masked = np.array(img_noise_masked, dtype=np.float32)

# This preprocessing layer rescales the images to be in the (0, 1) range
x_max = np.nanmax(img_noise)
x_min = np.nanmin(img_noise)
rescale = Rescaling(1./(x_max - x_min), offset=-x_min/(x_max - x_min), input_shape=(512, 512, 1))

img_noise = rescale(img_noise)
img_noise_masked = rescale(img_noise_masked)

# Converting to numpy, otherwise train_test_split function won't work
img_noise = img_noise.numpy()
img_noise_masked = img_noise_masked.numpy()

# test set is 10% of the whole dataset
x_train, x_test, y_train, y_test = train_test_split(
    img_noise_masked, img_noise, test_size=0.1, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Set NaNs to 0 before feeding the U-Net
x_train = tf.where(tf.math.is_nan(x_train), 0., x_train)
x_val = tf.where(tf.math.is_nan(x_val), 0., x_val)
x_test = tf.where(tf.math.is_nan(x_test), 0., x_test)

image_size = (x_train.shape[1], x_train.shape[2])

print("Images in training set:", len(x_train), 
      "\nImages in validation set:", len(x_val), 
      "\nImages in test set:", len(x_test), 
      "\nImage size:", image_size)

# Creating Tensorflow datasets

BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE).shuffle(100)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
#------------------------------------------------------------ LOSS FUNCTIONS -------------------------------------------------------------------#

"""
Define a custom "WEIGHTED" loss function MSE: it penalizes predictions of pixels 
with flux below average with more error than pixels having flux above average
"""
#1)
def weightedL2loss(w):
    def loss(y_true, y_pred):
        error = K.square(y_true - y_pred)
        error = K.switch(K.equal(y_pred, 0), w * error , error)
        return error 
    return loss

#2) Downweight bright pixels with a power law (alpha should be between 0 and 1)

def downweight_loss(alpha):
    def loss(y_true, y_pred):
        y_true_clipped = K.clip(y_true, K.epsilon(), 1.0)
        y_pred_clipped = K.clip(y_pred, K.epsilon(), 1.0)

        y_true_rescaled = K.pow(y_true_clipped, alpha)
        y_pred_rescaled = K.pow(y_pred_clipped, alpha)

        error = K.square(y_true_rescaled - y_pred_rescaled)
        return error
    return loss

def log_downweight_loss(y_true, y_pred):
    
    y_true_rescaled = tf.math.log(1 + y_true)
    y_pred_rescaled = tf.math.log(1 + y_pred)

    error = K.square(y_true_rescaled - y_pred_rescaled)
    return K.mean(error)

#----------------------------------------------------------- HYPER-PARAMETERS ------------------------------------------------------------------#
                                                                                                                                                
EPOCHS = 300                                                               
LEARNING_RATE = 1

print("Running for", EPOCHS, "epochs")

#----------------------------------------------------------------- MODEL -----------------------------------------------------------------------#

# Model: U-Net (usually used for Image Segmentation of Biomedical images)

# Define construction functions for fundamental blocks

def double_conv_block(x, n_filters):

    x = Conv2D(n_filters, 3, padding = "same", kernel_initializer = "he_normal")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(n_filters, 3, padding = "same", kernel_initializer = "he_normal")(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = MaxPool2D(2)(f)
    # p = Dropout(0.3)(p)
    return f, p

def upsample_block(x, conv_features, n_filters):
    # 3: kernel size
    # 2: strides
    x = Conv2DTranspose(n_filters, 3, 2, padding='same')(x)
    x = concatenate([x, conv_features])
    # x = Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x

# Build the U-Net model

def make_unet_model(image_size):
    inputs = Input(shape=(image_size[0], image_size[1], 1))

    # Encoder
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)
    
    # Bottleneck
    bottleneck = double_conv_block(p4, 1024)
    
    # Decoder
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    # Output
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(u9)

    unet_model = Model(inputs, outputs, name='U-Net')

    return unet_model

unet_model = make_unet_model(image_size)

unet_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE),
                  loss='mse',
                  metrics=['mse'])

#------------------------------------------------------------- CALLBACKS -----------------------------------------------------------------------#

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=5,
                                           restore_best_weights=True,
                                           start_from_epoch=150)

#------------------------------------------------------ TRAINING (on GPU 'gpu03') --------------------------------------------------------------#

hist = unet_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stop]
)
#--------------------------------------------------------------- SAVING ------------------------------------------------------------------------#

unet_model.save("/scratch/astro/nicolo.fiaba/trained_models/unet_model_4_180.h5")

print("U-Net trained and saved!")

import pickle
with open('/scratch/astro/nicolo.fiaba/trained_models/UNET_hist_4_180', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)

print("\nLearning History saved!")
#---------------------------------------------------------------- END --------------------------------------------------------------------------#
