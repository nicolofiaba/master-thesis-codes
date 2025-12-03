## Training a U-Net autoencoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Reshape, concatenate, Dropout, Rescaling, LeakyReLU
import tensorflow.keras.layers as L
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
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--batch", type=int, default=16, help="Batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--grism", type=str, default="RGS000_0", help="Grism + tilt combination")
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

# Hyperparameters
BATCH_SIZE = args.batch
LEARNING_RATE = args.lr
EPOCHS = args.epochs
# ---------------------------------------------------------------------------------
# Check if Grism combination is valid:



# ---------------------------------------------------------------------------------
folder_path = "/scratch/astro/nicolo.fiaba/training_sets/"

labels_filename = "labels_" + args.grism + ".fits"
train_filename = "train_" + args.grism + ".fits"

labels_path = folder_path + labels_filename
train_path = folder_path + train_filename

hdu_rect = fits.open(labels_path, memmap=True)
hdu_rect_masked = fits.open(train_path, memmap=True)

img_data = np.array([hdu.data for hdu in hdu_rect[1:]])
img_data_masked = np.array([hdu.data for hdu in hdu_rect_masked[1:]])

""" Mask object: 
- 0: Existing pixel
- 1: Missing pixel
"""
mask = np.isnan(img_data_masked).astype(np.float32)

# Setting a maximum flux threshold to 16000. All pixels brighter than that are set to 16000.
max_threshold = 16000
img_data[img_data > max_threshold] = max_threshold
img_data_masked[img_data_masked > max_threshold] = max_threshold

# Add noise
sigma = np.sqrt(800)
size = (len(img_data), 512, 512)

noise = np.random.normal(loc=0, scale=sigma, size=size)
img_noise = img_data #+ noise
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

# Input: masked images + mask (2 channels)
input_imgs = np.stack([img_noise_masked, mask], axis=-1)

# test set is 10% of the whole dataset
x_train, x_test, y_train, y_test = train_test_split(
    input_imgs, img_noise, test_size=0.1, random_state=42)

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

print("\n", tf.reduce_max(x_train))
print("\n", tf.reduce_min(x_train))

# Creating Tensorflow datasets

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

def get_gradients(img):
    # img: (batch, H, W, 1)
    if len(img.shape) == 3:
        img = tf.expand_dims(img, axis=-1)  # add channel
    # horizontal gradient (dx)
    gx = tf.image.sobel_edges(img)[..., 0]
    # vertical gradient (dy)
    gy = tf.image.sobel_edges(img)[..., 1]

    return gx, gy

def gradient_loss(y_true, y_pred):
    gx_true, gy_true = get_gradients(y_true)
    gx_pred, gy_pred = get_gradients(y_pred)

    loss_gx = tf.reduce_mean(tf.abs(gx_true - gx_pred))
    loss_gy = tf.reduce_mean(tf.abs(gy_true - gy_pred))

    return loss_gx + loss_gy

def total_gradient_loss(y_true, y_pred):
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    g = gradient_loss(y_true, y_pred)

    return l1 + 0.2 * g 

#----------------------------------------------------------- HYPER-PARAMETERS ------------------------------------------------------------------#                                                                                                                                           
print("Running for", EPOCHS, "epochs")

#----------------------------------------------------------------- MODEL -----------------------------------------------------------------------#

# Model: Attention gate - U-Net

# Define construction functions for fundamental blocks

def conv_block(x, num_filters):
    x = L.Conv2D(num_filters, 3, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    x = L.Conv2D(num_filters, 3, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    return x

def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    p = L.MaxPool2D((2,2))(x)
    return x, p

def attention_gate(g, s, num_filters):
    Wg = L.Conv2D(num_filters, (2, 32), padding='same')(g)
    Wg = L.BatchNormalization()(Wg)

    Ws = L.Conv2D(num_filters, (2, 32), padding='same')(s)
    Ws = L.BatchNormalization()(Ws)

    out = L.Activation("relu")(Wg + Ws)
    out = L.Conv2D(num_filters, (2, 32), padding='same')(out)
    out = L.Activation("sigmoid")(out)

    return out * s

def decoder_block(x, s, num_filters):
    x = L.UpSampling2D(interpolation='bilinear')(x)
    s = attention_gate(x, s, num_filters)
    x = L.Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x

# Build the Attention U-Net model

def attention_unet(image_size):
    """ Inputs """
    inputs = L.Input(shape=(image_size[0], image_size[1], 2))

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    """ Bridge / Bottleneck """
    b1 = conv_block(p4, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Outputs """
    outputs = L.Conv2D(1, (2, 16), padding='same', activation='sigmoid')(d4)

    attention_unet_model = Model(inputs, outputs, name='Attention-UNET')
    return attention_unet_model

att_unet_model = attention_unet(image_size)

att_unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=total_gradient_loss,
                  metrics=['mae'])

#------------------------------------------------------------- CALLBACKS -----------------------------------------------------------------------#

# Learning rate scheduler
def lr_schedule(epoch):
    if epoch < 80:
        return 2e-3
    elif epoch < 250:
        return 1e-4
    else:
    	return 1e-5

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Early stop
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=10,
                                           restore_best_weights=True,
                                           start_from_epoch=300)

#------------------------------------------------------ TRAINING (on GPU 'gpu03') --------------------------------------------------------------#

hist = att_unet_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stop, lr_callback]
)

#--------------------------------------------------------------- SAVING ------------------------------------------------------------------------#
saving_folder = "/scratch/astro/nicolo.fiaba/trained_models/"
saving_filename = "denoise_rect_attention_unet_model_" + args.grism + ".h5"

att_unet_model.save(saving_folder + saving_filename)

print("Attention U-Net trained and saved!")

history_filename = "denoise_rect_ATT_UNET_hist_" + args.grism
import pickle
with open(saving_folder + history_filename, 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)

print("\nLearning History saved!")
#---------------------------------------------------------------- END --------------------------------------------------------------------------#
