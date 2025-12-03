## Training a simple autoencoder

from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import keras
from keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Reshape, concatenate, Dropout, Rescaling, LeakyReLU
from keras.models import Model
from astropy.io import fits
import matplotlib.pyplot as plt
from gelsa import visu

print(tf.config.list_physical_devices('GPU'))

hdu_rect = fits.open("/scratch/astro/nicolo.fiaba/training_sets_square/training_set_square.fits", memmap=True)
hdu_rect_lines = fits.open("/scratch/astro/nicolo.fiaba/training_sets_square/training_set_square_EL.fits", memmap=True)

img_data = np.array([hdu.data for hdu in hdu_rect[1:]])
img_data_lines = np.array([hdu.data for hdu in hdu_rect_lines[1:]])


# ### Define the masks for the emission lines
# 
# - Take the difference between each image with and without ELs
# - Leave each 0 pixel to 0 and set each non-zero pixel to 1


diff = img_data_lines - img_data
threshold = np.mean(diff) + 2.5*np.std(diff)
mask_data = (diff > threshold).astype(np.uint8)


# test set is 10% of the whole dataset
x_train, x_test, y_train, y_test = train_test_split(
    img_data_lines, img_data, test_size=0.1, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

x_train_, x_test_, y_train_mask, y_test_mask = train_test_split(
    img_data_lines, mask_data, test_size=0.1, random_state=42)

x_train_, x_val_, y_train_mask, y_val_mask = train_test_split(
    x_train_, y_train_mask, test_size=0.2, random_state=42)


# This preprocessing layer rescales the images to be in the 0, 1 range

# If you are using images with lines, normalize to those
# rescale = Rescaling(scale=1/img_data_lines.max(), offset=0.0)

# If you are using ONLY images without lines, normalize to those
rescale = Rescaling(scale=1/img_data.max(), offset=0.0)

# scaled images. Recasted into the (0,1) range
train = rescale(y_train)
train_lines = rescale(x_train)
train_mask = y_train_mask

val = rescale(y_val)
val_lines = rescale(x_val)
val_mask = y_val_mask

test = rescale(y_test)
test_lines = rescale(x_test)
test_mask = y_test_mask

image_size = (train.shape[1], train.shape[2])

print("Images in training set:", len(train), 
      "\nImages in validation set:", len(val), 
      "\nImages in test set:", len(test), 
      "\nImage size:", image_size)

#---------------------------------------------------- LOSS FUNCTIONS ------------------------------------------------------------#

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

#----------------------------------------------------------- HYPER-PARAMETERS ------------------------------------------------------------------#

EPOCHS = 100                                                                                                                                    
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

print("Running for", EPOCHS, "epochs")

#---------------------------------------------------------- MODEL -----------------------------------------------------------------#

# Model: Autoencoder

# Define construction functions

def conv_block(x, n_filters):
    x = Conv2D(n_filters, 3, padding = "same", kernel_initializer = "he_normal")(x)
    x = LeakyReLU(alpha=0.3)(x)
    return x

def downsample(x, n_filters):
    d = conv_block(x, n_filters)
    d = MaxPool2D(2)(d)
    d = Dropout(0.3)(d)
    return d

def upsample(x, n_filters):
    # 3: kernel size
    # 2: strides
    u = Conv2DTranspose(n_filters, 3, 2, padding='same')(x)
    u = Dropout(0.3)(u)
    u = conv_block(u, n_filters)
    return u

# Build the U-Net model

def make_autoencoder(image_size):
    inputs = Input(shape=(image_size[0], image_size[1], 1))

    # Encoder
    x = downsample(inputs, 64)
    x = downsample(x, 128)
    x = downsample(x, 256)
    x = downsample(x, 512)
    
    # Bottleneck
    x = conv_block(x, 1024)
    
    # Decoder
    x = upsample(x, 512)
    x = upsample(x, 256)
    x = upsample(x, 128)
    x = upsample(x, 64)

    # Output
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(x)

    autoencoder = Model(inputs, outputs, name='Autoencoder')

    return autoencoder

autoencoder = make_autoencoder(image_size)

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=downweight_loss(1/4),
                  metrics="accuracy")

#------------------------------------------------------------- CALLBACKS -----------------------------------------------------------------------#

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=3,
                                           restore_best_weights=True,
                                           start_from_epoch=100)


#------------------------------------------------------ TRAINING (on GPU 'gpu03') --------------------------------------------------------------#

autoencoder.fit(
    train, train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(val, val),
    callbacks=[early_stop]
)

#--------------------------------------------------------------- SAVING  -----------------------------------------------------------------------#

autoencoder.save("/scratch/astro/nicolo.fiaba/trained_models/autoencoder_leakyrelu.h5")

print("Autoencoder trained and saved!")

#### End
