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

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
  except RuntimeError as e:
    print(e)

# Load images ---------------------------------------------------------------------------------
path = "/home/PERSONALE/nicolo.fiaba/.cache/kagglehub/datasets/aryankaushik005/pixart/versions/1/PixART dataset/automobile"
image_files = glob.glob(os.path.join(path, "*.jpg"))
images = [mpimg.imread(img_path) for img_path in image_files]
images = np.array(images, dtype=np.float32)

# Train-test split ----------------------------------------------------------------------------
# test set is 10% of the whole dataset
x_train, x_test, y_train, y_test = train_test_split(
    images, images, test_size=0.1, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

rescale = Rescaling(1./255, input_shape=(512, 512, 3))

train = rescale(y_train)
val = rescale(y_val)
test = rescale(y_test)

image_size = (train.shape[1], train.shape[2])

print("Images in training set:", len(train), 
      "\nImages in validation set:", len(val), 
      "\nImages in test set:", len(test), 
      "\nImage size:", image_size)

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
                                                                                                                                                
EPOCHS = 50                                                                                                                                
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

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
    inputs = Input(shape=(image_size[0], image_size[1], 3))

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
    outputs = Conv2D(3, 1, padding='same', activation='sigmoid')(u9)

    unet_model = Model(inputs, outputs, name='U-Net')

    return unet_model

unet_model = make_unet_model(image_size)

unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse',
                  metrics=['mse'])

#------------------------------------------------------------- CALLBACKS -----------------------------------------------------------------------#

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=3,
                                           restore_best_weights=True,
                                           start_from_epoch=100)

#------------------------------------------------------ TRAINING (on GPU 'gpu03') --------------------------------------------------------------#

unet_model.fit(
    train, train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(val, val),
    callbacks=[early_stop]
)

#--------------------------------------------------------------- SAVING ------------------------------------------------------------------------#

unet_model.save("/scratch/astro/nicolo.fiaba/trained_models/unet_model_cars.h5")

print("U-Net trained and saved!")

#---------------------------------------------------------------- END --------------------------------------------------------------------------#


