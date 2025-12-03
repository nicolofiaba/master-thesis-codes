#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ## Training an autoencoder

# In[2]:


# In[4]:


import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, Conv2DTranspose, Reshape, Concatenate
from keras.models import Model
from astropy.io import fits
import keras

# Set hyerparameters
# EPOCHS = 
# BATCHES = 
# LEARNING_RATE = 


# ### Importing the data

# In[5]:


hdu_rect = fits.open("/scratch/astro/nicolo.fiaba/training_sets/training_set.fits", memmap=True)
hdu_rect_lines = fits.open("/scratch/astro/nicolo.fiaba/training_sets/training_set_EL.fits", memmap=True)


# In[6]:


img_data = np.array([hdu.data for hdu in hdu_rect[1:]])
img_data_lines = np.array([hdu.data for hdu in hdu_rect_lines[1:]])


scale_img = img_data/img_data.max()
scale_img_lines = img_data_lines/img_data_lines.max()


# In[20]:


image_size = (scale_img.shape[1], scale_img.shape[2])
num_images = len(scale_img)

# ### Define the Autoencoder
# 
# We have two different architectures:
# 1. Standard autoencoder (encoder+decoder) with 3 stages
# 2. Autoencoder with skip connections, like U-Net

# In[22]:



# A different architecture: [Image Inpainting Using AutoEncoder and Guided Selection of Predicted Pixels by Mohammad H. Givkashi](https://arxiv.org/pdf/2112.09262)

# In[23]:


# Model 2: Encoder-Decoder architecture suggested Mohammad H. Givkashi paper "Image Inpainting Using AutoEncoder and Guided Selection of Predicted Pixels"

def make_autoencoder_model_2(image_size):
    inputs = Input(shape=(image_size[0], image_size[1], 1))

    # Encoder
    x1 = Conv2D(16, (2, 10), activation='relu', padding='same')(inputs)
    x1 = Conv2D(16, (2, 10), activation='relu', padding='same')(x1)
    p1 = MaxPooling2D((2, 2), padding='same')(x1)

    x2 = Conv2D(16, (2, 7), activation='relu', padding='same')(p1)
    x2 = Conv2D(16, (2, 7), activation='relu', padding='same')(x2)
    p2 = MaxPooling2D((2, 2), padding='same')(x2)

    x3 = Conv2D(16, (2, 5), activation='relu', padding='same')(p2)
    x3 = Conv2D(16, (2, 5), activation='relu', padding='same')(x3)
    p3 = MaxPooling2D((2, 2), padding='same')(x3)

    x4 = Conv2D(16, (2, 5), activation='relu', padding='same')(p3)
    x4 = Conv2D(16, (2, 5), activation='relu', padding='same')(x4)
    p4 = MaxPooling2D((2, 2), padding='same')(x4)

    bottleneck = Conv2D(16, (2, 2), activation='relu', padding='same')(p4)
    bottleneck = Conv2D(16, (2, 2), activation='relu', padding='same')(bottleneck)

    # Decoder with skip connections
    u1 = Conv2DTranspose(16, (2,2), strides=(2,2), activation='relu', padding='same')(bottleneck)
    u1 = Concatenate()([u1, x4])

    u2 = Conv2DTranspose(16, (2,2), strides=(2,2), activation='relu', padding='same')(u1)
    u2 = Concatenate()([u2, x3])

    u3 = Conv2DTranspose(16, (2,2), strides=(2,2), activation='relu', padding='same')(u2)
    u3 = Concatenate()([u3, x2])

    u4 = Conv2DTranspose(16, (2,2), strides=(2,2), activation='relu', padding='same')(u3)
    u4 = Concatenate()([u4, x1])

    outputs = Conv2DTranspose(1, (2,2), activation='sigmoid', padding='same')(u4)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


# In[24]:

autoencoder_2 = make_autoencoder_model_2(image_size)


# In[29]:


import os
checkpoint_path = "/scratch/astro/nicolo.fiaba/training_weights/ckp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=3,
                                          restore_best_weights=True,
					  start_from_epoch=150)


# In[30]:

batch_size = 50
epochs = 300
print(epochs)


# In[31]:


hist = autoencoder_2.fit(
    scale_img_lines, scale_img,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2,
    callbacks=[cp_callback, early_stop]
)

print("\nTraining DONE.\n")
print(len(hist.history['loss']))
