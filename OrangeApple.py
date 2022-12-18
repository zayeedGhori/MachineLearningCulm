'''
Training data provided by: https://www.kaggle.com/mbkinaci/fruit-images-for-object-detection/data. This 
program trains a classifier to classify between images of bananas, oranges, and apples. It uses the 
tensorflow library and the included images to train the neural network classifier. The program is not 
efficient enough to run on a computer, yet.

'''

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os

learn = tf.contrib.learn
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

labels = []
data = []

# Generator for raw images to convert into proper pixel grid format
train_image_generator = ImageDataGenerator(rescale=1./255)

train_dir = r'.\tf_files\fruit_photos'

# Saves images, assigning the class of images based on which subdirectory it is in
# Randomly assigns batches of 100 images
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                        shuffle=True,
                                                        batch_size=100,
                                                        class_mode='binary')

# unpacks the first hundred images into raw images and labels
data, labels = next(train_data_gen)

# converts to int array
labels = np.asarray(labels, dtype=np.int32)

# train classifier
feature_columns = learn.infer_real_valued_columns_from_input(data) # learns features from images
classifier = learn.LinearClassifier(feature_columns=feature_columns, n_classes=3) # sets up classifier to learn based on features
classifier.fit(data, labels, batch_size=100, steps=1000) # learns from training data

print(classifier.predict(data[0]))