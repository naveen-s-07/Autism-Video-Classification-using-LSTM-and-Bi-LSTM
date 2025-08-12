import sys
import os
import numpy as np
import keras
import tensorflow as tf
#import imutils
#from imutils import paths

import matplotlib.pyplot as pltpip
import pandas as pd
import numpy as np
import imageio
import cv2
from IPython.display import Image
from keras.applications import InceptionV3

# Check if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Allow memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU.")

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 40

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

os.system('cls')