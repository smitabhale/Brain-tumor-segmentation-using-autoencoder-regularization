import zipfile  # For faster extraction
import SimpleITK as sitk  # For loading the dataset
import numpy as np  # For data manipulation
from model import build_model  # For creating the model
import glob  # For populating the list of files
from scipy.ndimage import zoom  # For resizing
import re  # For parsing the filenames (to know their modality)
import math
import matplotlib.pyplot as plt
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMCallback
import matplotlib.pyplot as plt
import pandas as pd

def read_img(img_path):
    """
    Reads a .nii.gz image and returns as a numpy array.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))


def resize(img, shape, mode='constant', orig_shape=(155, 240, 240)):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """
    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0]/orig_shape[0],
        shape[1]/orig_shape[1], 
        shape[2]/orig_shape[2]
    )
    
    # Resize to the given shape
    return zoom(img, factors, mode=mode)


def preprocess(img, out_shape=None):
    """
    Preprocess the image.
    Just an example, you can add more preprocessing steps if you wish to.
    """
    if out_shape is not None:
        img = resize(img, out_shape, mode='constant')
    
    # Normalize the image
    mean = img.mean()
    std = img.std()
    return (img - mean) / std


def preprocess_label(img, out_shape=None, mode='nearest'):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET ? label 4), the peritumoral edema (ED ? label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET ? label 1)
    """
    # print(img.shape)
    # print(np.unique(img))
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 4  # GD-enhancing Tumor (ET)
    
    if out_shape is not None:
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)

    return np.array([ncr, ed, et], dtype=np.uint8)

input_shape = (4, 96, 112, 112)
output_channels = 3
model = build_model(input_shape=input_shape, output_channels=3)
model.load_weights('weights/weights.epoch_100-loss_-0.14610-dice_0.64230-val_dice_0.58464.hdf5')

sample = {'t1': 'HGG/Brats18_TCIA01_460_1/Brats18_TCIA01_460_1_t1.nii.gz',
           't2': 'HGG/Brats18_TCIA01_460_1/Brats18_TCIA01_460_1_t2.nii.gz',
           't1ce': 'HGG/Brats18_TCIA01_460_1/Brats18_TCIA01_460_1_t1ce.nii.gz',
           'flair': 'HGG/Brats18_TCIA01_460_1/Brats18_TCIA01_460_1_flair.nii.gz',
           'seg': 'HGG/Brats18_TCIA01_460_1/Brats18_TCIA01_460_1_seg.nii.gz'}


data_x = np.empty((1,) + input_shape, dtype=np.float32)
labels = np.empty((1, output_channels) + input_shape[1:], dtype=np.uint8)

data_x[0] = np.array([preprocess(read_img(sample[m]), input_shape[1:]) for m in ['t1', 't2', 't1ce', 'flair']], dtype=np.float32)
labels[0] = preprocess_label(read_img(sample['seg']), input_shape[1:])[None, ...]

# pred = model.predict(data_x)
# a = np.rint(pred)
# pred = a*255
# pred = pred[0]
# pred = pred.reshape()
img = labels[0,:, 50,:,:]
img = np.transpose(img, (1, 2, 0))
# plt.imshow(labels[0,:, 50,:,:], cmap='Greys_r')        
# plt.imshow(img, cmap='Greys_r')
