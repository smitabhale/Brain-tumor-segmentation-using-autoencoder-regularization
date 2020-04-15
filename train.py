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


class GenerateData(Sequence):
    def __init__(self, data, input_shape, output_channels, batch_size):
        self.data = data
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        imgs = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        data_x = np.empty((self.batch_size,) + self.input_shape, dtype=np.float32)
        labels = np.empty((self.batch_size, self.output_channels) + self.input_shape[1:], dtype=np.uint8)

        for index, img in enumerate(imgs):
            data_x[index] = np.array([preprocess(read_img(img[m]), self.input_shape[1:]) for m in ['t1', 't2', 't1ce', 'flair']], dtype=np.float32)
            labels[index] = preprocess_label(read_img(img['seg']), self.input_shape[1:])[None, ...]
        
        # batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        # batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        # print("X shape",data_x.shape)
        # print("Y shape",labels.shape)
        return data_x, labels

dataset_path = "../MICCAI_BraTS_2018_Data_Training.zip"  # Replace with your dataset path
zfile = zipfile.ZipFile(dataset_path)
zfile.extractall()

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
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
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
    


# Get a list of files for all modalities individually
t1 = glob.glob('*GG/*/*t1.nii.gz')
t2 = glob.glob('*GG/*/*t2.nii.gz')
flair = glob.glob('*GG/*/*flair.nii.gz')
t1ce = glob.glob('*GG/*/*t1ce.nii.gz')
seg = glob.glob('*GG/*/*seg.nii.gz')  # Ground Truth


pat = re.compile('.*_(\w*)\.nii\.gz')

data_paths = [{
    pat.findall(item)[0]:item
    for item in items
}
for items in list(zip(t1, t2, t1ce, flair, seg))]

input_shape = (4, 96, 112, 112)
# input_shape = (4, 155, 240, 240)
output_channels = 3
batch_size = 1
# data = np.empty((len(data_paths[:4]),) + input_shape, dtype=np.float32)
# labels = np.empty((len(data_paths[:4]), output_channels) + input_shape[1:], dtype=np.uint8)
# 
# total = len(data_paths[:4])
# step = 25 / total
# 
# for i, imgs in enumerate(data_paths[:4]):
#     try:
#         data[i] = np.array([preprocess(read_img(imgs[m]), input_shape[1:]) for m in ['t1', 't2', 't1ce', 'flair']], dtype=np.float32)
#         labels[i] = preprocess_label(read_img(imgs['seg']), input_shape[1:])[None, ...]
#         
#         # Print the progress bar
#         print('\r' + 'Progress: ' + "[%s %s]"%('=' * int((i+1) * step), ' ' * (24 - int((i+1) * step))) + "(%s percentage)"%(math.ceil((i+1) * 100 / (total))),
#             end='')
#     except Exception as e:
#         print('Something went wrong with %s, skipping...\n Exception:\n%s'%(imgs["t1"], str(e)))
#         continue

train_path = data_paths[0:275]
val_path = data_paths[275:]

print('train_images:', train_path)
print('val_path:', val_path)

train_generator = GenerateData(train_path, input_shape, output_channels, batch_size)
val_generator = GenerateData(val_path, input_shape, output_channels, batch_size)
nb_epoch = 10
model = build_model(input_shape=input_shape, output_channels=3)
check = ModelCheckpoint("weights/weights.epoch:{epoch:02d}-loss:{loss:.5f}-dice:{dice_coefficient:.5f}.hdf5", monitor='dice_coefficient', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
# model.fit_generator(generator=train_generator,
#                     validation_data = val_generator,
#                     use_multiprocessing=False,
#                     workers=1,
#                     verbose=0,
#                     epochs=nb_epoch,
#                     steps_per_epoch=len(train_generator),
#                     callbacks=[check,TQDMCallback()],)
# model.fit(data, labels, batch_size=1, epochs=5)
# preds = model.predict(np.array([data[0]]))

# pred = preds[0]
# print(pred.shape)
# print(pred[0,:,:,:].shape)
# pred[:, 50,:,:] #50th slice 
# img = pred.sum(axis=0)
# img = img.sum(axis=0)
# img = (img>1).astype(np.uint8)
# print(img.shape)
# print(np.unique(img))
# print(img.sum())
# plt.imshow(pred[:, 50,:,:], cmap='Greys_r')
