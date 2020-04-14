import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def read_img(img_path):
    """
    Reads a .nii.gz image and returns as a numpy array.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

slice_no = 60
s = read_img('/home/suraj/Documents/smita/MICCAI_BraTS_2018_Data_Training/LGG/Brats18_TCIA12_101_1/Brats18_TCIA12_101_1_seg.nii.gz')
# img = s[90 ,:,:]
# s = read_img('/home/suraj/Documents/smita/MICCAI_BraTS_2018_Data_Training/LGG/Brats18_TCIA12_101_1/Brats18_TCIA12_101_1_flair.nii.gz')
# print(s.shape)
# # s = s.sum(axis=0)
# plt.imshow(s[90 ,:,:], cmap='gray')
# plt.show()




fig = plt.figure() # make figure

# make axesimage object
# the vmin and vmax here are very important to get the color map correct
im = plt.imshow(s[0], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

# function to update figure
def updatefig(j):
    # set the data in the axesimage object
    print(j)
    print(np.unique(s[j]))
    im.set_array(s[j])
    # return the artists set
    return [im]
# kick off the animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(1,155), 
                              interval=50, blit=True)
plt.show()