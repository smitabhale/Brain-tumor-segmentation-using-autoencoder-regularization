img = read_img('LGG/Brats18_2013_0_1/Brats18_2013_0_1_seg.nii.gz')
resized = resize(img, shape=(144,240,240))
resized_ = sitk.GetImageFromArray(resized)
sitk.WriteImage(resized_, 'test.nii.gz'
