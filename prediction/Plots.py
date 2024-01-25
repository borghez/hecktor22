import nibabel as nib
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import numpy as np

from glob import glob

HECKTOR_PATH = nib.load('C:\\Users\\andre\\Desktop\\HECKTOR_DATA\\hecktor2022_training\\imagesTr\\CHUM-001__CT.nii.gz')

"""Exemple of an HECKTOR CT image slices"""

hecktor_image = HECKTOR_PATH.get_fdata()

fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
n_slice = hecktor_image.shape[2]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(ndi.rotate(hecktor_image[:, :, img], 90), cmap='gray')
    axs.flat[idx].axis('off')

plt.tight_layout()
plt.show()

#%%

"""Example of an UNIPD CT image slices"""

UNIPD_PATH = nib.load('C:\\Users\\andre\\Desktop\\DATI PADOVA\\Volums\\CT_nifti\\UNIPD-001__CT.nii.gz')
unipd_img = UNIPD_PATH.get_fdata()

n_slice = unipd_img.shape[2]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(ndi.rotate(hecktor_image[:, :, img], 90), cmap='gray')
    axs.flat[idx].axis('off')

plt.tight_layout()
plt.show()

#%%

"""Example of an HECKTOR mask image slices"""

HECKTOR_MASK_PATH = nib.load('C:\\Users\\andre\\Desktop\\HECKTOR_DATA\\hecktor2022_training\\labelsTr\\CHUM-001.nii.gz')

hecktor_mask = HECKTOR_MASK_PATH.get_fdata()

fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
n_slice = hecktor_mask.shape[2]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(ndi.rotate(hecktor_mask[:, :, img], 90), cmap='gray')
    axs.flat[idx].axis('off')

plt.tight_layout()
plt.show()

#%%

"""Example of an UNIPD mask image slices"""

UNIPD_MASK_PATH = nib.load('C:\\Users\\andre\\Desktop\\DATI PADOVA\\Masks\\CT_nifti\\TOT_MASK\\UNIPD-001.nii.gz')

unipd_mask = UNIPD_MASK_PATH.get_fdata()

fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
n_slice = unipd_mask.shape[2]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(ndi.rotate(unipd_mask[:, :, img], 90), cmap='gray')
    axs.flat[idx].axis('off')

plt.tight_layout()
plt.show()

#%%

"""

Plotting the same slice of all the different CT patient of UNIPD

Seems like not all the patient classified as CT in the Excel have an 
actual CT image
 
"""

data_dir = "C:/Users/andre/Desktop/DATI PADOVA/Volums/CT_nifti"
images = sorted(glob(os.path.join(data_dir, "UNIPD*.nii.gz")))
mask_dir = "C:/Users/andre/Desktop/DATI PADOVA/Masks/CT_nifti/TOT_MASK"
mask = sorted(glob(os.path.join(mask_dir, "UNIPD*.nii.gz")))

fig_rows = 4
fig_cols = 5
fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])
ax = axs.ravel()

for pat in range(len(images)):

    imm_load = nib.load(images[pat])
    imm = imm_load.get_fdata()
    msk_load = nib.load(mask[pat])
    msk = msk_load.get_fdata()

    ax[pat].imshow(imm[:, :, 20])
    #ax[pat].imshow(msk[:, :, 20])


plt.show()

fig.savefig('C:/Users/andre/Desktop/CT_patients.jpg')

#%%

"""Plotting the same slice of all the different MRI patient of UNIPD"""


data_dir = "C:/Users/andre/Desktop/DATI PADOVA/Volums/MRI_nifti"
images = sorted(glob(os.path.join(data_dir, "UNIPD*.nii.gz")))
mask_dir = "C:/Users/andre/Desktop/DATI PADOVA/Masks/MRI_nifti/TOT_MASK"
mask = sorted(glob(os.path.join(mask_dir, "UNIPD*.nii.gz")))

fig_rows = 4
fig_cols = 5
fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])
ax = axs.ravel()

for pat in range(len(images)):

    imm_load = nib.load(images[pat])
    imm = imm_load.get_fdata()
    msk_load = nib.load(mask[pat])
    msk = msk_load.get_fdata()

    #ax[pat].imshow(imm[:, :, 20])
    ax[pat].imshow(msk[:, :, 20])
    print(np.max(imm))


plt.show()


#%%

"""Plotting the images that do not seems to be CT images"""

data_dir = "C:/Users/andre/Desktop/DATI PADOVA/Volums/CT_nifti"
images = sorted(glob(os.path.join(data_dir, "UNIPD*.nii.gz")))
mask_dir = "C:/Users/andre/Desktop/DATI PADOVA/Masks/CT_nifti/TOT_MASK"
mask = sorted(glob(os.path.join(mask_dir, "UNIPD*.nii.gz")))

fig_rows = 2
fig_cols = 2
fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])
ax = axs.ravel()

imm_0 = nib.load(images[0])
imm0 = imm_0.get_fdata()
imm_1 = nib.load(images[1])
imm1 = imm_1.get_fdata()
imm_5 = nib.load(images[5])
imm5 = imm_5.get_fdata()
imm_15 = nib.load(images[15])
imm15 = imm_15.get_fdata()


ax[0].imshow(imm0[:, :, 20])
ax[1].imshow(imm1[:, :, 20])
ax[2].imshow(imm5[:, :, 20])
ax[3].imshow(imm15[:, :, 20])


fig.savefig('C:/Users/andre/Desktop/Images classified wrong as CT.jpg')
plt.show()
