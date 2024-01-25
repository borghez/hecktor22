from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import rotate

# example_names = [
#     "mda202", "chb001", "usz-001", "chum013", "chuv001", "chus003", "hgj018", "hmr001", "chup013"
# ]
# patient_ids = [
#     "MDA-202", "CHB-001", "USZ-001", "CHUM-013", "CHUV-001", "CHUS-003", "HGJ-018", "HMR-001", "CHUP-013",
# ]
example_names = [
    "chuv010", "chus003", "hgj018", "hmr001", "chup013"
]
patient_ids = [
    "CHUV-010", "CHUS-003", "HGJ-018", "HMR-001", "CHUP-013",
]
path_train = Path(
    "/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/")
path_test = Path(
    "/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/")
# example_paths = [
#     path_test / "imagesTs",
#     path_test / "imagesTs",
#     path_test / "imagesTs",
#     path_train / "imagesTr",
#     path_train / "imagesTr",
#     path_train / "imagesTr",
#     path_train / "imagesTr",
#     path_train / "imagesTr",
#     path_train / "imagesTr",
# ]
# example_paths_gtvt = [
#     Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/test_label"),
#     Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/test_label"),
#     Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/test_label"),
#     Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
#     Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
#     Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
#     Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
#     Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
#     Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
# ]

example_paths = [
    path_train / "imagesTr",
    path_train / "imagesTr",
    path_train / "imagesTr",
    path_train / "imagesTr",
    path_train / "imagesTr",
]
example_paths_gtvt = [
    Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
    Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
    Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
    Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
    Path("/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label"),
]


def to_np(image):
    return np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))


def clip(image, clip_range):
    image[image < clip_range[0]] = clip_range[0]
    image[image > clip_range[1]] = clip_range[1]
    image = (2 * image - clip_range[1] - clip_range[0]) / (clip_range[1] -
                                                           clip_range[0])
    return image


clipping_range_ct = [-140, 260]
clipping_range_pt = [0, 12]

resampler = sitk.ResampleImageFilter()
resampler.SetInterpolator(sitk.sitkBSpline)
delta_y = [20, 30, 20, 40, 30]
delta_z = [10, 10, 0, 0, 0]
for i, (path, gt_folder, patient_id) in enumerate(
        zip(example_paths, example_paths_gtvt, patient_ids)):
    ct_path = str((path / (patient_id + "__CT.nii.gz")).resolve())
    ct = sitk.ReadImage(ct_path)

    resampler.SetOutputSpacing((1, 1, 1))
    resampler.SetOutputOrigin(ct.GetOrigin())
    resampler.SetSize(tuple(int(l * r) for l, r in zip(ct.GetSize(), ct.GetSpacing())))

    pt_path = str((path / (patient_id + "__PT.nii.gz")).resolve())
    pt = sitk.ReadImage(pt_path)
    gt_path = str((gt_folder / (patient_id + ".nii.gz")).resolve())
    gt = sitk.ReadImage(gt_path)

    pt = to_np(resampler.Execute(pt))
    ct = to_np(resampler.Execute(ct))
    gt = to_np(resampler.Execute(gt))

    ct = np.flip(ct, axis=1)
    pt = np.flip(pt, axis=1)
    gt = np.flip(gt, axis=1)

    ct = clip(ct, clipping_range_ct)
    pt = clip(pt, clipping_range_pt)

    positions = np.where(gt == 1)
    slice_x = (np.max(positions[0]) + np.min(positions[0])) // 2

    fig = plt.figure(figsize=(2, 2))
    ax = plt.subplot(111, aspect='equal')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    ct_im = rotate(ct[slice_x, :, :], 90)
    pt_im = rotate(pt[slice_x, :, :], 90)
    ax.axis('off')
    ax.imshow(ct_im, cmap='gray')
    ax.imshow(pt_im, cmap='hot', alpha=0.5)
    ax.margins(tight=True)
    plt.savefig(f"{patient_id}.png")

from PIL import Image

image1 = Image.open("C:\\Users\\andre\\Desktop\\CHUP-013.png")
image2 = Image.open("C:\\Users\\andre\\Desktop\\CHUM-013.png")
image3 = Image.open("C:\\Users\\andre\\Desktop\\CHUV-010.png")
image4 = Image.open("C:\\Users\\andre\\Desktop\\CHUS-003.png")
image5 = Image.open("C:\\Users\\andre\\Desktop\\HGJ-018.png")
image6 = Image.open("C:\\Users\\andre\\Desktop\\HMR-001.png")
image7 = Image.open("C:\\Users\\andre\\Desktop\\MDA-202.png")
image8 = Image.open("C:\\Users\\andre\\Desktop\\USZ-001.png")
image9 = Image.open("C:\\Users\\andre\\Desktop\\CHB-001.png")

fig = plt.figure()#(figsize=(10, 7))

# Adds a subplot at the 1st position
fig.add_subplot(3, 3, 1)

# showing image
plt.imshow(image1)
plt.axis('off')
plt.title("(a) CHUP", y=-0.22)

fig.add_subplot(3, 3, 2)

plt.imshow(image2)
plt.axis('off')
plt.title("(b) CHUM", y=-0.22)

fig.add_subplot(3, 3, 3)

plt.imshow(image3)
plt.axis('off')
plt.title("(c) CHUV", y=-0.22)

fig.add_subplot(3, 3, 4)

plt.imshow(image4)
plt.axis('off')
plt.title("(d) CHUS", y=-0.22)

fig.add_subplot(3, 3, 5)

plt.imshow(image5)
plt.axis('off')
plt.title("(e) HGJ", y=-0.22)

fig.add_subplot(3, 3, 6)

plt.imshow(image6)
plt.axis('off')
plt.title("(f) HMR", y=-0.22)

fig.add_subplot(3, 3, 7)

plt.imshow(image7)
plt.axis('off')
plt.title("(g) MDA", y=-0.22)

fig.add_subplot(3, 3, 8)

plt.imshow(image8)
plt.axis('off')
plt.title("(h) USZ", y=-0.22)

fig.add_subplot(3, 3, 9)

plt.imshow(image9)
plt.axis('off')
plt.title("(i) CHB", y=-0.22)

#plt.tight_layout()
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.2   # the amount of height reserved for white space between subplots,
               # expressed as a fraction of the average axis height
#plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
#                wspace=wspace, hspace=hspace)
plt.subplots_adjust(wspace=0.1, hspace=0.2)

plt.show()
