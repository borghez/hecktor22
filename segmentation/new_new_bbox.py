from glob           import glob
from tqdm           import tqdm
from scipy.ndimage  import label, binary_dilation

import SimpleITK    as sitk
import numpy        as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import warnings


def write_nifti(sitk_img, path):
    """Save a SimpleITK Image to disk in NIfTI format."""
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(str(path))
    writer.Execute(sitk_img)


def get_attributes(sitk_image):
    """Get physical space attributes (meta-data) of the image."""
    attributes = {}
    attributes['orig_pixelid'] = sitk_image.GetPixelIDValue()
    attributes['orig_origin'] = sitk_image.GetOrigin()
    attributes['orig_direction'] = sitk_image.GetDirection()
    attributes['orig_spacing'] = np.array(sitk_image.GetSpacing())
    attributes['orig_size'] = np.array(sitk_image.GetSize(), dtype=int)
    return attributes


def resample_sitk_image(sitk_image,
                        new_spacing=[1, 1, 1],
                        new_size=None,
                        attributes=None,
                        interpolator=sitk.sitkLinear,
                        fill_value=0):

    sitk_interpolator = interpolator

    #provided attributes:
    if attributes:
        orig_pixelid = attributes['orig_pixelid']
        orig_origin = attributes['orig_origin']
        orig_direction = attributes['orig_direction']
        orig_spacing = attributes['orig_spacing']
        orig_size = attributes['orig_size']

    else:
        # use original attributes:
        orig_pixelid = sitk_image.GetPixelIDValue()
        orig_origin = sitk_image.GetOrigin()
        orig_direction = sitk_image.GetDirection()
        orig_spacing = np.array(sitk_image.GetSpacing())
        orig_size = np.array(sitk_image.GetSize(), dtype=int)

    # new image size:
    if not new_size:
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetOutputOrigin(orig_origin)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetDefaultPixelValue(fill_value)
    resample_filter.SetOutputPixelType(orig_pixelid)

    resampled_sitk_image = resample_filter.Execute(sitk_image)

    return resampled_sitk_image


'''train'''

PATH_IMGS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_training/hecktor2022/imagesTr/'
PATH_MASKS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_training/hecktor2022/labelsTr/'

patient_list = [x.split('__')[0].split('/')[-1] for x in glob(PATH_IMGS + '*PT.nii.gz')]  # change ('/') with ('\\')
len(patient_list)

box_size = [200, 200, 310]
box_size = np.array(box_size)

voxel_size = 1

for patient in tqdm(patient_list):
    pt_path = PATH_IMGS + patient + '__PT.nii.gz'
    ct_path = PATH_IMGS + patient + '__CT.nii.gz'
    mask_path = PATH_MASKS + patient + '.nii.gz'

    pt_itk = sitk.ReadImage(pt_path)
    ct_itk = sitk.ReadImage(ct_path)
    mask_itk = sitk.ReadImage(mask_path)

    attributes_pt = get_attributes(pt_itk)

    orig_origin = pt_itk.GetOrigin()
    orig_direction = pt_itk.GetDirection()
    orig_spacing = np.array(pt_itk.GetSpacing())


    pt_res = resample_sitk_image(pt_itk,
                                 new_spacing=[voxel_size] * 3,
                                 attributes=attributes_pt,
                                 interpolator=sitk.sitkBSpline)
    pt = sitk.GetArrayFromImage(pt_res).astype(np.float32)
    pt = np.transpose(pt, (2, 1, 0))


    ct_res = resample_sitk_image(ct_itk,
                                 new_spacing=[voxel_size] * 3,
                                 attributes=attributes_pt,
                                 interpolator=sitk.sitkBSpline)
    ct = sitk.GetArrayFromImage(ct_res).astype(np.float32)
    ct = np.transpose(ct, (2, 1, 0))


    mask_res = resample_sitk_image(mask_itk,
                                   new_spacing=[voxel_size] * 3,
                                   attributes=attributes_pt,
                                   interpolator=sitk.sitkNearestNeighbor)
    mask = sitk.GetArrayFromImage(mask_res).astype(np.float32)
    mask = np.transpose(mask, (2, 1, 0))

    before_sum = mask.sum().item()

    crop_len = int(0.75 * pt.shape[2])
    im = pt[..., crop_len:]

    msk = ((im - im.mean()) / im.std()) > 1
    comp_idx = np.argwhere(msk)

    center = np.mean(comp_idx, axis=0)
    xmin = np.min(comp_idx, axis=0)
    xmax = np.max(comp_idx, axis=0)

    xmin[:2] = center[:2] - box_size[:2] // 2
    xmax[:2] = center[:2] + box_size[:2] // 2

    xmax[2] = xmax[2] + crop_len
    xmin[2] = max(0, xmax[2] - box_size[2])

    box_start = xmin.astype(int)
    box_end = xmax.astype(int)

    pt = pt[box_start[0]: box_end[0], box_start[1]: box_end[1], box_start[2]: box_end[2]]
    pt_itk_bb = sitk.GetImageFromArray(np.transpose(pt, (2, 1, 0)))
    #pt_itk_bb.SetDirection(orig_direction)
    #pt_itk_bb.SetOrigin(orig_origin)
    #pt_itk_bb.SetSpacing(orig_spacing)
    writer_pt = sitk.ImageFileWriter()
    writer_pt.SetFileName('/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_pet/' + patient + '__PT.nii.gz')
    writer_pt.Execute(pt_itk_bb)

    ct = ct[box_start[0]: box_end[0], box_start[1]: box_end[1], box_start[2]: box_end[2]]
    ct_itk_bb = sitk.GetImageFromArray(np.transpose(ct, (2, 1, 0)))
    #ct_itk_bb.SetDirection(orig_direction)
    #ct_itk_bb.SetOrigin(orig_origin)
    #ct_itk_bb.SetSpacing(orig_spacing)
    writer_ct = sitk.ImageFileWriter()
    writer_ct.SetFileName('/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_ct/' + patient + '__CT.nii.gz')
    writer_ct.Execute(ct_itk_bb)

    mask = mask[box_start[0]: box_end[0], box_start[1]: box_end[1], box_start[2]: box_end[2]]
    mask_itk_bb = sitk.GetImageFromArray(np.transpose(mask, (2, 1, 0)))
    #mask_itk_bb.SetDirection(orig_direction)
    #mask_itk_bb.SetOrigin(orig_origin)
    #mask_itk_bb.SetSpacing(orig_spacing)
    writer_pt = sitk.ImageFileWriter()
    writer_pt.SetFileName('/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/train_label/' + patient + '.nii.gz')
    writer_pt.Execute(mask_itk_bb)

    after_sum = mask.sum().item()

    if before_sum != after_sum:
        warnings.warn("WARNING, H&N crop could be incorrect!!! ")

# '''test'''
PATH_IMGS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_testing/imagesTs/'
#PATH_IMGS = 'C:\\Users\\andre\\Desktop\\HECKTOR_DATA_SEGM\\test_PET\\'
#PATH_IMGSct = 'C:\\Users\\andre\\Desktop\\HECKTOR_DATA_SEGM\\test_CT\\'


PATH_MASKS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_testing_gt/labelsTs/'
#PATH_MASKS = 'C:\\Users\\andre\\Desktop\\HECKTOR_DATA\\hecktor2022_testing\\labelsTs\\'
#
#
patient_list = [x.split('__')[0].split('/')[-1] for x in glob(PATH_IMGS + '*PT.nii.gz')]  # change ('/') with ('\\')
len(patient_list)

#th = 0.01
box_size = [200, 200, 310]
box_size = np.array(box_size)

for patient in tqdm(patient_list):
    pt_path = PATH_IMGS + patient + '__PT.nii.gz'
    ct_path = PATH_IMGS + patient + '__CT.nii.gz'
    mask_path = PATH_MASKS + patient + '.nii.gz'

    pt_itk = sitk.ReadImage(pt_path)
    ct_itk = sitk.ReadImage(ct_path)
    mask_itk = sitk.ReadImage(mask_path)

    attributes_pt = get_attributes(pt_itk)

    orig_origin = pt_itk.GetOrigin()
    orig_direction = pt_itk.GetDirection()
    orig_spacing = np.array(pt_itk.GetSpacing())

    pt_res = resample_sitk_image(pt_itk,
                                 new_spacing=[voxel_size] * 3,
                                 attributes=attributes_pt,
                                 interpolator=sitk.sitkBSpline)
    pt = sitk.GetArrayFromImage(pt_res).astype(np.float32)
    pt = np.transpose(pt, (2, 1, 0))
    #pt = np.rot90(pt, 2)
    #pt = pt.T

    ct_res = resample_sitk_image(ct_itk,
                                 new_spacing=[voxel_size] * 3,
                                 attributes=attributes_pt,
                                 interpolator=sitk.sitkBSpline)
    ct = sitk.GetArrayFromImage(ct_res).astype(np.float32)
    ct = np.transpose(ct, (2, 1, 0))
    #ct = np.rot90(ct, 2)
    #ct = ct.T

    mask_res = resample_sitk_image(mask_itk,
                                   new_spacing=[voxel_size] * 3,
                                   attributes=attributes_pt,
                                   interpolator=sitk.sitkNearestNeighbor)
    mask = sitk.GetArrayFromImage(mask_res).astype(np.float32)
    mask = np.transpose(mask, (2, 1, 0))
    #mask = np.rot90(mask, 2)
    #mask = mask.T

    #pt = (pt - np.min(pt)) / (np.max(pt) - np.min(pt))
    #pt = np.divide(np.subtract(pt, np.nanmean(pt)), np.std(pt))
    # ct = (ct-np.nanmean(ct))/np.std(ct)
    # mask = (mask-np.nanmean(mask))/np.std(mask)
    #box_size = (box_size / np.array(pt_res.pixdim)).astype(int)

    crop_len = int(0.75 * pt.shape[2])
    im = pt[..., crop_len:]

    msk = ((im - im.mean()) / im.std()) > 1
    comp_idx = np.argwhere(msk)

    center = np.mean(comp_idx, axis=0)
    xmin = np.min(comp_idx, axis=0)
    xmax = np.max(comp_idx, axis=0)

    xmin[:2] = center[:2] - box_size[:2] // 2
    xmax[:2] = center[:2] + box_size[:2] // 2

    xmax[2] = xmax[2] + crop_len
    xmin[2] = max(0, xmax[2] - box_size[2])

    box_start = xmin.astype(int)
    box_end = xmax.astype(int)

    # pt_x_mean = np.mean(pt, axis=(1, 2))
    # pt_y_mean = np.mean(pt, axis=(0, 2))
    # pt_z_mean = np.mean(pt, axis=(0, 1))
    #
    # np_x_max = np.where(pt_x_mean == np.max(pt_x_mean))[0][0]
    # np_y_max = np.where(pt_y_mean == np.max(pt_y_mean))[0][0]
    # np_z_max = np.where(pt_z_mean == np.max(pt_z_mean))[0][0]
    #
    # xmin = max(0, int(np_x_max-150))
    # ymin = max(0, int(np_y_max-100))
    # zmin = max(0, int(np_z_max-100))
    #
    # xmax = min(int(np_x_max + 150), pt.shape[0])
    # ymax = min(int(np_y_max + 100), pt.shape[1])
    # zmax = min(int(np_z_max + 100), pt.shape[2])

    # xmin = max(0, int((np.where(pt_x_mean > th)[0][0])/1.5))
    # ymin = max(0, int((np.where(pt_y_mean > th)[0][0])/1.2))
    # zmin = max(0, int((np.where(pt_z_mean > th)[0][0])/1.2))
    #
    # xmax = min(int((np.where(pt_x_mean[int((xmin*1.5))::] < th)[0][1])*1.5), pt.shape[0])
    # ymax = min(int((np.where(pt_y_mean > th)[0][-1])*1.2), pt.shape[1])
    # zmax = min(int((np.where(pt_z_mean > th)[0][-1])*1.2), pt.shape[2])

    # if pt.shape[2] > x_max and pt.shape[1] > y_max and pt.shape[0] > z_max:
    #     xmax, ymax, zmax = x_max, y_max, z_max
    # else:
    #     xmax = pt.shape[2]
    #     ymax = pt.shape[1]
    #     zmax = pt.shape[0]

    #pt = pt[xmin:xmax, ymin:ymax, zmin:zmax]
    pt = pt[box_start[0]: box_end[0], box_start[1]: box_end[1], box_start[2]: box_end[2]]
    pt_itk_bb = sitk.GetImageFromArray(np.transpose(pt, (2, 1, 0)))
    #pt_itk_bb.SetDirection(orig_direction)
    #pt_itk_bb.SetOrigin(orig_origin)
    #pt_itk_bb.SetSpacing(orig_spacing)
    writer_pt = sitk.ImageFileWriter()
    writer_pt.SetFileName('/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/test_pet/' + patient + '__PT.nii.gz')
    writer_pt.Execute(pt_itk_bb)

    # ct = ct[xmin:xmax, ymin:ymax, zmin:zmax]
    ct = ct[box_start[0]: box_end[0], box_start[1]: box_end[1], box_start[2]: box_end[2]]
    ct_itk_bb = sitk.GetImageFromArray(np.transpose(ct, (2, 1, 0)))
    #ct_itk_bb.SetDirection(orig_direction)
    #ct_itk_bb.SetOrigin(orig_origin)
    #ct_itk_bb.SetSpacing(orig_spacing)
    writer_ct = sitk.ImageFileWriter()
    writer_ct.SetFileName('/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/test_ct/' + patient + '__CT.nii.gz')
    writer_ct.Execute(ct_itk_bb)
    #
    # mask = mask[xmin:xmax, ymin:ymax, zmin:zmax]
    mask = mask[box_start[0]: box_end[0], box_start[1]: box_end[1], box_start[2]: box_end[2]]
    mask_itk_bb = sitk.GetImageFromArray(np.transpose(mask, (2, 1, 0)))
    #mask_itk_bb.SetDirection(orig_direction)
    #mask_itk_bb.SetOrigin(orig_origin)
    #mask_itk_bb.SetSpacing(orig_spacing)
    writer_pt = sitk.ImageFileWriter()
    writer_pt.SetFileName('/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/test_label/' + patient + '.nii.gz')
    writer_pt.Execute(mask_itk_bb)


#
#
# '''train'''
# PATH_IMGS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_training/hecktor2022/imagesTr/'
# PATH_MASKS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_training/hecktor2022/labelsTr/'
#
# '''test'''
# #PATH_IMGS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_testing/imagesTs/'
# #PATH_MASKS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_testing_gt/labelsTs/'
#
#
# patient_list = [x.split('__')[0].split('/')[-1] for x in glob(PATH_IMGS + '*PT.nii.gz')]  # change ('/') with ('\\')
# len(patient_list)
# #
#
#
#
# def get_pad_amount(arr, target_shape):
#     half_missing = (target_shape - arr.shape) / 2
#     half_missing = half_missing.astype('int32')
#     first_half = half_missing
#     second_half = target_shape - (arr.shape + half_missing)
#     pad_amount = tuple([(first_half[i], second_half[i]) for i in range(3)])
#     return pad_amount
#
#
# voxel_size = 1
#
#
# x_min, x_max = [], []
# y_min, y_max = [], []
# z_min, z_max = [], []
# x_diff, y_diff, z_diff = [], [], []

# for patient in tqdm(patient_list[::4]):
#     pt_path = PATH_IMGS + patient + '__PT.nii.gz'
#     ct_path = PATH_IMGS + patient + '__CT.nii.gz'
#     mask_path = PATH_MASKS + patient + '.nii.gz'
#
#     pt_itk = sitk.ReadImage(pt_path)
#     ct_itk = sitk.ReadImage(ct_path)
#     mask_itk = sitk.ReadImage(mask_path)
#
#     attributes_pt = get_attributes(pt_itk)
#
#     pt_res = resample_sitk_image(pt_itk,
#                                  new_spacing=[voxel_size] * 3,
#                                  attributes=attributes_pt,
#                                  interpolator=sitk.sitkBSpline)
#     pt = sitk.GetArrayFromImage(pt_res).astype(np.float32)
#     pt = np.rot90(pt, 2)
#
#     ct_res = resample_sitk_image(ct_itk,
#                                  new_spacing=[voxel_size] * 3,
#                                  attributes=attributes_pt,
#                                  interpolator=sitk.sitkBSpline)
#     ct = sitk.GetArrayFromImage(ct_res).astype(np.float32)
#     ct = np.rot90(ct, 2)
#
#     mask_res = resample_sitk_image(mask_itk,
#                                    new_spacing=[voxel_size] * 3,
#                                    attributes=attributes_pt,
#                                    interpolator=sitk.sitkNearestNeighbor)
#     mask = sitk.GetArrayFromImage(mask_res).astype(np.float32)
#     mask = np.rot90(mask, 2)
#
#     row = {
#         'patient': patient,
#         'X_size': pt.shape[0],
#         'Y_size': pt.shape[1],
#         'Z_size': pt.shape[2],
#     }
#
#     all_masks = {}
#
#     _, nb_lesions = label((mask == 1).astype('int8'))
#     row['nb_lesions'] = nb_lesions
#
#     _, nb_lymphnodes = label((mask == 2).astype('int8'))
#     row['nb_lymphnodes'] = nb_lymphnodes
#
#     # everything_merged
#     all_masks['everything_merged'] = (mask != 0).astype('int32')
#
#     base_masks = list(all_masks.keys())
#     for masktype in base_masks:
#         cmask = all_masks[masktype]
#
#         # resegmented 2.5
#         try:
#             all_masks[masktype + '2.5'] = ((cmask != 0) & (pt > 2.5)).astype('int32')
#         except:
#             pass
#
#         # resegmented 4
#         try:
#             all_masks[masktype + '4'] = ((cmask != 0) & (pt > 4)).astype('int32')
#         except:
#             pass
#
#         # resegmented 40%
#         try:
#             suv_in_roi = pt[cmask == 1]
#             suv_max = np.max(suv_in_roi)
#             cutoff = suv_max * 0.4
#             all_masks[masktype + '40%'] = ((cmask != 0) & (pt > cutoff)).astype('int32')
#         except:
#             pass
#
#         # BBox
#         try:
#             x = np.any(cmask, axis=(1, 2))
#             y = np.any(cmask, axis=(0, 2))
#             z = np.any(cmask, axis=(0, 1))
#             xmin, xmax = np.where(x)[0][[0, -1]]
#             ymin, ymax = np.where(y)[0][[0, -1]]
#             zmin, zmax = np.where(z)[0][[0, -1]]
#             bbox = np.zeros(cmask.shape)
#             bbox[xmin:xmax, ymin:ymax, zmin:zmax] = 1
#             all_masks[masktype + 'BBox'] = bbox
#         except:
#             pass
#
#         # shell 2mm
#         try:
#             all_masks[masktype + 'shell2mm'] = binary_dilation(cmask, iterations=2) - cmask
#         except:
#             pass
#
#         # shell 4mm
#         try:
#             all_masks[masktype + 'shell4mm'] = binary_dilation(cmask, iterations=4) - cmask
#         except:
#             pass
#
#         # shell 8mm
#         try:
#             all_masks[masktype + 'shell8mm'] = binary_dilation(cmask, iterations=8) - cmask
#         except:
#             pass
#
#         for iterations in [1, 2, 4, 8, 16]:
#             try:
#                 all_masks[masktype + 'dilat' + str(iterations) + 'mm'] = binary_dilation(cmask,
#                                                                                          iterations=iterations).astype(
#                     'int8')
#                 # for axis in range(3):
#                 #    plt.subplot(1,3,axis+1)
#                 #    plt.imshow(np.max(all_masks[masktype+'dilat'+str(iterations)+'mm'], axis=axis))
#                 # plt.show()
#             except:
#                 pass
#
#     # bbox croping
#     bbox_mask = all_masks['everything_mergedBBox']
#     shell_mask = all_masks['everything_mergedshell8mm']
#     dilat_mask = all_masks['everything_mergeddilat16mm']
#
#     x = np.any(bbox_mask, axis=(1, 2))
#     y = np.any(bbox_mask, axis=(0, 2))
#     z = np.any(bbox_mask, axis=(0, 1))
#     xmin_bbox, xmax_bbox = np.where(x)[0][[0, -1]]
#     ymin_bbox, ymax_bbox = np.where(y)[0][[0, -1]]
#     zmin_bbox, zmax_bbox = np.where(z)[0][[0, -1]]
#
#     x = np.any(shell_mask, axis=(1, 2))
#     y = np.any(shell_mask, axis=(0, 2))
#     z = np.any(shell_mask, axis=(0, 1))
#     xmin_shell, xmax_shell = np.where(x)[0][[0, -1]]
#     ymin_shell, ymax_shell = np.where(y)[0][[0, -1]]
#     zmin_shell, zmax_shell = np.where(z)[0][[0, -1]]
#
#     x = np.any(dilat_mask, axis=(1, 2))
#     y = np.any(dilat_mask, axis=(0, 2))
#     z = np.any(dilat_mask, axis=(0, 1))
#     xmin_dilat, xmax_dilat = np.where(x)[0][[0, -1]]
#     ymin_dilat, ymax_dilat = np.where(y)[0][[0, -1]]
#     zmin_dilat, zmax_dilat = np.where(z)[0][[0, -1]]
#
#     xmin, xmax = np.min([xmin_bbox, xmin_shell, xmin_dilat]), np.max([xmax_bbox, xmax_shell, xmax_dilat])
#     ymin, ymax = np.min([ymin_bbox, ymin_shell, ymin_dilat]), np.max([ymax_bbox, ymax_shell, ymax_dilat])
#     zmin, zmax = np.min([zmin_bbox, zmin_shell, zmin_dilat]), np.max([zmax_bbox, zmax_shell, zmax_dilat])
#
#     pad = 2
#     xmin, xmax = xmin - pad, xmax + pad
#     ymin, ymax = ymin - pad, ymax + pad
#     zmin, zmax = zmin - pad, zmax + pad
#
#     xdiff = xmax - xmin
#     ydiff = ymax - ymin
#     zdiff = zmax - zmin
#
#     x_min.append(xmin)
#     x_max.append(xmax)
#     y_min.append(ymin)
#     y_max.append(ymax)
#     z_min.append(zmin)
#     z_max.append(zmax)
#     x_diff.append(xdiff)
#     y_diff.append(ydiff)
#     z_diff.append(zdiff)
# #

# xmin = int(np.nanmean(x_min))
# ymin = int(np.nanmean(y_min))
# zmin = int(np.nanmean(z_min))
# #
# maxxdiff = np.max(x_diff)
# maxydiff = np.max(y_diff)
# maxzdiff = np.max(z_diff)
# #
# # # x_max = int(np.nanmean(x_max)) #se non va cambia con max
# # # y_max = int(np.nanmean(y_max))
# # # z_max = int(np.nanmean(z_max))
# #
# x_max = int((xmin + maxxdiff)*2)
# y_max = int((ymin + maxydiff)*2)
# z_max = int((zmin + maxzdiff)*2)
#
# # # x_max = int(np.nanmean(x_max))
# # # y_max = int(np.nanmean(y_max))
# # # z_max = int(np.nanmean(z_max))

