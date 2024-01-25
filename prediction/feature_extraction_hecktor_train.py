from glob           import glob
from tqdm           import tqdm
from scipy.ndimage  import label, binary_dilation
from radiomics      import featureextractor

import ray
import SimpleITK    as sitk
import numpy        as np
import pandas       as pd


#PATH_IMGS = 'C:/Users/andre/Desktop/HECKTOR_DATA/hecktor2022_training/imagesTr/'
PATH_IMGS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_training/hecktor2022/imagesTr/'

#PATH_MASKS = 'C:/Users/andre/Desktop/HECKTOR_DATA/hecktor2022_training/labelsTr/'
PATH_MASKS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_training/hecktor2022/labelsTr/'

#PATH_MASK = 'C:/Users/andre/Desktop/HECKTOR_DATA/hecktor2022_testing/myorenko_labels'


patient_list = [x.split('__')[0].split('/')[-1] for x in glob(PATH_IMGS + '*PT.nii.gz')]  # change ('/') with ('\\')
len(patient_list)


#ray.shutdown()
#ray.init()


def read_nifti(path):
    """Read a NIfTI image. Return a SimpleITK Image."""
    nifti = sitk.ReadImage(str(path))
    return nifti


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
    """
    Resample a SimpleITK Image.

    Parameters
    ----------
    sitk_image : sitk.Image
        An input image.
    new_spacing : list of int
        A distance between adjacent voxels in each dimension given in physical units (mm) for the output image.
    new_size : list of int or None
        A number of pixels per dimension of the output image. If None, `new_size` is computed based on the original
        input size, original spacing and new spacing.
    attributes : dict or None
        The desired output image's spatial domain (its meta-data). If None, the original image's meta-data is used.
    interpolator
        Available interpolators:
            - sitk.sitkNearestNeighbor : nearest
            - sitk.sitkLinear : linear
            - sitk.sitkGaussian : gaussian
            - sitk.sitkLabelGaussian : label_gaussian
            - sitk.sitkBSpline : bspline
            - sitk.sitkHammingWindowedSinc : hamming_sinc
            - sitk.sitkCosineWindowedSinc : cosine_windowed_sinc
            - sitk.sitkWelchWindowedSinc : welch_windowed_sinc
            - sitk.sitkLanczosWindowedSinc : lanczos_windowed_sinc
    fill_value : int or float
        A value used for padding, if the output image size is less than `new_size`.

    Returns
    -------
    sitk.Image
        The resampled image.

    Notes
    -----
    This implementation is based on https://github.com/deepmedic/SimpleITK-examples/blob/master/examples/resample_isotropically.py
    """
    sitk_interpolator = interpolator

    # provided attributes:
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


def get_pad_amount(arr, target_shape):
    half_missing = (target_shape - arr.shape) / 2
    half_missing = half_missing.astype('int32')
    first_half = half_missing
    second_half = target_shape - (arr.shape + half_missing)
    pad_amount = tuple([(first_half[i], second_half[i]) for i in range(3)])
    return pad_amount


def worker_fn(patient):
    pt_path = PATH_IMGS + patient + '__PT.nii.gz'
    ct_path = PATH_IMGS + patient + '__CT.nii.gz'
    mask_path = PATH_MASKS + patient + '.nii.gz'

    pt_itk = sitk.ReadImage(pt_path)
    ct_itk = sitk.ReadImage(ct_path)
    mask_itk = sitk.ReadImage(mask_path)

    voxel_size = 1
    attributes_pt = get_attributes(pt_itk)
    pt = resample_sitk_image(pt_itk,
                             new_spacing=[voxel_size] * 3,
                             attributes=attributes_pt,
                             interpolator=sitk.sitkBSpline)
    pt = sitk.GetArrayFromImage(pt).astype(np.float32)
    pt = np.rot90(pt, 2)
    # pt_itk = sitk.GetImageFromArray(pt)
    # pt_itk.SetSpacing([voxel_size]*3)

    ct = resample_sitk_image(ct_itk,
                             new_spacing=[voxel_size] * 3,
                             attributes=attributes_pt,
                             interpolator=sitk.sitkBSpline)
    ct = sitk.GetArrayFromImage(ct).astype(np.float32)
    ct = np.rot90(ct, 2)
    # ct_itk = sitk.GetImageFromArray(ct)
    # ct_itk.SetSpacing([voxel_size]*3)

    mask = resample_sitk_image(mask_itk,
                               new_spacing=[voxel_size] * 3,
                               attributes=attributes_pt,
                               interpolator=sitk.sitkNearestNeighbor)
    mask = sitk.GetArrayFromImage(mask).astype(np.float32)
    mask = np.rot90(mask, 2)
    mask_itk = sitk.GetImageFromArray(mask)
    mask_itk.SetSpacing([voxel_size] * 3)

    imgs = {
        'CT': ct,
        'PT': pt,
    }
    row = {
        'patient': patient,
        'X_size': pt.shape[0],
        'Y_size': pt.shape[1],
        'Z_size': pt.shape[2],
    }
    all_masks = {}

    _, nb_lesions = label((mask == 1).astype('int8'))
    row['nb_lesions'] = nb_lesions

    _, nb_lymphnodes = label((mask == 2).astype('int8'))
    row['nb_lymphnodes'] = nb_lymphnodes

    # creating all masks

    # everything_merged
    all_masks['everything_merged'] = (mask != 0).astype('int32')

    # lesions_merged
    # all_masks['lesions_merged'] = (mask == 1).astype('int32')

    # lymphnodes_merged
    # all_masks['lymphnodes_merged'] = (mask == 2).astype('int32')

    base_masks = list(all_masks.keys())
    for masktype in base_masks:
        cmask = all_masks[masktype]

        # resegmented 2.5
        try:
            all_masks[masktype + '2.5'] = ((cmask != 0) & (pt > 2.5)).astype('int32')
        except:
            pass

        # resegmented 4
        try:
            all_masks[masktype + '4'] = ((cmask != 0) & (pt > 4)).astype('int32')
        except:
            pass

        # resegmented 40%
        try:
            suv_in_roi = pt[cmask == 1]
            suv_max = np.max(suv_in_roi)
            cutoff = suv_max * 0.4
            all_masks[masktype + '40%'] = ((cmask != 0) & (pt > cutoff)).astype('int32')
        except:
            pass

        # BBox
        try:
            x = np.any(cmask, axis=(1, 2))
            y = np.any(cmask, axis=(0, 2))
            z = np.any(cmask, axis=(0, 1))
            xmin, xmax = np.where(x)[0][[0, -1]]
            ymin, ymax = np.where(y)[0][[0, -1]]
            zmin, zmax = np.where(z)[0][[0, -1]]
            bbox = np.zeros(cmask.shape)
            bbox[xmin:xmax, ymin:ymax, zmin:zmax] = 1
            all_masks[masktype + 'BBox'] = bbox
        except:
            pass

        # shell 2mm
        try:
            all_masks[masktype + 'shell2mm'] = binary_dilation(cmask, iterations=2) - cmask
        except:
            pass

        # shell 4mm
        try:
            all_masks[masktype + 'shell4mm'] = binary_dilation(cmask, iterations=4) - cmask
        except:
            pass

        # shell 8mm
        try:
            all_masks[masktype + 'shell8mm'] = binary_dilation(cmask, iterations=8) - cmask
        except:
            pass

        for iterations in [1, 2, 4, 8, 16]:
            try:
                all_masks[masktype + 'dilat' + str(iterations) + 'mm'] = binary_dilation(cmask,
                                                                                         iterations=iterations).astype(
                    'int8')
                # for axis in range(3):
                #    plt.subplot(1,3,axis+1)
                #    plt.imshow(np.max(all_masks[masktype+'dilat'+str(iterations)+'mm'], axis=axis))
                # plt.show()
            except:
                pass

    # bbox croping
    bbox_mask = all_masks['everything_mergedBBox']
    shell_mask = all_masks['everything_mergedshell8mm']
    dilat_mask = all_masks['everything_mergeddilat16mm']

    x = np.any(bbox_mask, axis=(1, 2))
    y = np.any(bbox_mask, axis=(0, 2))
    z = np.any(bbox_mask, axis=(0, 1))
    xmin_bbox, xmax_bbox = np.where(x)[0][[0, -1]]
    ymin_bbox, ymax_bbox = np.where(y)[0][[0, -1]]
    zmin_bbox, zmax_bbox = np.where(z)[0][[0, -1]]

    x = np.any(shell_mask, axis=(1, 2))
    y = np.any(shell_mask, axis=(0, 2))
    z = np.any(shell_mask, axis=(0, 1))
    xmin_shell, xmax_shell = np.where(x)[0][[0, -1]]
    ymin_shell, ymax_shell = np.where(y)[0][[0, -1]]
    zmin_shell, zmax_shell = np.where(z)[0][[0, -1]]

    x = np.any(dilat_mask, axis=(1, 2))
    y = np.any(dilat_mask, axis=(0, 2))
    z = np.any(dilat_mask, axis=(0, 1))
    xmin_dilat, xmax_dilat = np.where(x)[0][[0, -1]]
    ymin_dilat, ymax_dilat = np.where(y)[0][[0, -1]]
    zmin_dilat, zmax_dilat = np.where(z)[0][[0, -1]]

    xmin, xmax = np.min([xmin_bbox, xmin_shell, xmin_dilat]), np.max([xmax_bbox, xmax_shell, xmax_dilat])
    ymin, ymax = np.min([ymin_bbox, ymin_shell, ymin_dilat]), np.max([ymax_bbox, ymax_shell, ymax_dilat])
    zmin, zmax = np.min([zmin_bbox, zmin_shell, zmin_dilat]), np.max([zmax_bbox, zmax_shell, zmax_dilat])

    pad = 2
    xmin, xmax = xmin - pad, xmax + pad
    ymin, ymax = ymin - pad, ymax + pad
    zmin, zmax = zmin - pad, zmax + pad

    for mask_name in all_masks.keys():
        cmask_arr = all_masks[mask_name]
        cmask_arr = cmask_arr[xmin:xmax, ymin:ymax, zmin:zmax]
        cmask_itk = sitk.GetImageFromArray(cmask_arr)
        cmask_itk.SetSpacing([voxel_size] * 3)
        all_masks[mask_name] = cmask_itk

    for modality in ['PT', 'CT']:
        img_arr = imgs[modality]
        img_arr = img_arr[xmin:xmax, ymin:ymax, zmin:zmax]
        img_itk = sitk.GetImageFromArray(img_arr)
        img_itk.SetSpacing([voxel_size] * 3)
        imgs[modality] = img_itk

    for mask_name in list(all_masks.keys()):
        for modality in ['PT', 'CT']:
            try:
                extractor = featureextractor.RadiomicsFeatureExtractor(
                    'pyradiomics_params_' + modality + '_no_resampling.yaml')
                features = extractor.execute(imgs[modality], all_masks[mask_name])
                for k in features:
                    if k[:12] != 'diagnostics_':
                        name = k.replace('original', modality)
                        if 'shape' in name and 'PT' in name:
                            name = name.replace('PT_', '')
                        name = mask_name + '_' + name
                        row[name] = features[k][()]
            except:
                pass

    return row


#@ray.remote
#def worker(patient):
#    try:
#        return worker_fn(patient)
#    except:
#        return None


all_results = []
for patient in tqdm(patient_list):
    all_results.append(worker_fn(patient))
    dfres = pd.DataFrame(all_results)
    dfres = dfres.rename(columns={"patient": 'PatientID'})
    dfres.to_csv('all_radiomics_hecktor_train.csv', index=False)
    dfres.to_csv('C:\\Users\\andre\\Desktop\\OUTCOME PREDICTION\\Radiomics\\all_radiomics_hecktor_train.csv', index=False)

# aa.exit
# all_results = []
# for patients in tqdm(np.array_split(patient_list, 20)):
#     workers = []
#     for patient in patients:
#         workers.append(worker.remote(patient))
#     all_results.append(ray.get(workers))


len(all_results)
#all_results = sum(all_results, [])

all_results_clean = [x for x in all_results if x is not None]
len(all_results_clean)

dfres = pd.DataFrame(all_results_clean)
dfres = dfres.rename(columns={"patient": 'PatientID'})
dfres.index = dfres['PatientID']

dfres.to_csv('all_features_hecktor_train_noNaN.csv')


"""feature extraction process stopped at 91%. Checked the missing patient 
and restarted the extraction process only for the remaining patients.

This is the part of the code added to feature_extraction_hecktor_train.py to check the remaining patient

PATH_IMGS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_training/hecktor2022/imagesTr/'

patient_list = [x.split('__')[0].split('/')[-1] for x in glob(PATH_IMGS + '*PT.nii.gz')] 

f = "/home/andrea/Codici/all_radiomics_v2_rem.csv"

pat_imported = pd.read_csv(f)
Ids = pat_imported.iloc[:, :1]
ids = Ids.sort_values(by=["PatientID"])
pat_list = pd.DataFrame(patient_list)
pat_list = pat_list.rename(columns={pat_list.columns[0]: 'PatientID'})

remaining_patient = pd.concat([pat_list, ids]).drop_duplicates(keep=False)

rem_pat = remaining_patient['PatientID'].tolist()


remaining_results = []
for patient in tqdm(rem_pat):
    remaining_results.append(worker_fn(patient))
    dfres = pd.DataFrame(remaining_results)
    dfres = dfres.rename(columns={"patient": 'PatientID'})
    dfres.to_csv('all_radiomics_v2_rem_remaining.csv', index=False)
"""

