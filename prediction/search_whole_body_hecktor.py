import SimpleITK    as sitk
import numpy        as np
import pandas as pd
from matplotlib import pyplot as plt

from tqdm           import tqdm
from glob           import glob


#PATH_IMGS_TR = 'C:/Users/andre/Desktop/HECKTOR_DATA/hecktor2022_training/imagesTr/'
PATH_IMGS_TR = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_training/hecktor2022/imagesTr/'

#PATH_IMGS_TS = 'C:/Users/andre/Desktop/HECKTOR_DATA/hecktor2022_testing/imagesTs/'
PATH_IMGS_TS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_testing/imagesTs/'


patient_list_tr = [x.split('__')[0].split('/')[-1] for x in glob(PATH_IMGS_TR + '*PT.nii.gz')]
print(len(patient_list_tr))

patient_list_ts = [x.split('__')[0].split('/')[-1] for x in glob(PATH_IMGS_TS + '*PT.nii.gz')]
print(len(patient_list_ts))


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


def find_whole_body_tr(patient, thresh):
    pt_path_tr = PATH_IMGS_TR + patient + '__PT.nii.gz'

    pt_itk = sitk.ReadImage(pt_path_tr)

    voxel_size = 1
    attributes_pt = get_attributes(pt_itk)
    pt = resample_sitk_image(pt_itk,
                             new_spacing=[voxel_size] * 3,
                             attributes=attributes_pt,
                             interpolator=sitk.sitkBSpline)
    pt = sitk.GetArrayFromImage(pt).astype(np.float32)
    #pt = np.rot90(pt, 2)

    if pt.shape[0] > thresh:
        is_whole_body = 1
    else:
        is_whole_body = 0

    return is_whole_body, pt.shape

def find_whole_body_ts(patient, thresh):
    pt_path_ts = PATH_IMGS_TS + patient + '__PT.nii.gz'

    pt_itk = sitk.ReadImage(pt_path_ts)

    voxel_size = 1
    attributes_pt = get_attributes(pt_itk)
    pt = resample_sitk_image(pt_itk,
                             new_spacing=[voxel_size] * 3,
                             attributes=attributes_pt,
                             interpolator=sitk.sitkBSpline)
    pt = sitk.GetArrayFromImage(pt).astype(np.float32)
    #pt = np.rot90(pt, 2)

    if pt.shape[0] > thresh:
        is_whole_body = 1
    else:
        is_whole_body = 0

    return is_whole_body, pt.shape


is_wb_tr = []
is_wb_ts = []
shape_tr = []
shape_ts = []
dim1 = []
dim2 = []
dim3 = []
dim1_ts = []
dim2_ts = []
dim3_ts = []

#np.zeros(len(patient_list_ts))

thresh = 400

for patient in tqdm(patient_list_tr):
    is_wb, shape_tr = find_whole_body_tr(patient, thresh)
    is_wb_tr.append(is_wb)
    dim1.append(shape_tr[0])
    dim2.append(shape_tr[1])
    dim3.append(shape_tr[2])

is_whole_body_tr = np.array(is_wb_tr)

numb_slices_tr = {'PatientID': patient_list_tr,
                  'whole_body_scan': is_whole_body_tr}

whole_body_tr = pd.DataFrame(numb_slices_tr)

pd.DataFrame(whole_body_tr).to_csv('Outputs/is_whole_body_tr.csv')


for patient in tqdm(patient_list_ts):
    is_wb, shape_ts = find_whole_body_ts(patient, thresh)
    is_wb_ts.append(is_wb)
    dim1_ts.append(shape_ts[0])
    dim2_ts.append(shape_ts[1])
    dim3_ts.append(shape_ts[2])

is_whole_body_ts = np.array(is_wb_ts)

numb_slices_ts = {'PatientID': patient_list_ts,
                  'whole_body_scan': is_whole_body_ts}

whole_body_ts = pd.DataFrame(numb_slices_ts)

pd.DataFrame(whole_body_ts).to_csv('Outputs/is_whole_body_ts.csv')



#%%

# #tot_numb_slices_tr_path = 'C:\\Users\\andre\\Desktop\\FEATURES_REBAUD\\dim1_tr.csv'
# tot_numb_slices_tr_path = 'dim1_tr.csv'
# #tot_numb_slices_ts_path = 'C:\\Users\\andre\\Desktop\\FEATURES_REBAUD\\dim1_ts.csv'
# tot_numb_slices_ts_path = 'dim1_ts.csv'
#
# tot_numb_slices_tr = pd.read_csv(tot_numb_slices_tr_path, index_col=0)
#
# tot_numb_slices_ts = pd.read_csv(tot_numb_slices_ts_path, index_col=0)
# tot_numb_slices_ts = np.array(tot_numb_slices_ts)
#
# is_whole_body_ts = []
#
#
# is_whole_body_tr = pd.read_csv('is_whole_body_tr.csv', index_col=0)
# is_whole_body_tr = np.array(is_whole_body_tr)
#
# for pat in range(len(tot_numb_slices_ts)):
#     if tot_numb_slices_ts[pat, 0] > thresh:
#         is_whole_body_ts.append(1)
#     else:
#         is_whole_body_ts.append(0)
#
# is_whole_body_ts = np.array(is_whole_body_ts)
#
# numb_slices_tr = {'PatientID': patient_list_tr,
#                   'whole_body_scan': is_whole_body_tr[:, 0]}
# whole_body_tr = pd.DataFrame(numb_slices_tr)
#
# numb_slices_ts = {'PatientID': patient_list_ts,
#                   'whole_body_scan': is_whole_body_ts}
# whole_body_ts = pd.DataFrame(numb_slices_ts)
#
# pd.DataFrame(whole_body_tr).to_csv('C:\\Users\\andre\\Desktop\\FEATURES_REBAUD\\is_whole_body_tr.csv')
# pd.DataFrame(whole_body_ts).to_csv('C:\\Users\\andre\\Desktop\\FEATURES_REBAUD\\is_whole_body_ts.csv')
#
#
#
# fig = plt.figure(figsize=(10, 10))
# plt.hist(tot_numb_slices_tr, 15)
# plt.show()
#
# fig1 = plt.figure(figsize=(10, 10))
# plt.hist(tot_numb_slices_ts, 15)
# plt.show()
