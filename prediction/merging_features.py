import pandas as pd
from glob import glob

PATH_IMGS_TR  =  'C:\\Users\\andre\\Desktop\\HECKTOR_DATA\\hecktor2022_training\\imagesTr\\'
#PATH_IMGS_TR = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_training/hecktor2022/imagesTr/'

patient_list_tr = [x.split('__')[0].split('\\')[-1] for x in glob(PATH_IMGS_TR + '*PT.nii.gz')]
print(len(patient_list_tr))

path    = "C:\\Users\\andre\\Desktop\\OUTCOME PREDICTION\\"

rad     = path + "Radiomics\\all_radiomics_hecktor_train.csv"
clin    = path + "Clinical_infos\\hecktor2022_clinical_info_training.csv"
endp    = path + "Endpoints\\hecktor2022_endpoint_training.csv"
wb      = path + "Whole_body\\is_whole_body_hecktor_train.csv"

radiomics       = pd.read_csv(rad)
clinical_info   = pd.read_csv(clin)
end_point       = pd.read_csv(endp)
whole_body      = pd.read_csv(wb)
whole_body      = whole_body.drop(columns='Unnamed: 0')

features_no_end = radiomics.merge(clinical_info, on="PatientID")
feat            = features_no_end.merge(whole_body, on='PatientID')

features_all = feat.merge(end_point, on='PatientID')
features_all = features_all.sort_values(by='PatientID')
features_all.reset_index(drop=True, inplace=True)

dfres       = pd.DataFrame(features_all)
dfres       = dfres.rename(columns={"patient": 'PatientID'})
dfres.index = dfres['PatientID']

dfres.to_csv(path + "All_features\\all_features_hecktor_train.csv")

#%%

PATH_IMGS_TS  =  'C:\\Users\\andre\\Desktop\\HECKTOR_DATA\\hecktor2022_testing\\imagesTs\\'
#PATH_IMGS_TS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_testing/imagesTs/'

patient_list_ts = [x.split('__')[0].split('\\')[-1] for x in glob(PATH_IMGS_TS + '*PT.nii.gz')]
print(len(patient_list_ts))

path    = "C:\\Users\\andre\\Desktop\\OUTCOME PREDICTION\\"

rad     =  path + "Radiomics\\all_radiomics_hecktor_test.csv"
clin    =  path + "Clinical_infos\\hecktor2022_clinical_info_testing.csv"
endp    =  path + "Endpoints\\hecktor2022_endpoint_testing.csv"
wb      =  path + "Whole_body\\is_whole_body_hecktor_test.csv"

radiomics       = pd.read_csv(rad)
clinical_info   = pd.read_csv(clin)
end_point       = pd.read_csv(endp)
whole_body      = pd.read_csv(wb)
whole_body      = whole_body.drop(columns='Unnamed: 0')

features_no_end = radiomics.merge(clinical_info, on="PatientID")
feat            = features_no_end.merge(whole_body, on='PatientID')

features_all = feat.merge(end_point, on='PatientID')
features_all = features_all.sort_values(by='PatientID')
features_all.reset_index(drop=True, inplace=True)

dfres       = pd.DataFrame(features_all)
dfres       = dfres.rename(columns={"patient": 'PatientID'})
dfres.index = dfres['PatientID']

dfres.to_csv(path + "All_features\\all_features_hecktor_test.csv")

#%%


PATH_IMGS_PD  = 'C:\\Users\\andre\\Desktop\\DATI PADOVA\\Volums\\CT_nifti\\'
#PATH_IMGS_TS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_testing/imagesTs/'

patient_list_ts = [x.split('__')[0].split('\\')[-1] for x in glob(PATH_IMGS_PD + '*CT.nii.gz')]
print(len(patient_list_ts))

path    = "C:\\Users\\andre\\Desktop\\OUTCOME PREDICTION\\"

rad     =  path + "Radiomics\\all_radiomics_PD_CT_with_shape.csv"
clin    =  path + "Clinical_infos\\clinical_info_UNIPD_CT.csv"
endp    =  path + "Endpoints\\endpoint_PD_CT.csv"
wb      =  path + "Whole_body\\is_whole_body_PD_CT.csv"

radiomics       = pd.read_csv(rad)
clinical_info   = pd.read_csv(clin)
end_point       = pd.read_csv(endp)
whole_body      = pd.read_csv(wb)
whole_body      = whole_body.drop(columns='Unnamed: 0')

features_no_end = radiomics.merge(clinical_info, on="PatientID")
feat            = features_no_end.merge(whole_body, on='PatientID')

features_all = feat.merge(end_point, on='PatientID')
features_all = features_all.sort_values(by='PatientID')
features_all.reset_index(drop=True, inplace=True)

dfres       = pd.DataFrame(features_all)
dfres       = dfres.rename(columns={"patient": 'PatientID'})
dfres.index = dfres['PatientID']

dfres.to_csv(path + "All_features\\all_features_PD_CT_with_shape.csv")

#%%


PATH_IMGS_PD  = 'C:\\Users\\andre\\Desktop\\DATI PADOVA\\Volums\\CT_nifti\\'
#PATH_IMGS_TS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_testing/imagesTs/'

patient_list_ts = [x.split('__')[0].split('\\')[-1] for x in glob(PATH_IMGS_PD + '*CT.nii.gz')]
print(len(patient_list_ts))

path    = "C:\\Users\\andre\\Desktop\\OUTCOME PREDICTION\\"

rad     =  path + "Radiomics\\all_radiomics_PD_CT_only.csv"
clin    =  path + "Clinical_infos\\clinical_info_UNIPD_CT.csv"
endp    =  path + "Endpoints\\endpoint_PD_CT.csv"
wb      =  path + "Whole_body\\is_whole_body_PD_CT.csv"

radiomics       = pd.read_csv(rad)
clinical_info   = pd.read_csv(clin)
end_point       = pd.read_csv(endp)
whole_body      = pd.read_csv(wb)
whole_body      = whole_body.drop(columns='Unnamed: 0')

features_no_end = radiomics.merge(clinical_info, on="PatientID")
feat            = features_no_end.merge(whole_body, on='PatientID')

features_all = feat.merge(end_point, on='PatientID')
features_all = features_all.sort_values(by='PatientID')
features_all.reset_index(drop=True, inplace=True)

dfres       = pd.DataFrame(features_all)
dfres       = dfres.rename(columns={"patient": 'PatientID'})
dfres.index = dfres['PatientID']

dfres.to_csv(path + "All_features\\all_features_PD_CT_only.csv")

#%%

#%%


PATH_IMGS_PD  = 'C:\\Users\\andre\\Desktop\\DATI PADOVA\\Volums\\CT_nifti\\'
#PATH_IMGS_TS = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/hecktor2022_testing/imagesTs/'

patient_list_ts = [x.split('__')[0].split('\\')[-1] for x in glob(PATH_IMGS_PD + '*CT.nii.gz')]
print(len(patient_list_ts))

path    = "C:\\Users\\andre\\Desktop\\OUTCOME PREDICTION\\"

rad     =  path + "Radiomics\\all_radiomics_PD_CT_only_shape.csv"
clin    =  path + "Clinical_infos\\clinical_info_UNIPD_CT.csv"
endp    =  path + "Endpoints\\endpoint_PD_CT.csv"
wb      =  path + "Whole_body\\is_whole_body_PD_CT.csv"

radiomics       = pd.read_csv(rad)
clinical_info   = pd.read_csv(clin)
end_point       = pd.read_csv(endp)
whole_body      = pd.read_csv(wb)
whole_body      = whole_body.drop(columns='Unnamed: 0')

features_no_end = radiomics.merge(clinical_info, on="PatientID")
feat            = features_no_end.merge(whole_body, on='PatientID')

features_all = feat.merge(end_point, on='PatientID')
features_all = features_all.sort_values(by='PatientID')
features_all.reset_index(drop=True, inplace=True)

dfres       = pd.DataFrame(features_all)
dfres       = dfres.rename(columns={"patient": 'PatientID'})
dfres.index = dfres['PatientID']

dfres.to_csv(path + "All_features\\all_features_PD_CT_only_shape.csv")
