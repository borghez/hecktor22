from scipy.stats import shapiro
import pandas as pd
import scipy.stats as stats

segresnet = pd.read_csv("C:\\Users\\andre\\Desktop\\dices_ct_segresnet.csv", squeeze=True)#.fillna(0)
swinunetr = pd.read_csv("C:\\Users\\andre\\Desktop\\dices_ct_swinunetr.csv", squeeze=True)#.fillna(0)
unet = pd.read_csv("C:\\Users\\andre\\Desktop\\dices_ct_unet.csv", squeeze=True)#.fillna(0)
unetr = pd.read_csv("C:\\Users\\andre\\Desktop\\dices_ct_unetr.csv", squeeze=True)#.fillna(0)

#segresnet= pd.read_csv("/home/andrea/dices_ct_segresnet.csv")

#print(f': {"Not Gaussian" if shapiro(segresnet)[1]<0.05 else "Gaussian"}  {shapiro(segresnet)}') #p_vale<0.5 NOT gauss
#print(f': {"Not Gaussian" if shapiro(swinunetr)[1]<0.05 else "Gaussian"}  {shapiro(swinunetr)}')
#print(f': {"Not Gaussian" if shapiro(unet)[1]<0.05 else "Gaussian"}  {shapiro(swinunetr)}')
#print(f': {"Not Gaussian" if shapiro(unetr)[1]<0.05 else "Gaussian"}  {shapiro(swinunetr)}')

#res_wilcoxon = stats.wilcoxon(segresnet, swinunetr)
#res_wilcoxon.statistic, res_wilcoxon.pvalue

print(stats.wilcoxon(segresnet, swinunetr, nan_policy='omit'))  #p_value<0.5  significant difference  no  0.0555
print(stats.wilcoxon(segresnet, unet, nan_policy='omit'))  #p_value<0.5  significant difference   no   0.1056
print(stats.wilcoxon(segresnet, unetr, nan_policy='omit'))  #p_value<0.5  significant difference   si   0.0003

pet_ct_segresnet = pd.read_csv("C:\\Users\\andre\\Desktop\\dices_pet_ct_segresnet.csv", squeeze=True)#.fillna(0)
pet_ct_swinunetr = pd.read_csv("C:\\Users\\andre\\Desktop\\dices_pet_ct_swinunetr.csv", squeeze=True)#.fillna(0)
print(stats.wilcoxon(pet_ct_segresnet, pet_ct_swinunetr, nan_policy='omit'))    #si   1.8511082914692608e-08

print(stats.wilcoxon(pet_ct_segresnet, segresnet, nan_policy='omit'))    #si  1.491811383668924e-31
print(stats.wilcoxon(pet_ct_segresnet, swinunetr, nan_policy='omit'))   #1.6791930555400654e-43
print(stats.wilcoxon(pet_ct_segresnet, unet, nan_policy='omit'))   #4.1215381231535863e-42
print(stats.wilcoxon(pet_ct_segresnet, unetr, nan_policy='omit'))   #2.8331745134241104e-51

print(stats.wilcoxon(pet_ct_swinunetr, segresnet, nan_policy='omit'))    #si  1.491811383668924e-31
print(stats.wilcoxon(pet_ct_swinunetr, swinunetr, nan_policy='omit'))   #1.6791930555400654e-43
print(stats.wilcoxon(pet_ct_swinunetr, unet, nan_policy='omit'))   #4.1215381231535863e-42
print(stats.wilcoxon(pet_ct_swinunetr, unetr, nan_policy='omit'))   #2.8331745134241104e-51

USZ = pd.read_csv("C:\\Users\\andre\\Desktop\\dices_intercenterUSZ.csv", squeeze=True)#.fillna(0)
MDA = pd.read_csv("C:\\Users\\andre\\Desktop\\dices_intercenterMDA.csv", squeeze=True)#.fillna(0)
CHB = pd.read_csv("C:\\Users\\andre\\Desktop\\dices_intercenterCHB.csv", squeeze=True)#.fillna(0)

print(stats.mannwhitneyu(MDA, USZ, nan_policy='omit'))    #no
print(stats.mannwhitneyu(MDA, CHB, nan_policy='omit'))    #no
print(stats.mannwhitneyu(CHB, USZ, nan_policy='omit'))    #0.027894989368361205