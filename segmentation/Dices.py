import glob
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from monai.utils import set_determinism
from monai.networks.layers.factories import Norm
from monai.transforms import (
    AsDiscrete,
    Compose,
    AsDiscreted,
    EnsureChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    SaveImaged,
    Invertd,
    SpatialPadd,
    ConcatItemsd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    EnsureTyped,
    RandFlipd,
    RandShiftIntensityd
)

from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, SegResNet, SwinUNETR
from monai.inferers import sliding_window_inference
from tqdm import tqdm

import resource

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:1024'

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

data_dir_test = '/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/'

device = torch.device("cuda:0")

model = SegResNet(
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    init_filters=16,
    in_channels=2,
    out_channels=3,
    dropout_prob=0.2,
    norm=Norm.BATCH
).to(device)

# model = SwinUNETR(
#     img_size=(96, 96, 96),
#     in_channels=2,
#     out_channels=3,
#     feature_size=48,
#     drop_rate=0.2,
#     attn_drop_rate=0.0,
#     dropout_path_rate=0.0,
#     use_checkpoint=True,
# ).to(device)

root_dir = '/home/andrea/Segm/data/'

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_pet_ct_segresnet_NEWBBOX.pth")))
#model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_pet_ct_swinunetr_NEWBBOX.pth")))
model.eval()


test_images_pt = sorted(glob.glob(os.path.join(data_dir_test, "test_pet", "*.nii.gz")))[312:313]
test_images_ct = sorted(glob.glob(os.path.join(data_dir_test, "test_ct", "*.nii.gz")))[312:313]
test_label = sorted(glob.glob(os.path.join(data_dir_test, "test_label", "*.nii.gz")))[312:313]
print(test_images_ct)

test_data = [{"image_pt_test": image_name_pt_test, "image_ct_test": image_name_ct_test, "label_test": image_name_label_test}
                for image_name_pt_test, image_name_ct_test, image_name_label_test in zip(test_images_pt, test_images_ct, test_label)]

test_org_transforms = Compose(
    [
        LoadImaged(keys=["image_pt_test", "image_ct_test", "label_test"]),
        EnsureChannelFirstd(keys=["image_pt_test", "image_ct_test", "label_test"]),
        NormalizeIntensityd(keys=["image_pt_test", "image_ct_test"], nonzero=True, channel_wise=True),
        ConcatItemsd(keys=["image_pt_test", "image_ct_test"], name="images_test", dim=0),
        EnsureTyped(keys="images_test")
    ]
)

#test_org_ds = CacheDataset(data=test_data, transform=test_org_transforms, cache_rate=1, num_workers=8)
test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

#output_dir = '/home/andrea/Segm/data/hecktor22/out/pet_ct_segresnet_NEWBBOX'

post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image_pt_test",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        #Activations(sigmoid=true),
        AsDiscreted(keys="pred", argmax=True, to_onehot=3),
        #SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir, output_postfix="seg", resample=False),
    ]
)

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

primary = []
lymph = []
means = []


with torch.no_grad():
    for test_data in tqdm(test_org_loader):
        test_inputs, test_gt = (
            test_data["images_test"].to(device),
            test_data["label_test"].to(device),
        )
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        test_data_pred = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        test_data_pred = [post_pred(i) for i in decollate_batch(test_data_pred)]
        test_gt = [post_label(i) for i in decollate_batch(test_gt)]

        dice = dice_metric(y_pred=test_data_pred, y=test_gt)
        metric = dice_metric.aggregate().item()
        # print(metric)
        # print(dice.cpu().numpy().tolist())
        prim = dice.cpu().numpy()[0, 0]
        lym = dice.cpu().numpy()[0, 1]
        # print(prim, lym)
        primary.append(prim)
        lymph.append(lym)
        med = np.nanmean([prim, lym])
        print(prim, lym, med)
        means.append(med)

        test_data = [post_transforms(i) for i in decollate_batch(test_data)]

# print("Model: Segresnet vera")
# print(f"test completed, dice score: {metric:.4f}")
# print(f"Highest Dice score: {np.nanmax(means):.4f}")
# print(f"Lowest Dice score: {np.nanmin(means):.4f}")
# print(f"Mean Dice score: {np.nanmean(means):.4f}")
# print(f"Std Dice score: {np.nanstd(means):.4f}")
# print(f"Mean Primary Dice score: {np.nanmean(primary):.4f}")
# print(f"Std Primary Dice score: {np.nanstd(primary):.4f}")
# print(f"Mean Lymph nodes Dice score: {np.nanmean(lymph):.4f}")
# print(f"Std Lymph nodes Dice score: {np.nanstd(lymph):.4f}")
#
#
# means_dataframe = pd.DataFrame(means)
# means_dataframe.to_csv('/home/andrea/dices_pet_ct_segresnet.csv', index=False)
# #means_dataframe.to_csv('/home/andrea/dices_pet_ct_swinunetr.csv', index=False)

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_pet_ct_segresnet_NEWBBOX.pth")))
model.eval()
with torch.no_grad():
    for i, val_data in enumerate(test_org_loader):
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(val_data["images_test"].to(device), roi_size, sw_batch_size, model)
        # plot the slice [:, :, 60]
        plt.figure("check", (18, 6))
        plt.suptitle(f"USZ-055 CT slice, ground truth manual segmentation, and precited labels from bimodal SegResNet. Dice score: {means[0]}")
        plt.subplot(1, 3, 1)
        #plt.title(f"image {i}")
        plt.title("CT")
        plt.imshow(val_data["images_test"][0, 2, :, :, 100], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("Ground truth")
        plt.imshow(val_data["label_test"][0, 0, :, :, 100])
        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 100])
        plt.show()
        plt.savefig(f"segm {i}")
        if i == 2:
            break