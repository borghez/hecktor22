import glob
import os

import pandas as pd
import torch

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

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:1024'

"""
root_dir: directory where to save the output file
data_dir: directory where to find the data, divided in PET, CT and label (nifti)

Data resampled isotropically with pixel dimension [1, 1, 1]. Each patient have
PET, CT and label images with same sizes. Calculated bounding box to reduce images
size by finding the bounding box that contains the tumor masks (used adapted Rebaud code)
"""

root_dir = '/home/andrea/Segm/data/'

#data_dir = '/home/andrea/Segm/data/resampled/'
data_dir = '/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/'


train_images_pt = sorted(glob.glob(os.path.join(data_dir, 'train_pet', "*.nii.gz")))
train_images_ct = sorted(glob.glob(os.path.join(data_dir, 'train_ct', "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, 'train_label', "*.nii.gz")))
data_dicts = [
            {"image_pt": image_name_pt, "image_ct": image_name_ct, "label": label_name}
            for image_name_pt, image_name_ct, label_name in zip(train_images_pt, train_images_ct, train_labels)]
train_files, val_files = data_dicts[:-100], data_dicts[-100:]

set_determinism(seed=42)

train_transforms = Compose(
    [
        LoadImaged(keys=["image_pt", "image_ct", "label"]),
        EnsureChannelFirstd(keys=["image_pt", "image_ct", "label"]),
        #SpatialPadd(keys=["image_pt", "image_ct", "label"], spatial_size=[96, 96, 96]),
        RandCropByPosNegLabeld(
            keys=["image_pt", "image_ct", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image_pt",
            image_threshold=0,
        ),
        RandFlipd(keys=["image_pt", "image_ct", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image_pt", "image_ct", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image_pt", "image_ct", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys=["image_pt", "image_ct"], nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys=["image_pt", "image_ct"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image_pt", "image_ct"], offsets=0.1, prob=0.5),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="images", dim=0),
        EnsureTyped(keys=["images", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image_pt", "image_ct", "label"]),
        EnsureChannelFirstd(keys=["image_pt", "image_ct", "label"]),
        NormalizeIntensityd(keys=["image_pt", "image_ct"], nonzero=True, channel_wise=True),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="images", dim=0),
        EnsureTyped(keys=["images", "label"]),
    ]
)

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1, num_workers=4)  #se da prob di mem riduco cache rate (1.0) e num_work=4
#train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1, num_workers=4)
#val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

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

loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()

max_epochs = 500
val_interval = 5
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)


for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["images"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        #epoch_loss += loss.item()
        optimizer.step()
        epoch_loss += loss.item()
        #optimizer.zero_grad()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["images"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model_pet_ct_segresnet_NEWBBOX.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
        print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

epoch_loss_dataframe = pd.DataFrame(epoch_loss_values)
epoch_loss_dataframe.to_csv('/home/andrea/Segm/data/losses_and_dices/train_loss_pet_ct_segresnet_NEWBBOX.csv', index=False)
metric_values_dataframe = pd.DataFrame(metric_values)
metric_values_dataframe.to_csv('/home/andrea/Segm/data/losses_and_dices/validation_dice_scores_pet_ct_segresnet_NEWBBOX.csv', index=False)

"""Prova anche con i test data resamplati"""

# data_dir_test = '/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/'
# test_images_pt = sorted(glob.glob(os.path.join(data_dir_test, "test_pet", "*.nii.gz")))
# test_images_ct = sorted(glob.glob(os.path.join(data_dir_test, "test_ct", "*.nii.gz")))
#
# test_data = [{"image_pt_test": image_name_pt_test, "image_ct_test": image_name_ct_test}
#              for image_name_pt_test, image_name_ct_test in zip(test_images_pt, test_images_ct)]
#
# test_org_transforms = Compose(
#     [
#         LoadImaged(keys=["image_pt_test", "image_ct_test"]),
#         EnsureChannelFirstd(keys=["image_pt_test", "image_ct_test"]),
#         NormalizeIntensityd(keys=["image_pt_test", "image_ct_test"], nonzero=True, channel_wise=True),
#         ConcatItemsd(keys=["image_pt_test", "image_ct_test"], name="images_test", dim=0),
#         EnsureTyped(keys="images_test")
#     ]
# )
#
# test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
# test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=2)
#
# output_dir = '/home/andrea/Segm/data/hecktor22/out/pet_ct_segresnet_NEWBBOX'
#
# post_transforms = Compose(
#     [
#         Invertd(
#             keys="pred",
#             transform=test_org_transforms,
#             orig_keys="image_pt_test",
#             meta_keys="pred_meta_dict",
#             orig_meta_keys="image_meta_dict",
#             meta_key_postfix="meta_dict",
#             nearest_interp=False,
#             to_tensor=True,
#         ),
#         #Activations(sigmoid=true),
#         AsDiscreted(keys="pred", argmax=True, to_onehot=3),
#         SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir, output_postfix="seg", resample=False),
#     ]
# )
#
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_pet_ct_segresnet_NEWBBOX.pth")))
# model.eval()
#
# with torch.no_grad():
#     for test_data in test_org_loader:
#         test_inputs = test_data["images_test"].to(device)
#         roi_size = (96, 96, 96)
#         sw_batch_size = 4
#         test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
#
#         test_data = [post_transforms(i) for i in decollate_batch(test_data)]

data_dir_test = '/home/andrea/Segm/data/resampled/PROVA_NEW_BBOX/'
test_images_pt = sorted(glob.glob(os.path.join(data_dir_test, "test_pet", "*.nii.gz")))
test_images_ct = sorted(glob.glob(os.path.join(data_dir_test, "test_ct", "*.nii.gz")))
test_label = sorted(glob.glob(os.path.join(data_dir_test, "test_label", "*.nii.gz")))


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

test_org_ds = CacheDataset(data=test_data, transform=test_org_transforms, cache_rate=1, num_workers=4)
#test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=2)

output_dir = '/home/andrea/Segm/data/hecktor22/out/pet_ct_segresnet_NEWBBOX'

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
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir, output_postfix="seg", resample=False),
    ]
)

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_pet_ct_segresnet_NEWBBOX.pth")))
model.eval()

with torch.no_grad():
    for test_data in test_org_loader:
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

        dice_metric(y_pred=test_data_pred, y=test_gt)

        metric = dice_metric.aggregate().item()

        test_data = [post_transforms(i) for i in decollate_batch(test_data)]

print(f"test completed, dice score: {metric:.4f}")

