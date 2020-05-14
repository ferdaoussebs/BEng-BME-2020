import argparse
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from os.path import split
from PIL import Image

from albumentations import Blur
from albumentations import ElasticTransform
from albumentations import HorizontalFlip
from albumentations import ImageCompression
from albumentations import RGBShift
from albumentations import RandomBrightnessContrast
from albumentations import ShiftScaleRotate
from albumentations import Transpose
from albumentations import VerticalFlip

import numpy as np
import datetime
import pathlib

from scipy.stats import entropy

import model
import training
from trocar_dataset import Transformation
from trocar_dataset import TrocarDataset

from dice_loss import dice_loss
from cross_entropy import cross_entropy
from generalised_dice import generalised_dice_loss
from dice_plus_bce import dice_plus_bce
from generalised_plus_bce import generalised_plus_bce

torch.manual_seed(7)
np.random.seed(7)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def rescale_for_rgb(image):
    mask_min = image.min()
    mask_max = image.max()
    image = (image - mask_min) / (mask_max - mask_min)
    image = image * 255.0
    image = image.astype(np.uint8)

    return image


def process_logits(image, loss_function):
    if loss_function in ["generalised_dice_loss", "dice_loss"]:
        image = torch.sigmoid(image)
    else:
        image = torch.nn.Softmax(dim=1)(image)

    return image


def calculate_entropy(arr):
    arr[arr <= 0.5] = 0
    arr[arr > 0.5] = 1

    return np.apply_along_axis(
        # Create a lambda function as a wrapper because of the multi return of
        # np.unique
        func1d=lambda x: entropy(
            pk=(
                # Get the counts of each unique element
                np.unique(ar=x, return_counts=True)[1]
                # Normalize the counts by the total count
                / len(x)
            )
        ),
        # Apply it on axis 0 which means that we apply it on a per pixel basis
        axis=0,
        arr=arr,
    )

# passing arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--training_dir",
    help="training directory",
    action="store",
    type=str,
    default="train",
)

parser.add_argument(
    "--validation_dir",
    help="validation directory",
    action="store",
    type=str,
    default="val",
)

parser.add_argument(
    "--testing_dir", help="testing directory", action="store", type=str, default="test"
)

parser.add_argument(
    "--output_dir", help="output directory", action="store", type=str, default="output"
)

parser.add_argument("--epochs", help="Number of epochs", action="store", type=np.uint)

parser.add_argument(
    "--lr_scheduler",
    help="learning rate scheduler",
    action="store",
    type=str,
    choices=["linear", "cyclic"],
    default="cyclic",
)
parser.add_argument(
    "--augmentations",
    help="whether to use aug",
    action="store",
    type=str,
    choices=["Uncertainty", "General", "None"],
    default="General",
)

parser.add_argument(
    "--dropout", help="whether to use dropout", action="store", type=float, default=0.5
)

parser.add_argument(
    "--device", help="which gpu to use", action="store", type=int, default=0
)

parser.add_argument(
    "--losses",
    help="loss functions",
    action="store",
    type=str,
    choices=[
        "generalised_dice_loss",
        "dice_loss",
        "cross_entropy",
        "dice_plus_bce",
        "generalised_plus_bce",
    ],
    default="generalised_dice_loss",
)

args = parser.parse_args(
    args=[
        "--training_dir",
        "C:\Users\Ferdaousse\OneDrive - King's College London\KCL19-20\BEng Research Project\Dissertation\FINAL_CODE\code_experiments\data\train",
        "--testing_dir",
        "C:\Users\Ferdaousse\OneDrive - King's College London\KCL19-20\BEng Research Project\Dissertation\FINAL_CODE\code_experiments\data\test",
        "--validation_dir",
        "C:\Users\Ferdaousse\OneDrive - King's College London\KCL19-20\BEng Research Project\Dissertation\FINAL_CODE\code_experiments\data\val",
        "--output_dir",
        "C:\Users\Ferdaousse\OneDrive - King's College London\KCL19-20\BEng Research Project\Dissertation\FINAL_CODE\code_experiments\data\output",
        "--epochs",
        "500",
        "--lr_scheduler",
        "cyclic",
        "--augmentations",
        "Uncertainty",
        "--device",
        "1",
        "--dropout",
        "0.5",
        "--losses",
        "generalised_plus_bce",
    ]
)

if args.dropout == 0.5:
    print("apply dropout p = 0.5")
elif args.dropout == 0:
    print("no dropout - p = 0")
if args.losses == "generalised_dice_loss":
    loss_function = generalised_dice_loss
    print("using GDL")
elif args.losses == "cross_entropy":
    loss_function = cross_entropy
    print("using cross entropy")
elif args.losses == "dice_loss":
    loss_function = dice_loss
    print("using dice loss")
elif args.losses == "dice_plus_bce":
    loss_function = dice_plus_bce
    print("using dice plus bce")
elif args.losses == "generalised_plus_bce":
    loss_function = generalised_plus_bce
    print("using generalised plus bce")


# General Variables
batch_size = 4
num_class = 2
torch.cuda.set_device(args.device)
device = torch.device("cuda")
output_directory = (
    args.output_dir
    + "/"
    + str(args.lr_scheduler)
    + "-"
    + str(args.augmentations)
    + "-"
    + str(args.dropout)
    + "-"
    + str(args.losses)
    + "-"
    + datetime.date.today().strftime("%b-%d-%Y")
    + "/"
)

pathlib.Path(output_directory).mkdir(parents=True, exist_ok=False)

transformations = [
    Transformation(VerticalFlip(), probability=0.25, apply_to_mask=True),
    Transformation(HorizontalFlip(), probability=0.25, apply_to_mask=True),
    Transformation(Transpose(), probability=0.25, apply_to_mask=True),
    Transformation(RandomBrightnessContrast(), probability=0.25, apply_to_mask=False),
    Transformation(
        ImageCompression(quality_lower=50), probability=0.25, apply_to_mask=False
    ),
    Transformation(RGBShift(), probability=0.25, apply_to_mask=False),
]

if args.augmentations == "General":
    transformations.extend(
        [
            # Transformation(ElasticTransform(), probability=0.25, apply_to_mask=True),
            Transformation(ShiftScaleRotate(), probability=0.25, apply_to_mask=True),
            Transformation(Blur(), probability=0.25, apply_to_mask=True),
        ]
    )

# Datasets
train_set = TrocarDataset(
    image_source=args.training_dir + "image/",
    mask_source=args.training_dir + "label/",
    transformations=transformations if args.augmentations != "None" else None,
)

val_set = TrocarDataset(
    image_source=args.validation_dir + "image/",
    mask_source=args.validation_dir + "label/",
    transformations=transformations if args.augmentations != "None" else None,
)

test_set = TrocarDataset(
    image_source=args.testing_dir + "image/",
    mask_source=args.testing_dir + "label/",
    transformations=transformations if args.augmentations != "None" else None,
)

data_loaders = {
    "train": DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    ),
    "val": DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    ),
    "test": DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    ),
}

trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# Model
model = model.UNet(num_class)
model.to(device)

# Optimizer
optimizer_ft = optim.Adam(model.parameters(), lr=1e-5)

if args.lr_scheduler == "linear":
    lr_scheduler = LambdaLR(optimizer_ft, lr_lambda=lambda epoch: 1.0)
    print("none")

elif args.lr_scheduler == "cyclic":
    print("cyclicLR")
    lr_scheduler = lr_scheduler.CyclicLR(
        optimizer_ft,
        base_lr=1e-5,
        max_lr=0.1,
        step_size_up=100,
        step_size_down=1000,
        mode="exp_range",
        gamma=0.98,
        scale_fn=None,
        scale_mode="cycle",
        cycle_momentum=False,
        base_momentum=0.1,
        max_momentum=0.9,
        last_epoch=-1,
    )

model = training.train_model(
    model,
    optimizer_ft,
    lr_scheduler,
    loss_function=loss_function,
    data_loaders=data_loaders,
    device=device,
    num_epochs=args.epochs,
    output_dir=output_directory,
)

for set_name, data_set in [("val", val_set), ("test", test_set)]:
    model.eval()  # Set model to the evaluation mode

    for i in range(len(data_set)):
        input, label = data_set.__getitem__(i)
        _, absolute_path = data_set.get_filenames(i)
        path, filename = split(absolute_path)

        im = Image.fromarray(np.transpose(input, (1, 2, 0)).astype(np.uint8), "RGB")
        im.save(output_directory + set_name +"-input-" + filename)

        im = Image.fromarray(label[0].astype(np.uint8), "L")
        im.save(output_directory + set_name +"-label-" + filename)

        input = torch.Tensor(input).to(device).float().unsqueeze_(0)
        label = torch.Tensor(label).to(device).float().unsqueeze_(0)

        pred = model(input).to(device)
        pred = process_logits(pred, args.losses)
        pred = pred.squeeze(0).data.cpu().numpy()[1, ...]
        input = np.transpose(input.squeeze(0).data.cpu().numpy(), (1, 2, 0))

        _, reverted_pred = data_set.revert_transformation(input, pred)

        im = Image.fromarray(rescale_for_rgb(pred), "L")
        im.save(output_directory + set_name +"-pred-" + filename)

        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
        im = Image.fromarray(rescale_for_rgb(pred), "L")
        im.save(output_directory + set_name +"-binarised_pred-" + filename)

        im = Image.fromarray(rescale_for_rgb(reverted_pred), "L")
        im.save(output_directory + set_name +"-reverted_pred-" + filename)

        reverted_pred[reverted_pred <= 0.5] = 0
        reverted_pred[reverted_pred > 0.5] = 1
        im = Image.fromarray(rescale_for_rgb(reverted_pred), "L")
        im.save(output_directory + set_name +"-reverted_binarised_pred-" + filename)


    if args.augmentations == "Uncertainty":
        # Epistemic Uncertainty
        model.train()
        dropout_samples = 10

        for i in range(len(data_set)):
            input, label = data_set.get_image_label_epistemic(i)
            _, absolute_path = data_set.get_filenames(i)
            path, filename = split(absolute_path)

            input = torch.Tensor(input).to(device).float().unsqueeze_(0)
            label = torch.Tensor(label).to(device).float().unsqueeze_(0)

            samples_array = np.zeros(shape=(dropout_samples, 512, 512), dtype=np.float)

            for sample in range(dropout_samples):
                pred = model(input).to(device)
                pred = process_logits(pred, args.losses)
                pred = pred.squeeze(0).data.cpu().numpy()
                samples_array[sample, ...] = pred[1, ...]

            mean_pred = np.mean(a=samples_array, axis=0)
            mean_pred = np.expand_dims(a=mean_pred, axis=-1)
            mean_pred = np.tile(A=mean_pred, reps=3)

            im = Image.fromarray(rescale_for_rgb(mean_pred), "RGB")
            im.save(output_directory + set_name +"-epistemic-mean_pred-" + filename)

            mean_pred[mean_pred <= 0.5] = 0
            mean_pred[mean_pred > 0.5] = 1
            im = Image.fromarray(rescale_for_rgb(mean_pred), "RGB")
            im.save(output_directory + set_name +"-epistemic-mean_binarized_pred-" + filename)

            epistemic_var = calculate_entropy(arr=samples_array)
            epistemic_var = np.expand_dims(a=epistemic_var, axis=-1)
            epistemic_var = np.tile(A=epistemic_var, reps=3)
            epistemic_var = rescale_for_rgb(epistemic_var)

            im = Image.fromarray(epistemic_var, "RGB")
            im.save(output_directory + set_name +"-epistemic-var_pred-" + filename)

        #aleatoric uncertainty
        model.eval()
        aug_samples = 10

        for i in range(len(data_set)):
            original_inputs, original_labels = data_set.get_image_label_aleatoric(i)
            _, absolute_path = data_set.get_filenames(i)
            path, filename = split(absolute_path)

            aug_array = np.zeros(shape=(aug_samples, 512, 512), dtype=np.float)

            for j in range(aug_samples):
                inputs_aug, labels_aug = data_set.get_transformation_aleatoric(
                    original_inputs, original_labels
                )

                inputs_aug = torch.Tensor(inputs_aug).to(device).float().unsqueeze(0)
                labels_aug = torch.Tensor(labels_aug).to(device).float().unsqueeze(0)
                inputs_aug = torch.transpose(inputs_aug, -1, -1)
                labels_aug = torch.transpose(labels_aug, -1, -1)

                pred_aug = model(inputs_aug)
                pred_aug = process_logits(pred_aug, args.losses)
                pred_aug = pred_aug.squeeze(0).data.cpu().numpy()[1, ...]

                # get reversed transform
                inputs_aug = inputs_aug.squeeze(0).permute((1, 2, 0)).data.cpu().numpy()
                rev_image, rev_mask = data_set.revert_transformation(inputs_aug, pred_aug)

                aug_array[j, ...] = rev_mask

            # mean
            mean_pred_al = np.mean(a=aug_array, axis=0)
            mean_pred_al = np.expand_dims(a=mean_pred_al, axis=-1)
            mean_pred_al = np.tile(A=mean_pred_al, reps=3)

            # variance
            aleatoric_var_al = calculate_entropy(arr=aug_array)
            aleatoric_var_al = np.expand_dims(a=aleatoric_var_al, axis=-1)
            aleatoric_var_al = np.tile(A=aleatoric_var_al, reps=3)
            aleatoric_var_al = rescale_for_rgb(aleatoric_var_al)

            im = Image.fromarray(rescale_for_rgb(mean_pred_al), "RGB")
            im.save(output_directory + set_name +"-aleatoric_mean_pred-" + filename)

            mean_pred_al[mean_pred_al <= 0.5] = 0
            mean_pred_al[mean_pred_al > 0.5] = 1
            im = Image.fromarray(rescale_for_rgb(mean_pred_al), "RGB")
            im.save(output_directory + set_name +"-aleatoric_mean_binarized_pred-" + filename)

            im = Image.fromarray(aleatoric_var_al, "RGB")
            im.save(output_directory + set_name +"-aleatoric_var_pred-" + filename)
