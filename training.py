import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from cross_entropy import cross_entropy
from dice_loss import dice_loss
from dice_plus_bce import dice_plus_bce
from generalised_dice import generalised_dice_loss
from generalised_plus_bce import generalised_plus_bce
from accuracy import accuracy


def train_model(
    model,
    optimizer,
    scheduler,
    loss_function,
    num_epochs,
    data_loaders,
    device,
    output_dir,
):
    GDL_saved = {"train": [], "val": [], "test": []}
    DICE_saved = {"train": [], "val": [], "test": []}
    GDL_CE_saved = {"train": [], "val": [], "test": []}
    CE_saved = {"train": [], "val": [], "test": []}
    DICE_CE_saved = {"train": [], "val": [], "test": []}
    ACC_saved = {"train": [], "val": [], "test": []}

    for epoch in range(num_epochs):

        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ["train", "val", "test"]:
            gdl_epoch = []
            dice_epoch = []
            gdl_ce_epoch = []
            ce_epoch = []
            dice_ce_epoch = []
            acc_epoch = []

            if phase == "train":
                # scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group["lr"])
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            real_epoch_loss = []

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.float().to(device)
                labels = (
                    (labels == 255).float().to(device)
                )  # true vs false then to float 0 vs 1

                # zero the parameter the gradients
                optimizer.zero_grad()

                # forward
                # track history if train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = loss_function(logits=outputs, target=labels)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()  # apply gd
                        scheduler.step()

                    losses = loss.cpu().item()
                    print("current loss: ", losses)

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.detach()
                    running_corrects += torch.sum(preds == labels.data)
                # save all losses and accuracy
                gdl_epoch.append(
                    generalised_dice_loss(logits=outputs, target=labels).cpu().item()
                )
                dice_epoch.append(dice_loss(logits=outputs, target=labels).cpu().item())
                ce_epoch.append(
                    cross_entropy(logits=outputs, target=labels).cpu().item()
                )
                gdl_ce_epoch.append(
                    generalised_plus_bce(logits=outputs, target=labels).cpu().item()
                )
                dice_ce_epoch.append(
                    dice_plus_bce(logits=outputs, target=labels).cpu().item()
                )
                acc_epoch.append(accuracy(logits=outputs, target=labels).cpu().item())

            GDL_saved[phase].append(gdl_epoch)
            DICE_saved[phase].append(dice_epoch)
            GDL_CE_saved[phase].append(gdl_ce_epoch)
            CE_saved[phase].append(ce_epoch)
            DICE_CE_saved[phase].append(dice_ce_epoch)
            ACC_saved[phase].append(acc_epoch)

            real_epoch_loss += [loss.detach().cpu().numpy()]
            epoch_loss = running_loss / len(data_loaders[phase].dataset)  # 2
            time_elapsed = time.time() - since

            print_format = "Epoch {}/{} - Phase {} - {:.0f}m {:.2f}s - Loss {:.4f}"
            print(
                print_format.format(
                    epoch,
                    num_epochs - 1,
                    str.capitalize(phase),
                    time_elapsed // 60,
                    time_elapsed % 60,
                    np.mean(real_epoch_loss),
                )
            )

    np.save(output_dir + "Training-GDL.npy", GDL_saved["train"])
    np.save(output_dir + "Validation-GDL.npy", GDL_saved["val"])
    np.save(output_dir + "Testing-GDL.npy", GDL_saved["test"])

    np.save(output_dir + "Training-Dice.npy", DICE_saved["train"])
    np.save(output_dir + "Validation-Dice.npy", DICE_saved["val"])
    np.save(output_dir + "Testing-Dice.npy", DICE_saved["test"])

    np.save(output_dir + "Training-GDL_CE.npy", GDL_CE_saved["train"])
    np.save(output_dir + "Validation-GDL_CE.npy", GDL_CE_saved["val"])
    np.save(output_dir + "Testing-GDL_CE.npy", GDL_CE_saved["test"])

    np.save(output_dir + "Training-CE.npy", CE_saved["train"])
    np.save(output_dir + "Validation-CE.npy", CE_saved["val"])
    np.save(output_dir + "Testing-CE.npy", CE_saved["test"])

    np.save(output_dir + "Training-CE_Dice.npy", DICE_CE_saved["train"])
    np.save(output_dir + "Validation-CE_Dice.npy", DICE_CE_saved["val"])
    np.save(output_dir + "Testing-CE_Dice.npy", DICE_CE_saved["test"])

    np.save(output_dir + "Training-ACC.npy", ACC_saved["train"])
    np.save(output_dir + "Validation-ACC.npy", ACC_saved["val"])
    np.save(output_dir + "Testing-ACC.npy", ACC_saved["test"])

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": real_epoch_loss,
            "epoch_loss": epoch_loss,
        },
        output_dir + "/segmentation.pt",
    )

    return model
