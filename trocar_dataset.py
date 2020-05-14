from glob import glob
from os.path import basename
from random import random

import numpy as np
from PIL import Image
from albumentations import ReplayCompose
from torch.utils.data import Dataset


class TrocarDataset(Dataset):
    def __init__(self, image_source, mask_source, transformations):
        self.image_source = image_source
        self.mask_source = mask_source
        self.filenames = [basename(f) for f in glob(image_source + "*.png")]
        self.transformations = transformations

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = np.array(
            Image.open(self.image_source + self.filenames[idx]).convert("RGB")
        ).astype(np.uint8)
        mask = np.array(
            Image.open(self.mask_source + self.filenames[idx]).convert("L")
        ).astype(np.uint8)

        for transformation in self.transformations if self.transformations else []:
            image, mask = transformation(image, mask)

        image = np.array(image).astype(float)
        mask = np.expand_dims(np.array(mask), -1).astype(float)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return [image, mask]

    def get_filenames(self, idx):
        return (
            self.image_source + self.filenames[idx],
            self.mask_source + self.filenames[idx],
        )

    def revert_transformation(self, image, mask):
        for transformation in (
            self.transformations[::-1] if self.transformations else []
        ):
            image, mask = transformation.recall(image, mask)

        return image, mask

    def get_image_label_epistemic(self, idx):
        image = np.array(
            Image.open(self.image_source + self.filenames[idx]).convert("RGB")
        ).astype(np.uint8)
        mask = np.array(
            Image.open(self.mask_source + self.filenames[idx]).convert("L")
        ).astype(np.uint8)

        image = np.array(image).astype(float)
        mask = np.expand_dims(np.array(mask), -1).astype(float)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return [image, mask]

    def get_image_label_aleatoric(self, idx):
        image = np.array(
            Image.open(self.image_source + self.filenames[idx]).convert("RGB")
        ).astype(np.uint8)
        mask = np.array(
            Image.open(self.mask_source + self.filenames[idx]).convert("L")
        ).astype(np.uint8)

        return [image, mask]

    def get_transformation_aleatoric(self, image, mask):
        for transformation in self.transformations if self.transformations else []:
            image, mask = transformation(image, mask)

        image = np.array(image).astype(float)
        mask = np.expand_dims(np.array(mask), -1).astype(float)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return [image, mask]


class Transformation(object):
    def __init__(self, transformation, probability, apply_to_mask=False):
        self.transformation = transformation
        self.transformation.p = 1
        self.transformation = ReplayCompose([self.transformation])

        self.probability = probability
        self.p = None

        self.apply_to_mask = apply_to_mask

    def __call__(self, image, mask):
        self.p = random()

        return self._call(image, mask)

    def _call(self, image, mask):
        if self.p <= self.probability:
            data = self.transformation(image=image, force_apply=True)

            image = data["image"]

            self.replay = data["replay"]

            if self.apply_to_mask:
                mask = self.transformation.replay(self.replay, image=mask)["image"]

        mask[mask > 1] = 255
        mask[mask <= 1] = 0

        return image, mask

    def recall(self, image, mask):
        if self.p <= self.probability:
            image = ReplayCompose.replay(self.replay, image=image)["image"]

            if self.apply_to_mask:
                mask = ReplayCompose.replay(self.replay, image=mask)["image"]

        return image, mask
