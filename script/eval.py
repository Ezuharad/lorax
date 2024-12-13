# 2024 Steven Chiacchira
from pathlib import Path
from typing import Final, List, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    ToDtype,
    Transform,
)

from lorax.data import PRODESDataset
from lorax.model import SegmentingVisionTransformer

DATA_DIR: Final[Path] = Path("data/")
DATA_DIR_10m: Final[Path] = DATA_DIR / "10m"
DATA_DIR_20m: Final[Path] = DATA_DIR / "20m"

SELECTED_INDICIES_FILE: Final[Path] = DATA_DIR / "selected_indices.nsv"

DATA_MEAN_STDDEV: Final[Tuple[Tuple[float, ...], Tuple[float, ...]]] = (
    (
        1195.3787184673965,
        1182.2289404599,
        1420.5364391212518,
        1273.570552341147,
        1742.5178133180627,
        3003.282179616849,
        3513.3258864826275,
        2565.040663300214,
        1760.488491343408,
        3733.973476172061,
    ),
    (
        538.399494860722,
        603.1013959776976,
        626.6649001515442,
        581.2624896727273,
        775.0420910235603,
        1331.9648049612458,
        1569.6819327119217,
        1142.6240792039875,
        810.7407695696734,
        1665.1815731204001,
    ),
)

BATCH_SIZE: Final[int] = 32
LEARNING_RATE: Final[float] = 0.001
NUM_EPOCHS: Final[int] = 100

TEST_SIZE: Final[int] = 1919

IMG_SIZE: Final[int] = 64
IMG_CHAN: Final[int] = 10
NUM_CLASSES: Final[int] = 1


def get_dataset(indices: List[int]) -> DataLoader[Tensor]:
    data: Dataset = PRODESDataset(
        IMG_SIZE,
        DATA_DIR_20m / "ALL.tif",
        DATA_DIR_20m / "BLA.tif",
        DATA_DIR_20m / "MASK.tif",
        transform=Compose(
            [
                ToDtype(torch.float32),
                RandomHorizontalFlip(),
                RandomRotation(90),  # type: ignore
            ]
        ),
        selected_indices=indices,
    )
    return DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        generator=torch.Generator(device="cuda"),
        num_workers=0,
    )


def get_model() -> nn.Module:
    return SegmentingVisionTransformer(
        (IMG_CHAN, IMG_SIZE, IMG_SIZE),
        NUM_CLASSES,
        patch_size=8,
        num_heads=12,
        num_layers=12,
        embed_dim=288,
    )


def main() -> None:
    x_normalize_transform: Final[Transform] = Normalize(*DATA_MEAN_STDDEV)
    selected_indices = list(
        map(lambda l: int(l.strip()), open(SELECTED_INDICIES_FILE))
    )

    data_train: Final[DataLoader[Tensor]] = get_dataset(
        selected_indices[TEST_SIZE:]
    )
    data_test: Final[DataLoader[Tensor]] = get_dataset(
        selected_indices[:TEST_SIZE]
    )

    model = get_model()
    model.load_state_dict(torch.load("data/model/checkpoint.pt"))
    confusion_matrix: List[List[int]] = [[0, 0], [0, 0]]
    for x_data, y_data in data_train:
        x_data: Tensor = x_normalize_transform(x_data).cuda()
        y_data: Tensor = y_data.cuda()

        pred = model(x_data)

        deforested_regions = y_data == 1
        forested_regions = y_data == 0

        pred_deforested_regions = pred == 1
        pred_forested_regions = pred == 0

        print(pred.sum())
        confusion_matrix[0][0] += (
            (forested_regions[pred_forested_regions]).sum().item()
        )
        confusion_matrix[1][0] += (
            (deforested_regions[pred_forested_regions]).sum().item()
        )
        confusion_matrix[1][1] += (
            (deforested_regions[pred_deforested_regions]).sum().item()
        )
        confusion_matrix[0][1] += (
            (forested_regions[pred_deforested_regions]).sum().item()
        )

    print(confusion_matrix)


if __name__ == "__main__":
    torch.set_default_device("cuda")
    main()
