# 2024 Steven Chiacchira
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from lorax.data import PRODESDataset
from lorax.model import SegmentingVisionTransformer

DATA_DIR: Final[Path] = Path("data/")
DATA_DIR_10m: Final[Path] = DATA_DIR / "10m"
DATA_DIR_20m: Final[Path] = DATA_DIR / "20m"

IMG_SIZE: Final[int] = 224
NUM_CLASSES: Final[int] = 1


def main() -> None:
    data_train: Dataset = PRODESDataset(
        IMG_SIZE,
        DATA_DIR_20m / "ALL.tif",
        DATA_DIR_20m / "BLA.tif",
        DATA_DIR_20m / "MASK.tif",
    )
    model: nn.Module = SegmentingVisionTransformer(
        (10, IMG_SIZE, IMG_SIZE), NUM_CLASSES
    )
    loss_fn: nn.Module = nn.BCELoss()
    optimizer: Optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(20):
        for x_data, y_data in DataLoader(
            data_train,
            batch_size=8,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
            num_workers=2,
        ):
            x_data: torch.Tensor = x_data.to(torch.float).cuda()
            y_data: torch.Tensor = y_data.to(torch.float).cuda()
            # one-hot encoding along second axis
            # why do they not have a standard function for this
            # y_data = (
            #     torch.permute(
            #         F.one_hot(torch.permute(y_data, (0, 3, 2, 1)), 2),
            #         (0, 4, 1, 2, 3),
            #     )
            #     .squeeze(4)
            #     .float()
            # )
            pred = model(x_data)

            loss = loss_fn(pred, y_data)
            print("----")
            print(loss.item())  # loss
            print((torch.sum(torch.round(pred) == y_data) / torch.numel(y_data)).item() * 100)  # accuracy
            print(torch.sum(torch.round(pred)).item())
            print(torch.sum(torch.round(y_data)).item())
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    torch.set_default_device("cuda")
    main()
