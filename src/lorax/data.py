# 2024 Steven Chiacchira
"""Custom dataset class for reading rasterized PRODES data.

__ISSUES__:
    * Passing a transform to :class:`PRODESDataset` currently breaks the dataset.
"""

from os import PathLike
from pathlib import Path
from typing import Any, List, Final, Optional, Tuple, Union

import rasterio as rio
import torchvision.transforms as T
import torch
from numpy.typing import NDArray
from rasterio.windows import Window
from torch import Tensor
from torch.utils.data import Dataset


class PRODESDataset(Dataset):
    """Custom :class:`torch.util.data.Dataset` that implements lazy-loading for raster images."""

    @staticmethod
    def collate_fn(batch: List[Tuple[Tensor]]) -> Tuple[Tensor, Tensor]:
        images, labels = zip(*batch)  # transpose elements

        images = torch.stack(images)
        labels = torch.stack(labels)

        return images, labels

    def __init__(
        self,
        img_size: int,
        x_file: PathLike,
        y_file: PathLike,
        mask_file: Optional[PathLike] = None,
        transform: Any = None,
        selected_indices: Optional[List[int]] = None,
    ) -> None:
        """Creates a new :class:`PRODESDataset`.

        Note that the images at `x_file`, `y_file`, and (optionally) `mask_file`, should be of the same shape, datatype, and use the same CRS.
        :param img_size: the sidelength of images to return.
        :param x_file: the path to the file containing input data in a raster format.
        :param y_file: the path to the file containing labels for the input data in a raster format.
        :param mask_file: the optional path to the file containing cloud masks for the input data. Mask data should be `0` where pixels are to be masked out and `1` otherwise.
        :param transform: the optional transform to apply to images.
        :param selected_indices: an optional list of indices to load.
        """
        super().__init__()
        self.x_file = Path(x_file)
        self.y_file = Path(y_file)
        if mask_file:
            self.mask_file = Path(mask_file)
        else:
            self.mask_file = None

        self.transform = transform
        self.img_size = img_size

        self.to_tensor = T.ToTensor()

        with rio.open(self.x_file) as src:
            prof: Final[Any] = src.profile
            self.height = prof["height"]
            self.width = prof["width"]

        self.selected_indices = selected_indices

    def __len__(self) -> int:
        """Returns the number of elements in the dataset.

        Returns the number of patches available in the image, or the number of selected indices.

        :returns: the number of patches available in the image, or the number of selected indices.
        """
        if self.selected_indices is not None:
            return len(self.selected_indices)
        return ((self.height // self.img_size) - 1) * (
            (self.width // self.img_size) - 1
        )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Returns data lazy-loaded from the images at index `index`.

        Data is lazily accessed from patches of an image as shown below:

        +---+---+---+
        | 0 | 1 | 2 |
        +---+---+---+
        | 3 | 4 | 5 |
        +---+---+---+
        | 6 | 7 | 8 |
        +---+---+---+
        | 9 | 10| 11|
        +---+---+---+

        with edge data padded using zeros.

        If a mask is present, masks will be applied to both the returned `x_data` and `y_data`.

        :param index: the index to access data from.
        :returns: a tuple of (`x_data`, `y_data`)
        """

        if self.selected_indices is not None:
            _index: int = self.selected_indices[index]
        else:
            _index: int = index

        col_offset: Final[int] = (_index * self.img_size) % self.width
        row_offset: Final[int] = (_index * self.img_size) // self.width
        win: Final[Window] = Window(
            col_offset,  # type: ignore
            row_offset,
            self.img_size,
            self.img_size,
        )

        with rio.open(self.x_file) as src_x, rio.open(
            self.y_file, **src_x.profile
        ) as src_y:
            x_data: Union[NDArray, Tensor] = src_x.read(
                window=win, boundless=True
            )
            y_data: Union[NDArray, Tensor] = src_y.read(
                window=win, boundless=True
            )

        # mask out clouds
        if self.mask_file is not None:
            with rio.open(self.mask_file) as src_mask:
                mask_data: Final[NDArray] = src_mask.read(
                    window=win, boundless=True
                )
                x_data *= mask_data
                y_data *= mask_data

        x_data = self.to_tensor(x_data)
        x_data = x_data.transpose(0, 1)

        y_data = self.to_tensor(y_data)
        y_data = y_data.transpose(0, 1)

        if self.transform is not None:
            state: Any = torch.get_rng_state()
            x_data = self.transform(x_data)
            torch.set_rng_state(state)
            y_data = self.transform(y_data)

        return x_data, y_data  # type: ignore
