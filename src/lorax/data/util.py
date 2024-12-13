# 2024 Steven Chiacchira
"""Utilities for calculating dataset statistics."""

from functools import reduce
from operator import mul
from typing import Final, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@torch.compile
def get_mean_stddev(
    dataset: Dataset[Tensor],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """TODO"""
    data_shape: Final[torch.Size] = dataset[0][0].shape
    num_channels: Final[int] = data_shape[0]
    num_pixels_per_channel: Final[int] = reduce(mul, data_shape[1:], 1)

    loader: Final[DataLoader[Tensor]] = DataLoader(dataset, batch_size=32)

    mean: Tensor = torch.zeros(num_channels, dtype=torch.float64)
    for x_data, _ in loader:
        mean += (
            x_data.transpose(0, 1).sum(dim=[1, 2, 3]) / num_pixels_per_channel
        )
    mean /= len(dataset)  # type: ignore

    variance: Tensor = torch.zeros(num_channels, dtype=torch.float64)
    for x_data, _ in loader:
        batch_differences: Tensor = (
            x_data.transpose(1, 3) - mean
        )  # B C H W -> B H W C
        batch_differences = torch.square(batch_differences).transpose(
            0, 3
        )  # B H W C -> C B H W
        variance += (
            batch_differences.sum(dim=[1, 2, 3]) / num_pixels_per_channel
        )  # C B H W -> C

    variance /= len(dataset)  # type: ignore
    stddev = torch.sqrt(variance)

    return tuple(mean.tolist()), tuple(stddev.tolist())
