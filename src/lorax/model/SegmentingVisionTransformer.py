# 2024 Steven Chiacchira
"""
Definitions of the vision transformer segmentation model components.
"""

from typing import Tuple, Optional

import torch
from torch import nn, Tensor


class PatchEmbedLayer(nn.Module):
    """Layer for embedding square patches of square images into feature vectors.

    The PatchEmbedLayer takes an image of shape (c, n, n) and splits it into (c / p)^2 patches, where p is the sidelength of the square patches.

    __IMAGE__    __PATCHES__
    +-------+    +---+---+
    |       |    |   |   |
    |       | -> +---+---+
    |       |    |   |   |
    +-------+    +---+---+

    Each patch is then transformed into a 1D patch embedding with a positional encoding.

    __PATCHES__
    +---+---+
    |   |   |                             _EMBEDDINGS_
    +---+---+ + positional encoding -> 4x |IIII...IIIIIIIIIIIII|
    |   |   |
    +---+---+

    """

    def _validate(self):
        if self.img_shape[1] != self.img_shape[2]:
            raise ValueError(
                "image height and width must be equivalent!"
                + f"({self.img_shape} != {self.img_shape[1]})"
            )
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                "image height and width must be divisible by patch_size! "
                + f"({self.img_size} % {self.patch_size} != 0)"
            )

    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        patch_size: int,
        embed_dim: int,
    ) -> None:
        """Creates a new `PatchEmbedLayer` object.

        :param img_shape: a 3-tuple of integer specifying the expected input image shape as (channels, width, height).
        :param patch_size: the side length of an image patch to use for embedding.
        :param embed_dim: the dimension of 1D embedding created from each patch.

        :raises ValueError: `img_shape` must specify a square image.
        :raises ValueError: the sidelength of the image specified by `img_shape` must be divisible by `patch_size`.
        """
        super().__init__()
        self.img_shape: Tuple[int, int, int] = img_shape
        self.num_channels: int = img_shape[0]
        self.img_size: int = img_shape[1]

        self.patch_size: int = patch_size
        self.num_patches: int = (self.img_size // patch_size) ** 2
        self.patch_dim: int = (patch_size**2) * self.num_channels

        self.embed_dim: int = embed_dim

        # sneaky way to use a Conv2D layer for embeddings:
        # effectively we're using the same dense kernel for each patch
        self.proj: nn.Module = nn.Conv2d(
            self.num_channels,
            self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )
        self.flatten: nn.Module = nn.Flatten(2)

        self._validate()

    def forward(self, x: Tensor) -> Tensor:
        """Splits each sample in batched tensor `x` into patches of shape (`img_size[0]`, `patch_size`, `patch_size`), then transforms each patch into a 1d embedding with length `embed_dim`.

        :param x: the batched tensor of shape (`b`, `img_size[0]`, `img_size[1]`, `img_size[2]`)

        :returns: a batched tensor of patch embeddings of shape (`b`, (`img_size[1]` / `patch_size`)^2, `embed_dim`)
        """
        x = self.proj(
            x
        )  # (batch_size, embed_dim, height / patch_size, width / patch_size)
        x = self.flatten(x)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        x += self.pos_embed

        return x


class MultiheadSelfAttentionLayer(nn.Module):
    """Performs self attention on sets of 1D embeddings."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Creates a new `MultiheadSelfAttentionLayer.

        :param embed_dim: the size of the 1D embeddings to perform attention on.
        :param num_heads: the number of heads to use for self-attention.
        """
        super().__init__()
        self.attn: nn.Module = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x: Tensor) -> Tensor:
        """Performs self-attention on each 1D patch embedding set.

        :param x: a batched tensor of shape (b, n, `embed_dim`).

        :returns: a batched tensor of shape (b, b, `embed_dim`).
        """
        return self.attn(x, x, x)[0]


class FeedForwardLayer(nn.Module):
    """A simple two layer feedforward layer using ReLU.

    Consists of two linear layers with ReLU activation functions.

    __LAYERS__
    * Linear: (b, num_patches, embed_dim) -> (b, num_patches, hidden_dim)
    * ReLU
    * Linear: (b, num_patches, hidden_dim) -> (b, num_patches, embed_dim)
    * ReLU
    """

    def __init__(
        self, embed_dim: int, hidden_dim: Optional[int] = None
    ) -> None:
        """Creates a new FeedForwardLayer.

        :param embed_dim: the size of the 1D patch embeddings which are processed by the FeedForwardLayer.
        :param hidden_dim: the size of the intermediate embeddings created by the first linear layer. Defaults to `embed_dim`.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim

        self.ffn: nn.Module = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Processes a batched tensor of sets of 1D patch embeddings.

        :param x: a batched tensor containing sets of 1D patch embeddings with shape (b, num_patches, embed_dim).

        :returns: a batched tensor containing processed sets of 1D patch embeddings with shape (b, num_patches, embed_dim)
        """
        return self.ffn(x)


class TransformerEncoderLayer(nn.Module):
    """Intermediate Vision transformer processing layer. Performs attention and linear layer processing with skip connections on inputs while retaining their shapes.

    __LAYERS__
    * Skip-connected block 1:
        * Layer Normalization (b, num_patches, embed_dim) -> (b, num_patches, embed_dim)
        * Multihead attention layer
    * Skip-connected block 2:
        * Layer Normalization (b, num_patches, embed_dim) -> (b, num_patches, embed_dim)
        * Linear: (b, num_patches, embed_dim) -> (b, num_patches, hidden_dim)
        * ReLU
        * Linear: (b, num_patches, hidden_dim) -> (b, num_patches, embed_dim)
        * ReLU
    """

    def __init__(
        self, embed_dim: int, num_heads: int, hidden_dim: Optional[int] = None
    ):
        """Creates a bew TransformerEncoder Layer.

        :param embed_dim: the size of the 1D patch embeddings to be processed by the network.
        :param num_heads: the number of attention heads used by the MultiheadAttentionLayer.
        :param hidden_dim: the size of the intermediate 1D patch embeddings calculated in the feedforward skip-connected block. Defaults to `embed_dim`.
        """
        super().__init__()
        self.attn: nn.Module = MultiheadSelfAttentionLayer(
            embed_dim, num_heads
        )
        self.ffn: nn.Module = FeedForwardLayer(embed_dim, hidden_dim)
        self.ln1: nn.Module = nn.LayerNorm(embed_dim)
        self.ln2: nn.Module = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Processes x with a self-attention skip-connected block and a feedforward skip-connected block.

        :param x: a batched tensor containing sets of 1D patch embeddings with shape (b, num_patches, embed_dim).

        :returns: a batched tensor containing processed sets of 1D patch embeddings with shape (b, num_patches, embed_dim).
        """
        x_res: Tensor = x  # (batch_size, num_patches, embed_dim)
        x = self.ln1(x)  # (batch_size, num_patches, embed_dim)
        x = self.attn(x)  # (batch_size, num_patches, embed_dim)
        x = x + x_res  # (batch_size, num_patches, embed_dim)

        x_res = x  # (batch_size, num_patches, embed_dim)
        x = self.ln2(x)  # (batch_size, num_patches, embed_dim)
        x = self.ffn(x)  # (batch_size, num_patches, embed_dim)
        x = x + x_res  # (batch_size, num_patches, embed_dim)

        return x


class SegmentationDecoderLayer(nn.Module):
    """Layer for decoding 1D patch embeddings into patches.

    Each patch embedding is first decoded into a patch **without** a positional encoding, then the patches are stitched into images:

                                    __PATCHES__    __IMAGE__
                                    +---+---+      +---+---+
     _EMBEDDINGS_                   |   |   |      |       |
     4x |IIII...IIIIIIIIIIIII|  ->  +---+---+  ->  |       |
                                    |   |   |      |       |
                                    +---+---+      +-------+

    """

    def __init__(
        self, embed_dim: int, patch_size: int, num_classes: int
    ) -> None:
        """Creates a new SegmentationDecoderLayer.

        :param embed_dim: the size of the 1D patch embeddings to be expanded into patches.
        :param num_classes: the number of classes the output will have.
        """
        super().__init__()

        self.decoder: nn.Module = nn.Sequential(
            nn.ConvTranspose2d(
                embed_dim,
                num_classes,
                kernel_size=patch_size,
                stride=patch_size,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Decodes batches of sets of 1D image patches into (`num_classes`, `patch_size`, `patch_size`) patches, then stitches them together to form a new image.

        :param x: a batched tensor containing sets of 1D patch embeddings with shape (b, num_patches, embed_dim).

        :returns: an batched tensor containing images of shape (b, num_classes, img_size[1], img_size[2])
        """
        x = self.decoder(x)

        return x


class SegmentingVisionTransformer(nn.Module):
    """Vision transformer model. Predicts a class for each pixel in the image passed to it.

    Images go through the following pipeline for segmentation:
    1. Images are encoded using a :class:`PatchEmbedLayer`.
    2. Resulting patch encodings are processed with self-attention using a series of :class:`TransformerEncoderLayer`.
    3. Processed patch encodings are decoded using a :class:`SegmentationDecoderLayer`.
    4a. Segmentation values are passed through a sigmoid function FOR BINARY CLASSIFICATION ONLY.
    4b. Segmentation values are passed through a softmax function FOR NONBINARY CLASSIFICATION ONLY.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        num_classes: int,
        patch_size: int = 16,
        num_heads: int = 12,
        num_layers: int = 12,
        embed_dim: int = 768,
        hidden_dim: Optional[int] = None,
    ) -> None:
        """Creates a new SegmentingVisionTransformer model.

        :param img_shape: a tuple representing the input shape for the model.
        :param num_classes: the number of channels the output segmentation will have.
        :param patch_size: the size of the patches the processed patch encodings will be decoded to. Defaults to 16.
        :param num_heads: the number of heads used in the self-attention block. Defaults to 12.
        :param num_layers: the number of :class:`TransformerEncoderLayer`s the model will contain. Defaults 12.
        :param embed_dim: the size of the 1D patch encodings. Defaults to 768.
        :param hidden_dim: the size of the hidden dimension in the feedforward block. Defaults to `embed_dim`.
        """
        super().__init__()

        self.num_channels: int = img_shape[0]
        self.img_size: int = img_shape[1]
        self.patch_size: int = patch_size

        self.encoder: nn.Module = PatchEmbedLayer(
            img_shape, patch_size, embed_dim
        )

        self.processor: nn.Module = nn.Sequential(
            *[
                TransformerEncoderLayer(embed_dim, num_heads, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.decoder: nn.Module = SegmentationDecoderLayer(
            embed_dim, patch_size, num_classes
        )
        if num_classes == 1:
            self.out_func: nn.Module = nn.Sigmoid()
        else:
            self.out_func: nn.Module = nn.Softmax()

    def forward(self, x: Tensor) -> Tensor:
        """Generates a segmentation map of `x`.

        :param x: the batched tensor of shape (`b`, `img_size[0]`, `img_size[1]`, `img_size[2]`) containing images.

        :returns: the batched tensor of shape (`b`, `num_classes`, `img_size[1]`, `img_size[2]`) segmenting `x`.
        """
        # x: (batch_size, channels, height, width)
        x = self.encoder(x)  # (batch_size, num_patches, embed_dim)
        x = self.processor(x)  # (batch_size, num_patches, embed_dim)
        x = x.transpose(
            1, 2
        ).reshape(  # (batch_size, embed_dim, num_patches, num_patches)
            x.size(0),
            -1,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
        )
        x = self.decoder(x)  # (batch_size, num_classes, height, width)
        x = self.out_func(x)

        return x
