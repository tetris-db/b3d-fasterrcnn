"""Shared B3D helpers: paths, per-video crop region and mask, and inference tiling."""
from __future__ import annotations

import math
import os
from xml.etree import ElementTree

import cv2
import numpy as np


# Host /polyis-data/sources is mounted read-only at /sources by ./run.
SOURCES = '/sources'
# B3D human-labeled training crops (COCO JSON + images), from the b3d-project/b3d repo.
VISION = os.path.join(SOURCES, 'b3d-vision', 'vision')
# CVAT XML holding each video's 'crop' box and 'domain' polygons.
MASKS = os.path.join(SOURCES, 'b3d', 'annotations.xml')
# Native-resolution (3840x2160) source videos of the four B3D test datasets.
VIDEOS = {
    'jnc0': os.path.join(SOURCES, 'jnc0', 'jnc00.mp4'),
    'jnc2': os.path.join(SOURCES, 'jnc2', 'jnc02.mp4'),
    'jnc6': os.path.join(SOURCES, 'jnc6', 'jnc06.mp4'),
    'jnc7': os.path.join(SOURCES, 'jnc7', 'jnc07.mp4'),
}
# Detector input size; RF-DETR-Large's default square resolution.
TILE = 704
# Tile overlap in pixels; covers the p99 longest side of B3D labeled vehicles (227 px).
OVERLAP = 256


def load_crop_and_mask(video_path: str, width: int, height: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Return the crop box (left, top, right, bottom) and the uint8 domain mask inside it.

    Mirrors tetris polyis.utilities.build_b3d_mask_and_crop so detections line up with
    the existing ground-truth coordinates.
    """
    # Find this video's entry in the CVAT XML (keyed by the video name with .jpg).
    root = ElementTree.parse(MASKS).getroot()
    name = os.path.basename(video_path).replace('.mp4', '.jpg')
    image = root.find(f'.//image[@name="{name}"]')
    assert image is not None, f'No mask entry for {name}'
    # Parse the crop box used by the tetris preprocessing.
    box = image.find('.//box[@label="crop"]')
    assert box is not None, f'No crop box for {name}'
    left, top = round(float(box.attrib['xtl'])), round(float(box.attrib['ytl']))
    right, bottom = round(float(box.attrib['xbr'])), round(float(box.attrib['ybr']))
    # Rasterize the union of all 'domain' polygons on the full frame.
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in image.findall('.//polygon[@label="domain"]'):
        points = np.array([[float(v) for v in p.split(',')] for p in polygon.attrib['points'].split(';')])
        cv2.fillPoly(mask, [points.round().astype(np.int32)], 255)
    # Keep only the crop region of the mask.
    return (left, top, right, bottom), mask[top:bottom, left:right]


def tile_origins(length: int, tile: int = TILE, overlap: int = OVERLAP) -> list[int]:
    """Evenly spaced tile starts covering [0, length) with at least `overlap` pixels shared."""
    # A region no longer than one tile needs a single tile (padded by the caller).
    if length <= tile:
        return [0]
    # Fewest tiles whose stride does not exceed tile - overlap, with both edge tiles flush.
    count = math.ceil((length - tile) / (tile - overlap)) + 1
    return [round(i * (length - tile) / (count - 1)) for i in range(count)]


def tiles(mask: np.ndarray, tile: int = TILE, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    """Top-left corners of tiles over the crop region that contain any masked-in pixel."""
    height, width = mask.shape
    return [
        (x, y)
        for y in tile_origins(height, tile, overlap)
        for x in tile_origins(width, tile, overlap)
        # Skip tiles that fall entirely outside the road domain.
        if mask[y:y + tile, x:x + tile].any()
    ]


if __name__ == '__main__':
    # Self-check for tile_origins: full coverage, flush edges, and the overlap bound.
    for length in [100, 704, 705, 1240, 1938, 2031, 5000]:
        starts = tile_origins(length)
        assert starts[0] == 0 and starts[-1] + TILE >= length
        if length > TILE:
            assert starts[-1] + TILE == length
            assert all(b - a <= TILE - OVERLAP for a, b in zip(starts, starts[1:]))
    print('ok')
