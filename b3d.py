"""Shared B3D helpers: paths and the per-video crop region and road mask."""
from __future__ import annotations

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
# Segments per source video, and the five test segments (jncN/test/teNN.mp4) of every B3D dataset.
NUM_SEGMENTS = 18
TEST_SEGMENTS = [1, 4, 8, 12, 13]


def segment_range(total_frames: int, segment: int) -> tuple[int, int]:
    """Source-frame range [start, end) of a segment; mirrors tetris get_segment_frame_range."""
    size = total_frames / NUM_SEGMENTS
    return int(segment * size), min(int((segment + 1) * size), total_frames)


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
