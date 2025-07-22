import os
import os.path as osp
import logging
import random
from typing import List, Tuple

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset
from data.read_write_model import read_model, qvec2rotmat


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Convert 3×4 world-to-cam matrix to 3×4 cam-to-world (same shape)."""
    R = w2c[:, :3]           # (3,3)
    t = w2c[:, 3:]           # (3,1)
    R_inv = R.T
    t_inv = -R_inv @ t
    return np.hstack([R_inv, t_inv])   

def colmap_intrinsics_to_opencv(cam) -> np.ndarray:
    """
    Convert COLMAP camera parameters to a 3×3 OpenCV intrinsic matrix K.
    Supported models: PINHOLE, SIMPLE_PINHOLE, OPENCV*, SIMPLE_RADIAL, RADIAL.
    Distortion terms are ignored – add them to your augmentations if needed.
    """
    model = cam.model.upper()
    p = cam.params                     # numpy.ndarray, float64

    if model == "PINHOLE":             # fx, fy, cx, cy
        fx, fy, cx, cy = p
    elif model == "SIMPLE_PINHOLE":    # f, cx, cy   → fx = fy = f
        f, cx, cy = p
        fx = fy = f
    elif model.startswith("OPENCV"):   # fx, fy, cx, cy, ...
        fx, fy, cx, cy = p[:4]
    elif model in {"SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = p[:3]
        fx = fy = f
    else:
        raise NotImplementedError(f"Camera model “{model}” not supported.")

    K = np.eye(3, dtype=np.float32)
    K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = cx, cy
    return K


def colmap_extrinsics_to_opencv(img) -> np.ndarray:
    """
    Build the 3×4 [R|t] matrix that brings world points into the
    **camera (OpenCV) coordinate frame**:

        X_cam = R · X_world + t

    COLMAP’s qvec / tvec are already given in that convention, so we simply
    concatenate them.
    """
    R = qvec2rotmat(img.qvec).astype(np.float32)     # (3,3)
    t = img.tvec.astype(np.float32).reshape(3, 1)    # (3,1)
    return np.hstack([R, t])                         # (3,4)
# --------------------------------------------------------------------------- #


class ColmapDataset(BaseDataset):
    """
    Dataset that reads `cameras.txt` and `images.txt` from every <sparse> folder
    under COLMAP_DIR/scene_name/ and returns cameras in **OpenCV convention**.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        COLMAP_DIR: str = "/home/works/coolant-dataset/dataset",
        min_num_images: int = 24,
        len_train: int = 100_000,
        len_test: int = 10_000,
        expand_ratio: int = 8,
    ):
        super().__init__(common_conf=common_conf)

        # --- stash config -------------------------------------------------- #
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.COLMAP_DIR = COLMAP_DIR
        self.expand_ratio = expand_ratio
        self.min_num_images = min_num_images
        self.depth_max = 80.0
        self.len_train = len_train if split == "train" else len_test
        # ------------------------------------------------------------------- #

        # ------------------------------------------------------------------- #
        # Scan every scene folder that contains <scene>/sparse/cameras.txt
        # ------------------------------------------------------------------- #
        self.scene_map = {}         # key: (scene_name, image_id) → meta dict

        scene_names = sorted(os.listdir(COLMAP_DIR))
        if self.debug:                                 # ↓ quick dev loop
            scene_names = scene_names[:1]

        for scene in scene_names:
            scene_path = osp.join(COLMAP_DIR, scene)
            sparse_dir = osp.join(scene_path, "sparse")

            if not osp.isfile(osp.join(sparse_dir, "cameras.txt")):
                continue

            cameras, images = read_model(sparse_dir)   # .txt or .bin is fine

            # Skip very small reconstructions
            if len(images) < min_num_images:
                continue

            for img_id, img in images.items():
                img_path = osp.join(scene_path, "images", img.name)
                if not osp.isfile(img_path):
                    continue

                self.scene_map[(scene, img_id)] = {
                    "image": img,
                    "camera": cameras[img.camera_id],
                    "img_path": img_path,
                }

        self.entries: List[Tuple[str, int]] = list(self.scene_map.keys())
        logging.info(
            f"[ColmapDataset] {len(self.entries)} images | "
            f"{len(set(s for s, _ in self.entries))} scenes."
        )

    # ----------------------------------------------------------------------- #
    def __len__(self):
        return self.len_train
    # ----------------------------------------------------------------------- #

    def _sample_entries(self, img_per_seq: int, ids):
        """
        Decide which (scene, image_id) tuples to load for this call.
        """
        if ids is not None:                       # caller supplied explicit list
            return ids

        if self.inside_random and self.training:  # completely random sample
            return random.sample(self.entries, k=img_per_seq)

        # Default: contiguous chunk from the list (repeatable across epochs)
        start = random.randrange(len(self.entries) - img_per_seq + 1)
        return self.entries[start:start + img_per_seq]
    # ----------------------------------------------------------------------- #

    def get_data(
        self,
        seq_index=None,
        img_per_seq: int = 1,
        seq_name=None,           # <- kept for API compatibility
        ids=None,
        aspect_ratio: float = 1.0,
    ):
        entry_tuples = self._sample_entries(img_per_seq, ids)
        ids_numeric = np.fromiter((i for (_, i) in entry_tuples),
                                  dtype=np.int32)

        target_shape = self.get_target_shape(aspect_ratio)

        # --- containers ---------------------------------------------------- #
        images, depths = [], []
        intrinsics, extrinsics = [], []
        cam_points, world_points, point_masks, original_sizes = \
            [], [], [], []
        # ------------------------------------------------------------------- #

        for scene_name, img_id in entry_tuples:
            meta = self.scene_map[(scene_name, img_id)]
            img_data, cam_data = meta["image"], meta["camera"]
            img_path = meta["img_path"]

            # --- load RGB -------------------------------------------------- #
            rgb = read_image_cv2(img_path)
            original_size = np.asarray(rgb.shape[:2], dtype=np.int32)

            # --- placeholders --------------------------------------------- #
            depth_fake = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            # NB: depth_fake keeps the pipeline unchanged; replace later.

            # --- intrinsics / extrinsics ----------------------------------- #
            K = colmap_intrinsics_to_opencv(cam_data)
            ext = colmap_extrinsics_to_opencv(img_data)
            ext = w2c_to_c2w(ext)

            # --- standard preprocessing (resize, norm, points, …) --------- #
            (
                rgb_proc,
                depth_proc,
                ext_proc,
                K_proc,
                world_pts,
                cam_pts,
                mask_pts,
                _
            ) = self.process_one_image(
                rgb,
                depth_fake,
                ext,
                K,
                original_size,
                target_shape,
                filepath=img_path,
            )

            # --- collect --------------------------------------------------- #
            images.append(rgb_proc)
            depths.append(depth_proc)
            intrinsics.append(K_proc)
            extrinsics.append(ext_proc)
            cam_points.append(cam_pts)
            world_points.append(world_pts)
            point_masks.append(mask_pts)
            original_sizes.append(original_size)

        return {
            "seq_name": "colmap_batch",
            "ids": ids_numeric,
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
