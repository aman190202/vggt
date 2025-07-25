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
        COLMAP_DIR: str = "/home/works/sample",
        min_num_images: int = 48,
        len_train: int = 100,
        len_test: int = 10,
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
            if not hasattr(self, "scene_index"):
                self.scene_index = {}

            # Sort the image list by image name or image_id (your choice)
            sorted_imgs = sorted(images.items(), key=lambda kv: kv[1].name)  # or kv[0] for ID order
            self.scene_index[scene] = []  # create empty list for this scene

            # Skip very small reconstructions
            if len(images) < min_num_images:
                continue

            for img_id, img in sorted_imgs:
                img_path = osp.join(scene_path, "images", img.name)
                base_name = img.name
                for ext in [".jpeg", ".jpg", ".png"]:
                    if base_name.lower().endswith(ext):
                        base_name = base_name[: -len(ext)]
                        break
                depth_path = osp.join(scene_path, "depths", base_name + ".h5")
                gps_path = osp.join(scene_path, "metadata", img.name + ".json")

                if not osp.isfile(img_path):
                    continue

                self.scene_index[scene].append(img_id)  # <- Save image ID in order

                self.scene_map[(scene, img_id)] = {
                    "image": img,
                    "camera": cameras[img.camera_id],
                    "img_path": img_path,
                    "depth_path": depth_path,
                    "gps_path": gps_path,
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
        Returns a list of (scene_name, image_id) pairs in serial order.
        """
        if ids is not None:
            return ids

        if len(self.entries) == 0:
            raise RuntimeError(
                f"[ColmapDataset] No valid images found in '{self.COLMAP_DIR}'. "
                "Ensure each scene has a valid sparse reconstruction with registered images."
            )

        # --- serial sampling across scenes ---
        # Find scenes that have enough frames
        valid_scenes = [
            s for s in self.scene_index
            if len(self.scene_index[s]) >= img_per_seq
        ]

        if not valid_scenes:
            raise RuntimeError(
                f"[ColmapDataset] No scene has at least {img_per_seq} images. "
                "Check `min_num_images` and dataset size."
            )

        # Pick a scene randomly
        scene = random.choice(valid_scenes)
        scene_frames = self.scene_index[scene]  # ordered list of image_ids for this scene

        # Pick a contiguous window
        start = random.randint(0, len(scene_frames) - img_per_seq)
        selected_ids = scene_frames[start : start + img_per_seq]

        return [(scene, img_id) for img_id in selected_ids]

    # ----------------------------------------------------------------------- #

    def get_data(
        self,
        seq_index=None,
        img_per_seq: int = 48,
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
        metadatas = []  # Collect per-image metadata here
        # ------------------------------------------------------------------- #

        for scene_name, img_id in entry_tuples:
            meta = self.scene_map[(scene_name, img_id)]
            img_data, cam_data = meta["image"], meta["camera"]
            img_path = meta["img_path"]
            gps_path = meta["gps_path"]
            depth_path = meta["depth_path"]
            gps_data = None
            depth_data = None

            # Read GPS data if available
            if os.path.isfile(gps_path):
                import json
                with open(gps_path, "r") as f:
                    gps_data = json.load(f)

            # Read depth data from .h5 file if available
            if os.path.isfile(depth_path):
                import h5py
                with h5py.File(depth_path, "r") as f:
                    # Try common keys, fallback to first dataset if unknown
                    if "depth" in f:
                        depth_data = f["depth"][:]
                    elif len(f.keys()) > 0:
                        first_key = list(f.keys())[0]
                        depth_data = f[first_key][:]
                    else:
                        raise ValueError(f"HDF5 file {depth_path} is empty or malformed")

                # Validate depth_data shape
                if depth_data is None or not isinstance(depth_data, np.ndarray) or depth_data.ndim != 2:
                    raise ValueError(f"Depth map at {depth_path} is not a valid 2D array. Got shape: {getattr(depth_data, 'shape', None)}")
            else:
                # If depth file is not found, make depth all ones with the same size as the image
                rgb = read_image_cv2(img_path)
                height, width = rgb.shape[:2]
                depth_data = np.ones((height, width), dtype=np.float32)

            if gps_data is not None:
                metadata = [
                    gps_data.get("latitude", 0.0),
                    gps_data.get("longitude", 0.0),
                    gps_data.get("altitude", 0.0),
                    gps_data.get("pitch", 0.0),
                    gps_data.get("roll", 0.0),
                    gps_data.get("yaw", 0.0),
                ]
            else:
                metadata = [0,0,0,0,0,0]
            # --- load RGB -------------------------------------------------- #
            rgb = read_image_cv2(img_path)
            original_size = np.asarray(rgb.shape[:2], dtype=np.int32)

            # --- placeholders --------------------------------------------- #
            depth_fake = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            # NB: depth_fake keeps the pipeline unchanged; replace later.

            # --- intrinsics / extrinsics ----------------------------------- #
            K = colmap_intrinsics_to_opencv(cam_data)
            ext = colmap_extrinsics_to_opencv(img_data)

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
                depth_data,
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
            metadatas.append(metadata)

        # Convert metadata list to numpy array for proper tensor conversion
        metadatas = np.asarray(metadatas, dtype=np.float32)

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
            "metadata" : metadatas
        }
