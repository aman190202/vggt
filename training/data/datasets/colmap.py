import os
import os.path as osp
import logging
import random

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset
from data.read_write_model import read_model, qvec2rotmat


class ColmapDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split="train",
        COLMAP_DIR="/home/works/coolant-dataset/dataset",
        min_num_images=24,
        len_train=100000,
        len_test=10000,
        expand_ratio=8,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.COLMAP_DIR = COLMAP_DIR
        self.expand_ratio = expand_ratio
        self.min_num_images = min_num_images
        self.depth_max = 80

        self.len_train = len_train if split == "train" else len_test

        # Collect all scenes
        self.scene_map = {}  # (scene_name, image_id) → metadata
        for scene in sorted(os.listdir(COLMAP_DIR)):
            scene_path = osp.join(COLMAP_DIR, scene)
            if not osp.isdir(scene_path):
                continue

            sparse_dir = osp.join(scene_path, "sparse")
            if not osp.exists(osp.join(sparse_dir, "cameras.txt")):
                continue

            cameras, images, _ = read_model(sparse_dir)
            for img_id, img_data in images.items():
                self.scene_map[(scene, img_id)] = {
                    "image": img_data,
                    "camera": cameras[img_data.camera_id],
                    "scene_path": scene_path,
                    "image_path": osp.join(scene_path, "images", img_data.name)
                }

        self.entries = list(self.scene_map.keys())
        logging.info(f"Loaded {len(self.entries)} images across {len(set(k[0] for k in self.entries))} scenes")

    def __len__(self):
        return self.len_train

    def get_data(
        self,
        seq_index=None,
        img_per_seq=1,
        seq_name=None,
        ids=None,
        aspect_ratio=1.0,
    ):
        # ``ids`` will internally hold (scene_name, image_id) tuples while we build the batch.
        # But the final "ids" field returned to the caller should match the other datasets:
        # a 1-D NumPy array of integers (image indices).  We therefore keep a separate list
        # ``entry_tuples`` for internal use and convert to np.ndarray before returning.

        if self.inside_random and self.training:
            entry_tuples = random.sample(self.entries, k=img_per_seq)
        elif ids is None:
            entry_tuples = [random.choice(self.entries)]
        else:
            # If the caller provides ids as list of tuples (scene_name, image_id) keep as is
            entry_tuples = ids

        # Numeric ids array (image ids only) for downstream code
        ids_numeric = np.array([img_id for (_, img_id) in entry_tuples], dtype=np.int32)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, extrinsics, intrinsics = [], [], []
        depths = []
        cam_points, world_points, point_masks, original_sizes = [], [], [], []

        for scene_name, image_id in entry_tuples:
            meta = self.scene_map[(scene_name, image_id)]
            img_data = meta["image"]
            cam_data = meta["camera"]
            img_path = meta["image_path"]

            image = read_image_cv2(img_path)
            # Use one channel of the RGB image (convert to grayscale) as a pseudo-depth map.
            # This keeps the data pipeline unchanged while providing a non-None depth array.
            depth_placeholder = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            original_size = np.array(image.shape[:2])

            # intrinsics
            if cam_data.model in ["SIMPLE_PINHOLE", "PINHOLE"]:
                fx, fy = cam_data.params[0], cam_data.params[1] if cam_data.model == "PINHOLE" else cam_data.params[0]
                cx, cy = cam_data.params[-2], cam_data.params[-1]
            else:
                raise NotImplementedError(f"Unsupported camera model: {cam_data.model}")

            K = np.eye(3)
            K[0, 0], K[1, 1] = fx, fy
            K[0, 2], K[1, 2] = cx, cy

            R = qvec2rotmat(img_data.qvec)
            t = img_data.tvec.reshape(3, 1)
            extrinsic = np.concatenate([R, t], axis=1)

            (
                image,
                depth_map,
                extrinsic,
                K,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_placeholder,
                extrinsic,
                K,
                original_size,
                target_image_shape,
                filepath=img_path,
            )

            images.append(image)
            depths.append(depth_map)
            intrinsics.append(K)
            extrinsics.append(extrinsic)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        batch = {
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

        return batch
