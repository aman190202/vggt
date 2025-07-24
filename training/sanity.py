import torch
import numpy as np
from hydra import initialize, compose
from data.datasets.colmap import ColmapDataset

def save_ply(points, colors, filename):
    import open3d as o3d                
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    # Open3D expects colors in [0, 1] for floating-point arrays
    if points_visual_rgb.max() > 1.0:
        points_visual_rgb = points_visual_rgb / 255.0
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

# Usage example

# -------------------------------------------------------------------------
# Load the common_config from the Hydra YAML so that ColmapDataset gets all
# the attributes it expects (debug, img_size, etc.).
# -------------------------------------------------------------------------

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="default")  # or "default" if you want the full cfg

# DictConfig that supports attribute access (cfg.data.train.common_config.debug, ...)
common_conf = cfg.data.train.common_config

dataset = ColmapDataset(common_conf=common_conf)
# -------------------------------------------------------------------------

batch = dataset.get_data()
save_ply(
    batch["world_points"][0].reshape(-1, 3), 
    batch["images"][0].reshape(-1, 3), 
    "debug.ply"
)
