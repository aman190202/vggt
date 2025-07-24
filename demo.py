import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401 – activates 3-D proj
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# ------------------------------------------------------------------
# 1.  Run the model exactly as you did in demo.py
# ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT(enable_camera=True, enable_point=False,
             enable_depth=False, enable_track=False).to(device)
model.load_state_dict(torch.load("/home/works/vggt/training/logs/exp003/ckpts/checkpoint.pt"), strict=False)

# ─── your image list here ─────────────────────────────────────────
import glob
image_root = "/home/works/GreenTrees/GT_AV_F25_P087/images"
image_names = sorted(glob.glob(f"{image_root}/*"))
meta = True
if meta :
    images, metadata = load_and_preprocess_images(image_names, return_metadata=True)
    metadata = metadata.to(device)
else:
    images = load_and_preprocess_images(image_names, return_metadata=False)
images = images.to(device)

if meta : 
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(images, metadata=metadata)
else: 
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(images)

pose_enc = preds["pose_enc"].cpu()      # shape (S, 9)

# ------------------------------------------------------------------
# 2.  Convert pose encodings → extrinsic & intrinsic
# ------------------------------------------------------------------
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    pose_enc,                 # (S, 9)
    images.shape[-2:]         # (H, W) – needed to produce intrinsics
)                             # extrinsic: (S, 3, 4)  intrinsic: (S, 3, 3)

if extrinsic.shape[0] == 1:
    extrinsic = extrinsic.squeeze(0)      # [S, 3, 4]
else:
    extrinsic = extrinsic.reshape(-1, 3, 4)   # flattens B and S

# ------------------------------------------------------------------
# 3.  Extract camera centres & axes for plotting
# ------------------------------------------------------------------
cam_centres = []
cam_dirs    = []     # camera forward axes (negative Z in camera space)

for i in range(extrinsic.shape[0]):
    # extrinsic is [R | t] mapping world→camera
    R, t = extrinsic[i, :3, :3], extrinsic[i, :3, 3]
    # Camera centre in world coords:  C = -Rᵀ t
    C = -R.T @ t
    cam_centres.append(C)

    # Forward direction of camera in world coordinates
    forward = R.T @ np.array([0, 0, -1.0])
    cam_dirs.append(forward / np.linalg.norm(forward))

cam_centres = np.stack(cam_centres)     # (S, 3)
cam_dirs    = np.stack(cam_dirs)        # (S, 3)

# ------------------------------------------------------------------
# 4.  Plot
# ------------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')

# Scatter the camera centres
ax.scatter(cam_centres[:, 0], cam_centres[:, 1], cam_centres[:, 2],
           c='r', s=40, label='Camera centres')

# Draw viewing axes (scaled arrows)
scale = np.linalg.norm(cam_centres.std(axis=0)) * 0.2
for C, d in zip(cam_centres, cam_dirs):
    ax.quiver(C[0], C[1], C[2],
              d[0], d[1], d[2],
              length=scale, color='b', linewidth=1)

# ------------------------------------------------------------------
# 5.  Make scale uniform (equal aspect ratio)
# ------------------------------------------------------------------
# Matplotlib < 3.4 lacks an official way, so we rescale axes manually.
# For newer Matplotlib versions you can simply call:
#   ax.set_box_aspect([1, 1, 1])
# We keep a backward-compatible fallback.

try:
    # Matplotlib 3.4+: built-in method
    ax.set_box_aspect([1, 1, 1])
except AttributeError:
    # Fallback for older versions – match the largest span to all axes
    mins = cam_centres.min(axis=0)
    maxs = cam_centres.max(axis=0)
    span = max(maxs - mins)
    mid  = (maxs + mins) / 2
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title(f'{len(cam_centres)} estimated camera poses')
ax.legend()
plt.tight_layout()
plt.show()