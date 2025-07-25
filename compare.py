# Copyright (c) 2025 Coolant
# Licensed under the same license terms as VGGT (Meta Platforms, Inc. and affiliates)
"""
Compare VGGT camera‑pose predictions from two model variants (FiLM‑conditioned fine‑tuned vs. vanilla 
pre‑trained) against COLMAP ground‑truth poses contained in a *sparse* directory, and visualize all three 
sets side‑by‑side in the browser with **viser**.

Usage
-----
python vggt_compare_poses.py \
    --image_folder   /path/to/scene/images \
    --sparse_folder  /path/to/scene/sparse \
    --film_ckpt      /path/to/finetuned_film.pt \
    [--no_points] [--port 8080] [--mask_sky]

The script will spin up a viser server (default http://localhost:8080) showing:
* **Red**   – camera frustums predicted by the FiLM fine‑tuned checkpoint
* **Green** – camera frustums predicted by the vanilla pre‑trained checkpoint (no FiLM)
* **Blue**  – ground‑truth cameras read from COLMAP `cameras.txt`/`images.txt`
Optionally (default) the point cloud from the FiLM model is visualised for context.
"""

import argparse
import glob
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
from numpy.linalg import svd

# --- VGGT & helper utilities -------------------------------------------------
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# --- COLMAP reading utilities ----------------------------------------------
try:
    from training.data.read_write_model import read_model, qvec2rotmat  # your project util (DTU‑like)
except ImportError as e:  # fall back to original colmap‑utils pip package if available
    raise ImportError("Could not import COLMAP read_model utility – make sure it is in PYTHONPATH") from e

# ----------------------------------------------------------------------------
COLORS = {
    "film": (1.0, 0.0, 0.0, 1.0),   # Red
    "pretrain": (0.0, 1.0, 0.0, 1.0), # Green
    "gt": (0.0, 0.0, 1.0, 1.0),       # Blue
}

# -----------------------------------------------------------------------------
# Similarity alignment (Umeyama) ----------------------------------------------
# -----------------------------------------------------------------------------


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Return similarity that aligns *src* points to *dst*.

    src, dst: (N,3) arrays of corresponding 3-D points.
    Returns (scale, R, t) such that:  dst ≈ scale * R @ src + t
    """
    assert src.shape == dst.shape and src.shape[1] == 3, "Input shapes must match N×3"

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Compute covariance matrix
    cov = dst_centered.T @ src_centered / src.shape[0]

    U, S, Vt = svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt

    if with_scale:
        var_src = (src_centered ** 2).sum() / src.shape[0]
        scale = (S @ np.ones_like(S)) / var_src
    else:
        scale = 1.0

    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def apply_similarity_to_extrinsics(extri: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply similarity transform to camera-to-world extrinsics.

    extri: (S,3,4) camera-to-world matrices.
    similarity: X' = scale*R@X + t.
    Only translation column is affected; rotation remains in world coords, so we rotate it too.
    """
    out = extri.copy()
    out[:, :3, 3] = scale * (R @ extri[:, :3, 3].T).T + t
    out[:, :3, :3] = R @ out[:, :3, :3]  # rotate orientation into aligned frame
    return out

# ----------------------------------------------------------------------------

def load_colmap_poses(sparse_dir: str, image_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load extrinsics & intrinsics from a COLMAP *sparse* directory and return them ordered
    to match *image_names* (list of absolute paths)."""
    # read_model may return 2 or 3 values depending on version
    rm_out = read_model(sparse_dir, ext=".txt")
    if len(rm_out) == 3:
        cameras, images, _ = rm_out
    else:
        cameras, images = rm_out

    name_to_idx = {os.path.basename(p): i for i, p in enumerate(image_names)}
    S = len(image_names)
    extri = np.zeros((S, 3, 4), dtype=np.float32)
    intri = np.zeros((S, 3, 3), dtype=np.float32)

    for img in images.values():
        if img.name not in name_to_idx:
            continue  # skip cameras without matching RGB in our folder
        i = name_to_idx[img.name]
        R = qvec2rotmat(img.qvec)
        t = img.tvec.reshape(3, 1)
        extri[i] = np.concatenate([R, t], axis=1)
        cam = cameras[img.camera_id]
        intri[i] = colmap_intrinsics_to_opencv(cam)

    return extri, intri


def colmap_intrinsics_to_opencv(cam) -> np.ndarray:
    """Convert COLMAP camera parameters to a 3×3 OpenCV intrinsic matrix K."""
    model = cam.model.upper()
    p = cam.params
    if model == "PINHOLE":
        fx, fy, cx, cy = p
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        fx = fy = f
    elif model.startswith("OPENCV"):
        fx, fy, cx, cy, *_ = p
    elif model in {"SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy, *_ = p
        fx = fy = f
    else:
        raise NotImplementedError(f"Camera model {model} not supported in this viewer")
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K

# ----------------------------------------------------------------------------

def load_vggt(checkpoint: str, use_film: bool, device: str) -> VGGT:
    model = VGGT(use_film=use_film, enable_point=False, enable_depth=False, enable_track=False)
    if checkpoint is None:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state = torch.hub.load_state_dict_from_url(_URL, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        state = torch.load(checkpoint, map_location=device)
        state_dict = state.get("model", state)
        model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model

# ----------------------------------------------------------------------------

def predict_poses(model: VGGT, images: torch.Tensor, metadata: torch.Tensor | None, device: str,
                  dtype: torch.dtype) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Forward pass and convert to numpy extrinsics & intrinsics.
    Returns (extrinsic, intrinsic, aux_predictions)."""
    images = images.to(device)
    if metadata is not None:
        metadata = metadata.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        preds = model(images, metadata=metadata)

    extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])

    out = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in preds.items()}
    return extri, intri, out

# ----------------------------------------------------------------------------

# NOTE: viser <0.2 does not accept line_color kwarg; set after creation.
def add_camera_set(server: viser.ViserServer, name_prefix: str, color_rgba: Tuple[float, float, float, float],
                   extri: np.ndarray, images: np.ndarray, intri: np.ndarray | None = None):
    """Add a set of camera frames + frustums into the viser scene."""
    S = extri.shape[0]

    frames = []
    frustums = []

    for i in range(S):
        cam2world = extri[i]
        T = viser_tf.SE3.from_matrix(cam2world)
        frame = server.scene.add_frame(
            f"{name_prefix}_frame_{i}",
            wxyz=T.rotation().wxyz,
            position=T.translation(),
            axes_length=0.04,
            axes_radius=0.002,
            origin_radius=0.002,
        )
        # Use the first channel of color tuple to modulate axis length slightly for visual separation
        # Frustum FOV: derive from intrinsics when available else heuristic
        img = images[i]
        img_vis = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
        h, w = img_vis.shape[:2]
        if intri is not None:
            fx = float(intri[i, 0, 0]) if intri is not None else None
            if fx is not None and fx > 0:
                fov = 2.0 * np.arctan2(w / 2.0, fx)
            else:
                fov = np.deg2rad(60.0)
        else:
            fov = 60.0 * np.pi / 180.0
        frustum = server.scene.add_camera_frustum(
            f"{name_prefix}_frustum_{i}",
            fov=fov,
            aspect=w / h,
            scale=0.05,
            line_width=1.0,
            image=img_vis,
        )
        # Try to set color if attribute exists (depends on viser version)
        if hasattr(frustum, "line_color"):
            try:
                frustum.line_color = color_rgba
            except Exception:
                pass
        # Link frustum transform to frame
        frustum.wxyz = frame.wxyz
        frustum.position = frame.position

        frames.append(frame)
        frustums.append(frustum)

    return frames, frustums

# ----------------------------------------------------------------------------

def to_np_extri(extri: np.ndarray | torch.Tensor) -> np.ndarray:
    """Ensure extrinsics are numpy with shape (S,3,4)."""
    if isinstance(extri, torch.Tensor):
        extri = extri.detach().cpu().numpy()
    if extri.ndim == 4 and extri.shape[0] == 1:
        extri = extri[0]  # remove batch dim
    if extri.ndim == 2 and extri.shape == (3, 4):
        extri = extri[None]  # single camera -> length-1 sequence
    return extri


def to_np_intri(intri: np.ndarray | torch.Tensor | None) -> np.ndarray | None:
    if intri is None:
        return None
    if isinstance(intri, torch.Tensor):
        intri = intri.detach().cpu().numpy()
    if intri.ndim == 4 and intri.shape[0] == 1:
        intri = intri[0]
    if intri.ndim == 2 and intri.shape == (3, 3):
        intri = intri[None]
    return intri

# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Compare VGGT pose predictions vs. COLMAP ground‑truth in viser")
    parser.add_argument("--image_folder", required=True, help="Folder with RGB images (input order is alphabetical)")
    parser.add_argument("--sparse_folder", required=True, help="COLMAP sparse model directory")
    parser.add_argument("--film_ckpt", required=True, help="Path to fine‑tuned FiLM checkpoint")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no_points", action="store_true", help="Do NOT visualise point cloud")
    parser.add_argument("--mask_sky", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # ---------------------------------------------------------------------
    # 1. Load images (and metadata for FiLM)
    # ---------------------------------------------------------------------
    image_paths = sorted(glob.glob(os.path.join(args.image_folder, "*")))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {args.image_folder}")

    # FiLM model expects metadata tensor; load with return_metadata=True
    imgs_film, metadata = load_and_preprocess_images(image_paths, mode="pad", return_metadata=True)
    imgs_pre = imgs_film.clone()  # same preprocessed images for the no‑FiLM model

    # ---------------------------------------------------------------------
    # 2. Run VGGT (FiLM) ---------------------------------------------------
    # ---------------------------------------------------------------------
    model_film = load_vggt(args.film_ckpt, use_film=True, device=device)
    extri_film, intri_film, preds_film = predict_poses(model_film, imgs_film, metadata, device, dtype)

    # Optionally obtain point cloud for context (from depth head)
    if not args.no_points and "depth" in preds_film:
        depth_map = preds_film["depth"]  # (S, H, W, 1)
        world_pts = unproject_depth_map_to_point_map(depth_map, extri_film, intri_film)
        colors = imgs_film.cpu().numpy().transpose(0, 2, 3, 1)  # (S,H,W,3)
    else:
        world_pts = None
        colors = None

    # ---------------------------------------------------------------------
    # 3. Run VGGT (pre‑trained, no FiLM) ----------------------------------
    # ---------------------------------------------------------------------
    model_pre = load_vggt(checkpoint=None, use_film=False, device=device)
    extri_pre, intri_pre, _ = predict_poses(model_pre, imgs_pre, None, device, dtype)

    # ---------------------------------------------------------------------
    # 4. Load COLMAP ground‑truth -----------------------------------------
    # ---------------------------------------------------------------------
    extri_gt, intri_gt = load_colmap_poses(args.sparse_folder, image_paths)

    # ---------------------------------------------------------------------
    # 5. Start viser -------------------------------------------------------
    # ---------------------------------------------------------------------
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Ensure consistent shapes and convert to numpy
    extri_film = to_np_extri(extri_film)
    extri_pre = to_np_extri(extri_pre)
    extri_gt = to_np_extri(extri_gt)

    intri_film = to_np_intri(intri_film)
    intri_pre = to_np_intri(intri_pre)
    intri_gt = to_np_intri(intri_gt)

    # Align predicted camera centers to COLMAP using Umeyama similarity (scale+rot+trans)
    try:
        scale_film, R_film, t_film = umeyama_alignment(extri_film[:, :3, 3], extri_gt[:, :3, 3])
        extri_film = apply_similarity_to_extrinsics(extri_film, scale_film, R_film, t_film)

        scale_pre, R_pre, t_pre = umeyama_alignment(extri_pre[:, :3, 3], extri_gt[:, :3, 3])
        extri_pre = apply_similarity_to_extrinsics(extri_pre, scale_pre, R_pre, t_pre)
    except Exception as e:
        print(f"[Warning] Alignment failed: {e}. Showing raw poses.")

    # Recenter all for nicer visual range (subtract mean GT center)
    center = extri_gt[:, :3, 3].mean(axis=0)
    for arr in (extri_film, extri_pre, extri_gt):
        arr[:, :3, 3] -= center

    # (Optional) Point cloud ----------------------------------------------------------------
    if world_pts is not None:
        pts_centered = (world_pts.reshape(-1, 3) - center).astype(np.float32)
        clr = (colors.reshape(-1, 3) * 255).astype(np.uint8)
        server.scene.add_point_cloud(
            "pcd_film", points=pts_centered, colors=clr, point_size=0.001, point_shape="circle"
        )

    # Cameras -----------------------------------------------------------------------------
    imgs_np = imgs_film.cpu().numpy()
    film_frames, film_frust = add_camera_set(server, "film", COLORS["film"], extri_film, imgs_np, intri_film)
    pre_frames, pre_frust = add_camera_set(server, "pre", COLORS["pretrain"], extri_pre, imgs_np, intri_pre)
    gt_frames, gt_frust = add_camera_set(server, "gt", COLORS["gt"], extri_gt, imgs_np, intri_gt)

    # GUI checkboxes to toggle visibility
    gui_film = server.gui.add_checkbox("Show FiLM", initial_value=True)
    gui_pre = server.gui.add_checkbox("Show Pretrained", initial_value=True)
    gui_gt = server.gui.add_checkbox("Show COLMAP GT", initial_value=True)

    def set_visibility(handles, visible: bool):
        for h in handles:
            h.visible = visible

    @gui_film.on_update
    def _(_) -> None:
        set_visibility(film_frames + film_frust, gui_film.value)

    @gui_pre.on_update
    def _(_) -> None:
        set_visibility(pre_frames + pre_frust, gui_pre.value)

    @gui_gt.on_update
    def _(_) -> None:
        set_visibility(gt_frames + gt_frust, gui_gt.value)

    # Run forever -------------------------------------------------------------------------
    print(f"Server running on http://localhost:{args.port}  (press Ctrl+C to quit)")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
