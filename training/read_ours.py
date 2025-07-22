import numpy as np

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


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

from data.read_write_model import read_model
cameras, images = read_model("/home/works/coolant-dataset/dataset/GT_AV_F2025_P_P144/sparse")
for img_id, img in images.items():
    img_data, cam_data = img, cameras[img.camera_id]
    K = colmap_intrinsics_to_opencv(cam_data)
    ext = colmap_extrinsics_to_opencv(img_data)
    print(K)
    print(ext)
    break