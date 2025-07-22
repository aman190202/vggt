import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def qvec2rotmat(qvec):
    """Convert quaternion [qw, qx, qy, qz] to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y**2 - 2 * z**2,     2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,       1 - 2 * x**2 - 2 * z**2,   2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,       2 * y * z + 2 * x * w,     1 - 2 * x**2 - 2 * y**2]
    ])


def read_images_txt(path):
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("#") or line == "":
            continue
        elems = line.split()

        # COLMAP `images.txt` format (after header lines):
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME
        # Followed by a line of 2-D keypoints that we can ignore.
        # Some tools save a simplified 8-column version without CAMERA_ID.
        # We therefore accept both lengths.

        if len(elems) in (8, 10):
            image_id = int(elems[0])
            qvec = np.array(list(map(float, elems[1:5])))       # (4,)
            tvec = np.array(list(map(float, elems[5:8])))       # (3,)

            # In the 10-column case elems[8] is camera_id which we ignore here
            filename = elems[-1]

            R = qvec2rotmat(qvec)
            t = tvec.reshape(3, 1)
            extrinsic = np.hstack((R, t))                       # 3×4

            images[image_id] = {
                "R": R,
                "t": tvec,
                "extrinsic": extrinsic,
                "filename": filename
            }

    return images


def visualize_poses(images_dict):
    cam_centres = []
    cam_dirs = []

    for data in images_dict.values():
        R = data["R"]
        t = data["t"]
        C = -R.T @ t
        cam_centres.append(C)

        # forward = -Z axis in camera space → apply Rᵀ
        forward = R.T @ np.array([0, 0, -1.0])
        cam_dirs.append(forward / np.linalg.norm(forward))

    cam_centres = np.stack(cam_centres)
    cam_dirs = np.stack(cam_dirs)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(cam_centres[:, 0], cam_centres[:, 1], cam_centres[:, 2],
               c='r', s=40, label='Camera centres')

    scale = np.linalg.norm(cam_centres.std(axis=0)) * 0.2
    for C, d in zip(cam_centres, cam_dirs):
        ax.quiver(C[0], C[1], C[2], d[0], d[1], d[2],
                  length=scale, color='b', linewidth=1)

    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        mins = cam_centres.min(axis=0)
        maxs = cam_centres.max(axis=0)
        span = max(maxs - mins)
        mid = (maxs + mins) / 2
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{len(cam_centres)} COLMAP poses")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_path = "/home/works/coolant-dataset/dataset/GT_AV_F2025_P_P144/sparse"  # ← change this
    images_txt = os.path.join(base_path, "images.txt")
    cameras_txt = os.path.join(base_path, "cameras.txt")

    images = read_images_txt(images_txt)
    visualize_poses(images)
