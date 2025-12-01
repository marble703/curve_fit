#!/usr/bin/env python3
"""
简单的三维可视化：绘制拟合曲线（x,z -> y=0 平面）并可叠加原始三维点。

用法示例：
  python3 visualize.py --fitted fitted_traj.txt --world data_world.txt

如果只有拟合曲线（x z 每行），也能绘制。
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import yaml

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def load_world(path):
    return np.loadtxt(path)


def load_fitted(path):
    # assume two columns: x z
    arr = np.loadtxt(path)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1,2)
    # return Nx3 with y=0
    xyz = np.zeros((arr.shape[0], 3))
    xyz[:,0] = arr[:,0]
    xyz[:,2] = arr[:,1]
    return xyz


def load_camera_params(path):
    if not path:
        return None
    if not os.path.exists(path):
        print('Camera file does not exist:', path)
        return None
    # try OpenCV FileStorage first
    fs = None
    try:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    except Exception as exc:  # catch cv2.error or SystemError
        print('OpenCV FileStorage failed:', exc)
        fs = None
    if fs is not None and fs.isOpened():
        K = fs.getNode('K').mat()
        dist = fs.getNode('dist').mat()
        R = fs.getNode('R').mat()
        t = fs.getNode('t').mat()
        fs.release()
        return {'K': K, 'dist': dist, 'R': R, 't': t}

    # fallback to PyYAML parsing
    try:
        with open(path, 'r') as f:
            raw = f.read()
        clean = raw.replace('!!opencv-matrix', '')
        data = yaml.safe_load(clean)
    except Exception as exc:
        print('PyYAML failed to parse camera file:', path, exc)
        return None

    def parse_mat(node):
        if node is None:
            return None
        if isinstance(node, dict) and 'data' in node:
            arr = np.array(node['data'], dtype=np.float64)
            rows = node.get('rows')
            cols = node.get('cols')
            if rows is not None and cols is not None:
                try:
                    return arr.reshape((rows, cols))
                except Exception:
                    pass
            return arr
        return np.array(node, dtype=np.float64)

    return {
        'K': parse_mat(data.get('K')),
        'dist': parse_mat(data.get('dist')),
        'R': parse_mat(data.get('R')),
        't': parse_mat(data.get('t')),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitted', required=True, help='拟合轨迹文件，每行: x z')
    parser.add_argument('--world', default=None, help='原始三维坐标文件，每行: x y z')
    parser.add_argument('--image', default=None, help='可选：要叠加投影的图片路径')
    parser.add_argument('--camera', default=None, help='可选：相机内参文件（OpenCV YAML，包含 K, dist, R, t）')
    parser.add_argument('--overlay-save', default=None, help='可选：保存带投影的图片到文件')
    parser.add_argument('--no-3d', action='store_true', help='不显示 3D 可视化，只有图像叠加')
    parser.add_argument('--show-camera', action='store_true', help='是否显示相机位置（默认不显示）')
    parser.add_argument('--save', default=None, help='保存图片到文件（可选）')
    args = parser.parse_args()

    fitted = load_fitted(args.fitted)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # plot fitted curve
    ax.plot(fitted[:,0], fitted[:,1]*0.0, fitted[:,2], '-r', label='fitted')

    if args.world is not None:
        world = load_world(args.world)
        ax.scatter(world[:,0], world[:,1], world[:,2], c='b', s=20, label='world')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 1. 收集所有点的坐标
    if args.world is not None:
        all_pts = np.vstack([fitted, world])
    else:
        all_pts = fitted

    # 2. 计算各轴数据范围
    max_range = np.ptp(all_pts, axis=0)          # [dx, dy, dz]
    max_len = max_range.max()                    # 最大跨度

    # 3. 以各轴中心为基准，重新计算等长显示区间
    centers = all_pts.mean(axis=0)               # [cx, cy, cz]
    half = max_len / 2.0
    x_min, x_max = centers[0] - half, centers[0] + half
    y_min, y_max = centers[1] - half, centers[1] + half
    z_min, z_max = centers[2] - half, centers[2] + half

    # 4. 应用
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.legend()
    ax.view_init(elev=20., azim=-60)

    # If camera and image provided, also project fitted trajectory onto image
    if args.image is not None and args.camera is not None:
        cam = load_camera_params(args.camera)
        if cam is None or cam.get('K') is None:
            print('Failed to load camera intrinsics from', args.camera)
        else:
            img = cv2.imread(args.image)
            if img is None:
                print('Failed to read image:', args.image)
            else:
                K = np.array(cam.get('K'), dtype=np.float64)
                dist = cam.get('dist')
                R = cam.get('R')
                t = cam.get('t')
                # prepare 3D points (Nx3)
                pts3 = fitted.astype(np.float64)
                if R is None or R.size == 0:
                    R = np.eye(3, dtype=np.float64)
                if t is None or t.size == 0:
                    t = np.zeros((3,1), dtype=np.float64)

                try:
                    rvec, _ = cv2.Rodrigues(R)
                except Exception:
                    rvec = np.array(R, dtype=np.float64).reshape(3,1)
                tvec = np.array(t, dtype=np.float64).reshape(3,1)

                if dist is None or dist.size == 0:
                    dist = np.zeros((5,), dtype=np.float64)
                else:
                    dist = np.array(dist, dtype=np.float64).reshape(-1)

                img_pts, _ = cv2.projectPoints(pts3, rvec, tvec, K, dist)
                img_pts = img_pts.reshape(-1,2)

                overlay = img.copy()
                pts_int = np.round(img_pts).astype(np.int32)
                if pts_int.shape[0] >= 2:
                    cv2.polylines(overlay, [pts_int], isClosed=False, color=(0,255,0), thickness=2)
                for p in pts_int:
                    x, y = int(p[0]), int(p[1])
                    cv2.circle(overlay, (x,y), 3, (0,0,255), -1)

                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10,8))
                plt.imshow(overlay_rgb)
                plt.axis('off')
                plt.title('Projected fitted trajectory')
                if args.overlay_save:
                    plt.savefig(args.overlay_save, dpi=200, bbox_inches='tight')
                    print('Saved overlay to', args.overlay_save)
                else:
                    plt.show()

    # show or save 3D plot unless disabled
    if not args.no_3d:
        if args.save:
            plt.savefig(args.save, dpi=200)
            print('Saved figure to', args.save)
        else:
            plt.show()


if __name__ == '__main__':
    main()
