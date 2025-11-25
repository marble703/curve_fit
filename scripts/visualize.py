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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitted', required=True, help='拟合轨迹文件，每行: x z')
    parser.add_argument('--world', default=None, help='原始三维坐标文件，每行: x y z')
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

    if args.save:
        plt.savefig(args.save, dpi=200)
        print('Saved figure to', args.save)
    else:
        plt.show()


if __name__ == '__main__':
    main()
