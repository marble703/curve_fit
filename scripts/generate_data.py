#!/usr/bin/env python3
"""
生成合成轨迹与像素观测的脚本。

输出：
 - {out_prefix}_pixels.txt : 每行 `u v`（像素，带噪声）
 - {out_prefix}_world.txt  : 每行 `x y z`（三维真实坐标，可带噪声）
 - {out_prefix}_camera.yml : 相机参数（YAML，OpenCV 可读）

模型：一阶阻力抛体（文档中公式）。
"""
import argparse
import numpy as np
import cv2
import math
import os


def write_camera_yaml(path, K, dist, R, t):
    # Write in OpenCV yaml matrix format so C++ FileStorage can read.
    def mat_to_str(name, mat):
        arr = mat.flatten().tolist()
        s = f"{name}: !!opencv-matrix\n  rows: {mat.shape[0]}\n  cols: {mat.shape[1] if mat.ndim>1 else 1}\n  data: ["
        s += ", ".join([repr(float(x)) for x in arr])
        s += "]\n"
        return s

    s = ""
    s += mat_to_str('K', K)
    s += mat_to_str('dist', dist.reshape(1, -1))
    s += mat_to_str('R', R)
    s += mat_to_str('t', t.reshape(-1,1))

    with open(path, 'w') as f:
        f.write(s)


def gen_points(x0, z0, vx0, vz0, k, times, g=9.81):
    # returns Nx3 array of (x,y,z) with y=0
    pts = []
    for t in times:
        if k == 0:
            # no drag limit
            x = x0 + vx0 * t
            z = z0 + vz0 * t - 0.5 * g * t * t
        else:
            exp_mt = math.exp(-k * t)
            x = x0 + vx0 / k * (1.0 - exp_mt)
            z = z0 + vz0 / k * (1.0 - exp_mt) - (g / k) * t + (g / (k * k)) * (1.0 - exp_mt)
        pts.append([x, 0.0, z])
    return np.array(pts, dtype=float)


def project_points(pts_world, K, dist, R, tvec):
    # pts_world: Nx3
    # R: 3x3 rotation matrix (world->camera)
    # tvec: 3x1 translation (world->camera)
    # use cv2.projectPoints which expects rvec
    rvec, _ = cv2.Rodrigues(R)
    pts = pts_world.reshape(-1,1,3).astype(np.float64)
    imgpts, _ = cv2.projectPoints(pts, rvec, tvec.reshape(3,), K, dist)
    imgpts = imgpts.reshape(-1,2)
    return imgpts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-prefix', default='data', help='输出文件前缀')
    parser.add_argument('--N', type=int, default=30, help='数据点数量')
    parser.add_argument('--x0', type=float, default=0.4, help='起始点 x')
    parser.add_argument('--z0', type=float, default=0.0, help='起始点 z（高度）')
    parser.add_argument('--vx0', type=float, default=10.0, help='初速 x 分量 (m/s)')
    parser.add_argument('--vz0', type=float, default=10.0, help='初速 z 分量 (m/s, 正为向上)')
    parser.add_argument('--k', type=float, default=0.3, help='阻力系数 k=b/m')
    parser.add_argument('--tmax', type=float, default=2.0, help='最大时间 (s)')
    parser.add_argument('--pixel-noise', type=float, default=1.0, help='像素噪声标准差 (px)')
    parser.add_argument('--world-noise', type=float, default=0.05, help='三维点噪声 std (m)')
    parser.add_argument('--camera-yaml', default=None, help='可选：输出/使用的相机 yaml 路径')
    parser.add_argument('--save-plot', default=None, help='保存轨迹图路径（如 traj.png）')
    args = parser.parse_args()

    out_prefix = args.out_prefix
    N = args.N
    times = np.linspace(0.0, args.tmax, N)

    # generate world points
    pts = gen_points(args.x0, args.z0, args.vx0, args.vz0, args.k, times)

    # Add world noise
    if args.world_noise > 0.0:
        pts += np.random.normal(scale=args.world_noise, size=pts.shape)

    # Camera: placed at (0, -10, 2) looking at +Y direction (towards Y=0 plane)
    # R rotates world +Y to camera -Z (optical axis), world +Z to camera +Y
    # Simple: rotate 90 deg around X so that camera looks along +Y
    # R = Rx(90deg): [[1,0,0],[0,0,-1],[0,1,0]]
    # tvec = -R @ C_world = -R @ [0,-10,2]^T
    fx = 800.0; fy = 800.0; cx = 640.0; cy = 360.0
    K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]], dtype=float)
    dist = np.zeros((5,), dtype=float)
    # Rotation: camera looks along world +Y, up is world +Z
    R = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]], dtype=float)
    C_world = np.array([0, -10, 2], dtype=float)  # camera center in world
    tvec = -R @ C_world  # tvec = -R @ C_world

    # project
    imgpts = project_points(pts, K, dist, R, tvec.reshape(3,1))

    # add pixel noise
    if args.pixel_noise > 0.0:
        imgpts += np.random.normal(scale=args.pixel_noise, size=imgpts.shape)

    # save files
    os.makedirs(os.path.dirname(out_prefix) if os.path.dirname(out_prefix) else '.', exist_ok=True)
    pix_path = f"{out_prefix}_pixels.txt"
    world_path = f"{out_prefix}_world.txt"
    cam_path = f"{out_prefix}_camera.yml" if args.camera_yaml is None else args.camera_yaml
    times_path = f"{out_prefix}_times.txt"

    np.savetxt(pix_path, imgpts, fmt='%.6f')
    np.savetxt(world_path, pts, fmt='%.6f')
    np.savetxt(times_path, times, fmt='%.6f')
    write_camera_yaml(cam_path, K, dist, R, tvec.reshape(3,1))

    print(f"Wrote pixels -> {pix_path}")
    print(f"Wrote world  -> {world_path}")
    print(f"Wrote camera -> {cam_path}")
    print(f"Wrote times  -> {times_path}")

    # Plot and optionally save
    import matplotlib
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.markers import MarkerStyle
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(pts[:,0], pts[:,2], 'b.-', label='轨迹 (x-z)')
    ax.scatter([pts[0,0]], [pts[0,2]], c='g', s=80, marker=MarkerStyle('o'), label='起点')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title('合成抛物线轨迹 (Y=0 平面)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    plot_path = args.save_plot if args.save_plot else f"{out_prefix}_traj.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote trajectory plot -> {plot_path}")


if __name__ == '__main__':
    main()
