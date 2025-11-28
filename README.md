# Curve Fit CPP

这是一个基于 OpenCV 与 Ceres Solver 的三维弹道轨迹重建项目。项目实现了从单目相机观测到的像素坐标恢复带有空气阻力的抛物线轨迹参数（初速度、阻力系数等）。

## 主要特性

- **物理模型**: 采用一阶空气阻力模型 ($F_{drag} = -k \cdot v$)。
- **鲁棒的背投影**: 使用射线-平面交点（Ray-Plane Intersection）代替单应性矩阵（Homography），解决了相机光心位于平面上或平移量为 0 时的奇异性问题。
- **时间同步**: 支持传入精确的时间戳进行拟合，从而能够准确恢复物理参数（如阻力系数 $k$）。
- **鲁棒的输入解析**: 内置简单的 YAML 解析器作为 OpenCV `FileStorage` 的后备方案，提高了兼容性。

## 依赖项

- CMake >= 3.10
- C++17 编译器
- OpenCV 4.x
- Ceres Solver
- Python 3 (用于数据生成和可视化)
  - numpy
  - matplotlib
  - opencv-python

## 构建说明

```bash
mkdir -p build
cd build
cmake ..
make -j
```

## 使用指南

### 1. 生成合成数据

使用 Python 脚本生成模拟的相机参数、三维轨迹点、像素坐标观测值和时间戳。

```bash
# 生成数据，保存在 data/ 目录下
python3 scripts/generate_data.py \
    --out-prefix data/test \
    --vx0 10.0 --vz0 10.0 --k 0.5 \
    --pixel-noise 2 --world-noise 0.05
```

这将生成：
- `data/test_camera.yml`: 相机内参和外参
- `data/test_pixels.txt`: 像素坐标观测值
- `data/test_times.txt`: 对应的时间戳
- `data/test_world.txt`: 真实世界坐标（用于验证）

### 2. 运行轨迹拟合 (C++)

运行编译好的 `curve_fit` 程序进行参数估计(--time 参数已弃用， 现在是否传入不影响结果)。

```bash
# 基本用法
./build/curve_fit data/test_camera.yml data/test_pixels.txt

# 可选参数: 指定起始点猜测 (x0, z0)
./build/curve_fit data/test_camera.yml data/test_pixels.txt --x0 0.4 --z0 0.0
```

**注意**: 强烈建议提供 `--times` 参数。如果不提供时间戳，程序将假设点之间是等时间间距的，这可能导致恢复出的物理参数（特别是阻力系数 $k$）与真实值存在比例偏差。

### 3. 结果可视化

使用可视化脚本对比拟合结果和真实数据。

```bash
python3 scripts/visualize.py \
    --fitted fitted_traj.txt \
    --world data/test_world.txt \
    --save result.png
```

## 算法细节

### 物理模型
假设物体受重力和与速度成正比的空气阻力影响：
$$
\begin{cases}
\ddot{x} = -k \dot{x} \\
\ddot{z} = -g - k \dot{z}
\end{cases}
$$
积分得到位置公式：
$$
\begin{aligned}
x(t) &= x_0 + \frac{v_{x0}}{k}(1 - e^{-kt}) \\
z(t) &= z_0 + \frac{v_{z0}}{k}(1 - e^{-kt}) - \frac{g}{k}t + \frac{g}{k^2}(1 - e^{-kt})
\end{aligned}
$$

### 优化过程
1. **背投影**: 利用相机参数 $K, R, t$ 将像素坐标 $(u, v)$ 反投影为空间射线，计算射线与平面 $Y=0$ 的交点，作为观测点的初始三维位置估计。
2. **非线性最小二乘**: 使用 Ceres Solver 优化参数 $\theta = [v_{x0}, v_{z0}, \log(k)]$，最小化重投影误差或三维空间误差（当前实现最小化三维空间中预测点与观测点的距离）。

## 输入文件格式

### camera.yml
OpenCV YAML 格式，必须包含 `K` (3x3), `R` (3x3), `t` (3x1)。可选 `dist` (畸变系数)。

```yaml
K: !!opencv-matrix
   rows: 3
   cols: 3
   data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
R: ...
t: ...
```

### pixels.txt
每行两个浮点数，表示 $u, v$ 坐标。

### times.txt
每行一个浮点数，表示对应的时间戳（秒）。
