**已知相机参数时的三维轨迹重建：计算流程总结**

---

### **输入**
- 像素坐标序列：$\{\mathbf u_i=(u_i,v_i)\}_{i=1}^N$（已去畸变，或已知畸变系数）
- 相机内参矩阵：$K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}$
- 相机外参：旋转矩阵 $R$、平移向量 $\mathbf t$
- 起始点三维坐标：$P_0=(x_0,0,z_0)$（位于 $Y=0$ 竖直平面）

---

### **输出**
- 一阶阻力运动参数：阻力系数 $k=b/m$，初始速度 $\mathbf v_0=(\dot x_0,\dot z_0)$
- 或等价参数：$\alpha=\dot x_0/k$，$\beta=\dot z_0/k$，$\sigma=g/k^2$

---

### **计算流程（6 步）**

---

#### **Step 1: 图像去畸变（如需要）**
若像素坐标未经校正，先进行径向畸变修正：
$$
\begin{aligned}
u' &= u + (u-c_x)(k_1r^2+k_2r^4) + 2p_1uv + p_2(r^2+2u^2),\\
v' &= v + (v-c_y)(k_1r^2+k_2r^4) + p_1(r^2+2v^2) + 2p_2uv,
\end{aligned}
$$
其中 $r^2=(u-c_x)^2+(v-c_y)^2$。后续所有计算使用校正后的 $\mathbf u'_i$。

---

#### **Step 2: 构造平面单应矩阵 $H$**
提取外参矩阵的前两列 $\mathbf r_1,\mathbf r_2$（对应 $X,Z$ 轴）：
$$
H = K\,[\mathbf r_1\; \mathbf r_2\; \mathbf t] \quad\in\mathbb R^{3\times3}.
$$
**验证**：检查 $\det(H)\neq0$，确保相机光轴不与平面平行。

---

#### **Step 3: 反投影像素点到三维平面 $Y=0$**
对每个像素 $\mathbf u_i=(u_i,v_i,1)^T$，计算其在平面上的交点坐标：
$$
\lambda_i\begin{bmatrix}x_i\\z_i\\1\end{bmatrix}
= H^{-1}\mathbf u_i,\qquad \lambda_i>0.
$$
**数值实现**：
```python
H_inv = np.linalg.inv(H)
points_2d_hom = H_inv @ pixels_hom  # shape (3, N)
points_2d = points_2d_hom[:2] / points_2d_hom[2:3]  # (x_i, z_i)
```

---

#### **Step 4: 建立运动方程参数化模型**
引入无量纲参数 $\tau\in[0,1)$，使得：
$$
\begin{aligned}
x(\tau) &= x_0 + \alpha\,\tau,\\[4pt]
z(\tau) &= z_0 + \beta\,\tau + \sigma\bigl[\tau+\ln(1-\tau)\bigr],
\end{aligned}
$$
其中未知参数为 $\theta=(\alpha,\beta,\sigma)$。

**几何意义**：
- $\alpha$ 控制水平尺度，
- $\beta$ 控制竖直初速尺度，
- $\sigma = g/k^2$ 综合阻力与重力。

---

#### **Step 5: 非线性最小二乘拟合**
**目标**：找到 $\theta^*$ 最小化数据点到模型的误差：
$$
\theta^* = \arg\min_{\theta}\sum_{i=1}^N\Bigl\|
\mathbf p_i - \mathbf f(\tau_i;\theta)
\Bigr\|^2,
$$
其中 $\mathbf p_i=(x_i,z_i)$ 来自 Step 3，$\mathbf f(\tau;\theta)=(x(\tau),z(\tau))$。

**求解策略**：
- **未知 $\tau_i$**：将每点的 $\tau_i$ 也作为优化变量，加入单调约束 $\tau_{i+1}>\tau_i$。
- **已知时间戳**：若像素序列对应等时间间隔，可约束 $\tau_i=1-e^{-k t_i}$，减少自由度。

**实现**：
```python
# 使用 Ceres, SciPy 或 g2o
def residual(params, x_data, z_data):
    alpha, beta, sigma = params[:3]
    taus = params[3:]
    x_model = x0 + alpha * taus
    z_model = z0 + beta * taus + sigma * (taus + np.log(1 - taus))
    return np.concatenate([x_data - x_model, z_data - z_model])

# 初始值：忽略阻力，用抛物线拟合得到 alpha0, beta0, sigma0=0
result = least_squares(residual, x0=initial_params, args=(x_data, z_data))
```

---

#### **Step 6: 还原物理参数**
从优化结果 $\theta^*=(\alpha^*,\beta^*,\sigma^*)$ 计算原始物理量：
$$
\boxed{
\begin{aligned}
k &= \sqrt{g/\sigma^*},\\[6pt]
\dot x_0 &= \alpha^*\,k,\\[6pt]
\dot z_0 &= \beta^*\,k.
\end{aligned}}
$$
**轨迹生成**：对任意时间 $t\ge0$：
$$
\begin{aligned}
x(t) &= x_0 + \frac{\dot x_0}{k}\bigl(1-e^{-kt}\bigr),\\
z(t) &= z_0 + \frac{\dot z_0}{k}\bigl(1-e^{-kt}\bigr)-\frac{g}{k}t+\frac{g}{k^2}\bigl(1-e^{-kt}\bigr).
\end{aligned}
$$

---

### **关键公式速查**

| 步骤 | 公式 | 说明 |
|------|------|------|
| 单应矩阵 | $H=K[\mathbf r_1\;\mathbf r_2\;\mathbf t]$ | 平面 $Y=0$ 到像素的投影 |
| 反投影 | $(x_i,z_i,1)^T\propto H^{-1}(u_i,v_i,1)^T$ | 直接求得平面坐标 |
| 参数化模型 | $x=x_0+\alpha\tau$, $z=z_0+\beta\tau+\sigma\bigl[\tau+\ln(1-\tau)\bigr]$ | 消去时间变量 |
| 物理参数 | $k=\sqrt{g/\sigma}$, $\dot x_0=\alpha k$, $\dot z_0=\beta k$ | 最终输出 |

---

### **实现提示**

- **点数要求**：至少 6–8 个像素点，覆盖上升、顶点、下降段，避免退化。
- **初值选择**：
  - $\alpha_0,\beta_0$ 可通过忽略阻力（$\sigma=0$）的抛物线拟合得到；
  - $\sigma_0=0$ 或根据经验设为 $g/(5^2)\approx0.4$（假设 $k\approx5\,\text{s}^{-1}$）。
- **数值稳定**：对 $\ln(1-\tau)$ 使用 $\tau\le0.99$ 的截断，防止溢出。
- **鲁棒性**：在最小二乘中加入 Huber 损失函数，抵御像素提取噪声。

完成上述 6 步，即可从像素序列直接得到一阶阻力抛体运动方程的完整参数，进而生成任意时刻的三维空间坐标。