# SDE Interpretation Guide for X_tar_next Generation

**Date**: October 23, 2025  
**File**: `test_1_script_nd_sphere_kernel_edmd_full_semi_sphere.py`  
**Lines**: 224-242

---

## 概述

在生成 `X_tar_next` 时，代码实现了流形约束的 Langevin 动力学。本指南说明如何在 **Itô** 和 **Stratonovich** 解释之间切换。

---

## 当前代码（Line 224-242）

```python
# Step 2: Project to tangent space (manifold constraint)
X_norm = X_tar / (np.linalg.norm(X_tar, axis=1, keepdims=True) + 1e-12)
# Projection matrix: P = I - n⊗n^T (removes normal component)
proj = np.eye(X_tar.shape[1])[None, :, :] - X_norm[:, :, None] * X_norm[:, None, :]

# ============================================================
# SDE Interpretation: Choose ONE of the following options
# ============================================================
# 
# Option 1: Itô SDE (original, no correction needed)
# - Interpretation: dX = P·∇log(π)dt + √2·P·dW (Itô form)
# - Discretization: Euler-Maruyama (consistent with Itô calculus)
# - No geometric correction term needed
# score_tan = np.einsum('nij,ni->nj', proj, score_eucl)
#
# Option 2: Stratonovich SDE with Itô-Stratonovich correction
# - Interpretation: dX = P·∇log(π)dt + √2·P∘dW (Stratonovich form)
# - Since we use Euler-Maruyama (Itô discretization), need correction term
# - Correction: -(d-1)/2·n accounts for the difference between Itô and Stratonovich
# - For unit sphere S^(d-1): Stratonovich drift = Itô drift - (1/2)∇·(σσᵀ)
geometric_drift = -(d - 1) / 2 * X_norm  # Shape: (n, d), Itô-Stratonovich correction
score_tan = np.einsum('nij,ni->nj', proj, score_eucl) + dt_edmd * np.einsum('nij,ni->nj', proj, geometric_drift)
# ============================================================
```

---

## 如何切换

### **使用 Itô SDE（无修正项）**

注释掉包含 `geometric_drift` 的行：

```python
# Option 1: Itô SDE (active)
score_tan = np.einsum('nij,ni->nj', proj, score_eucl)

# Option 2: Stratonovich with correction (commented out)
# geometric_drift = -(d - 1) / 2 * X_norm
# score_tan = np.einsum('nij,ni->nj', proj, score_eucl) + dt_edmd * np.einsum('nij,ni->nj', proj, geometric_drift)
```

### **使用 Stratonovich SDE（有 Itô-Stratonovich 修正）**（当前默认）

保持当前代码：

```python
# Option 1: Itô SDE (commented out)
# score_tan = np.einsum('nij,ni->nj', proj, score_eucl)

# Option 2: Stratonovich with correction (active)
geometric_drift = -(d - 1) / 2 * X_norm
score_tan = np.einsum('nij,ni->nj', proj, score_eucl) + dt_edmd * np.einsum('nij,ni->nj', proj, geometric_drift)
```

---

## 理论背景

### Itô vs Stratonovich SDE

| 特性 | Itô | Stratonovich |
|------|-----|--------------|
| **标准形式** | $dX = b(X_t)dt + \sigma(X_t)dW_t$ | $dX = \tilde{b}(X_t)dt + \sigma(X_t) \circ dW_t$ |
| **在流形上** | 原始形式，无需修正 | 用 Euler 离散化需要修正项 |
| **离散化** | Euler-Maruyama（一阶） | 理想：Heun/Midpoint（二阶） |
| **坐标变换** | 需要 Itô 引理 | 协变（形式不变） ✅ |
| **几何意义** | 前向差分 | 中点规则 |

### Itô-Stratonovich 转换关系

**关键公式**：对于 Stratonovich SDE
$$
dX_t = b(X_t)dt + \sigma(X_t) \circ dW_t
$$

等价的 Itô SDE 是：
$$
dX_t = \left[b(X_t) + \frac{1}{2}\sum_{j=1}^m \sigma_j \frac{\partial \sigma_j}{\partial x}\right]dt + \sigma(X_t) dW_t
$$

其中额外的项 $\frac{1}{2}\sum_j \sigma_j \frac{\partial \sigma_j}{\partial x}$ 就是 **Itô-Stratonovich 修正项**。

### 流形上的修正项推导

对于单位球面 $\mathbb{S}^{d-1}$，扩散项是 $\sigma = \sqrt{2} \mathbf{P}_x$（投影矩阵）。

计算 Stratonovich 修正：
$$
\frac{1}{2}\nabla \cdot (\sigma \sigma^T) = \frac{1}{2}\nabla \cdot (2\mathbf{P}_x) = \nabla \cdot \mathbf{P}_x
$$

对于单位球面：
$$
\nabla \cdot \mathbf{P}_x = \nabla \cdot (I - \mathbf{n}\otimes\mathbf{n}^T) = -(d-1)\mathbf{n}(x)
$$

因此修正项为：
$$
\frac{1}{2}\nabla \cdot (\sigma \sigma^T) = -\frac{d-1}{2}\mathbf{n}(x)
$$

**结论**：
- **Itô SDE**：Drift = $\mathbf{P}_x \nabla \log \pi(x)$（无修正）
- **Stratonovich SDE**（用 Itô 离散化）：Drift = $\mathbf{P}_x \nabla \log \pi(x) - \frac{d-1}{2}\mathbf{n}(x)$（有修正）

### 为什么需要投影？

```python
# 错误的做法（会导致维度不匹配）
score_tan = np.einsum('nij,ni->nj', proj, score_eucl) + dt_edmd * proj @ geometric_drift  # ❌

# 正确的做法（使用 einsum 进行批量投影）
score_tan = np.einsum('nij,ni->nj', proj, score_eucl) + dt_edmd * np.einsum('nij,ni->nj', proj, geometric_drift)  # ✅
```

**原因**：
- `proj` 的形状是 `(n, d, d)` - 每个数据点有自己的投影矩阵
- `geometric_drift` 的形状是 `(n, d)` - 每个数据点有一个向量
- 需要对每个点分别应用投影：`proj[i] @ geometric_drift[i]`
- `einsum('nij,ni->nj', proj, geometric_drift)` 正确地执行了批量矩阵-向量乘法

---

## 实验对比

### 预期差异

对于 **2D 半圆**（$d=2$）：

| 方法 | 修正项 | 物理解释 |
|------|-------|---------|
| **Itô SDE** | 无（$\mathbf{0}$） | 前向 Euler，原始 Langevin 动力学 |
| **Stratonovich SDE** | $-\frac{1}{2}\mathbf{n}$ | 修正 Itô 离散化，更接近连续 Stratonovich |

**注意**：由于代码中有再归一化步骤（`X_tar_next = X_step / ||X_step||`），两种方法在实践中差异可能很小。

### 哪个更"正确"？

这取决于你想模拟的物理系统：

**使用 Itô**（无修正）：
- ✅ 如果你的系统本质上是 Itô 过程（如金融、离散观测）
- ✅ 标准 MCMC/Langevin 采样理论
- ✅ Fokker-Planck 方程直接对应

**使用 Stratonovich**（有修正）：
- ✅ 如果你的系统是物理/几何的（如布朗运动、热力学）
- ✅ 坐标变换不变性（在流形上很重要！）
- ✅ 更接近连续时间极限

### 验证方法

运行代码后，检查生成的 `X_tar_next`：

```python
# 验证是否在单位圆上
radii = np.linalg.norm(X_tar_next, axis=1)
print(f"Radii mean: {np.mean(radii):.6f}")
print(f"Radii std:  {np.std(radii):.6f}")

# 理想情况：mean ≈ 1.0, std ≈ 0.0
```

---

## 常见问题

### Q1: 当前使用哪个版本？

**A**: 当前使用 **Stratonovich SDE with Itô-Stratonovich correction**（带修正的 Stratonovich SDE）。

### Q2: 为什么用 KDE 估计已知的均匀分布？

**A**: 这是方法论设计：
- **目的**：验证算法在未知分布上的通用性
- **实际场景**：通常只有样本数据，没有解析密度
- **好处**：代码可以直接迁移到任意复杂分布

对于圆/半圆均匀分布，理论上 `score = 0`，但 KDE 估计会有有限样本误差。

### Q3: 两种方法哪个更好？

**A**: 
- **理论上**：Stratonovich 在流形上更自然（坐标不变性）
- **实践上**：对于小的 `dt_edmd`，两者差异很小
- **当前选择**：Stratonovich + 修正项，理论上更适合流形几何

**推荐**：
- 如果关注流形几何和坐标不变性 → **Stratonovich**（当前）
- 如果关注标准 MCMC 理论 → **Itô**（注释掉修正项）

### Q4: 修正项的符号和系数是否正确？

**A**: 是的！修正项为：

$$
-\frac{d-1}{2} \mathbf{n}(x)
$$

- **负号**：指向球心（向内）
- **系数 $(d-1)/2$**：来自 $\frac{1}{2}\nabla \cdot \mathbf{P}_x = -\frac{d-1}{2}\mathbf{n}$
- **对于 2D 圆**：修正项 = $-\frac{1}{2}\mathbf{n}$

---

## 参考文献

1. **Manifold Langevin Dynamics**: Girolami & Calderhead (2011), "Riemann manifold Langevin and Hamiltonian Monte Carlo methods"
2. **Itô vs Stratonovich**: Øksendal (2003), "Stochastic Differential Equations"
3. **Geometric correction**: Hsu (2002), "Stochastic Analysis on Manifolds"

---

## 修改历史

- **2025-10-23**: 初始版本，修正维度匹配错误
- **2025-10-23**: 添加详细注释和切换指南
