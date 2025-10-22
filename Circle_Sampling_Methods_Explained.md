# 圆形采样方法详解 (Circle Sampling Methods)

**文档创建时间**: 2025年10月22日  
**适用代码**: `test_1_script_nd_sphere_dmps_full_semi_sphere.py`

---

## 📊 **问题背景**

在圆形或球面上均匀采样点时，有两种主要方法：
1. **极坐标/角度法** (Polar/Angular Method)
2. **高斯归一化法** (Gaussian Normalization / Marsaglia Method)

---

## 🎯 **方法 1: 极坐标法（显式角度）**

### **2D 圆形采样**

```python
theta = np.random.uniform(0, 2 * np.pi, n)
x = np.cos(theta)
y = np.sin(theta)
```

#### **数学原理**
圆的参数化表示：
$$
\mathbf{x}(\theta) = \begin{bmatrix} \cos\theta \\ \sin\theta \end{bmatrix}, \quad \theta \in [0, 2\pi)
$$

由于圆的弧长参数化，均匀采样 $\theta$ 等价于在圆周上均匀分布点。

#### **优点**
✅ **直观**：明确使用角度 $\theta$  
✅ **高效**：仅需一次三角函数计算  
✅ **精确控制**：可以轻松限制角度范围（如半圆 $[0, \pi]$）  
✅ **易于理解**：符合几何直觉

#### **缺点**
❌ **维度限制**：不易推广到高维球面（需要多个角度）  
❌ **非各向同性**：各向异性变换后需要重新归一化

---

## 🔬 **方法 2: 高斯归一化法（Marsaglia 方法）**

### **实现**

```python
u = np.random.normal(0, 1, (n, d))  # 高斯采样
u_trans = u / np.linalg.norm(u, axis=1, keepdims=True)  # 归一化
```

### **数学原理**

**定理（Marsaglia, 1972）**：设 $\mathbf{X} = (X_1, \ldots, X_d)$ 其中 $X_i \sim \mathcal{N}(0, 1)$ 独立同分布，定义：
$$
\mathbf{U} = \frac{\mathbf{X}}{\|\mathbf{X}\|}
$$

那么 $\mathbf{U}$ 在 $d$ 维单位球面 $\mathbb{S}^{d-1}$ 上**均匀分布**。

#### **证明核心思想**

1. **旋转不变性**  
   多元正态分布 $\mathcal{N}(0, I_d)$ 对任意正交变换 $O \in SO(d)$ 保持不变：
   $$
   O\mathbf{X} \sim \mathcal{N}(0, I_d) \quad \text{当} \quad \mathbf{X} \sim \mathcal{N}(0, I_d)
   $$

2. **极坐标分解**  
   在极坐标下：
   $$
   \mathbf{X} = R \cdot \mathbf{U}
   $$
   其中 $R = \|\mathbf{X}\|$ (径向距离) 和 $\mathbf{U}$ (方向) **独立**。

3. **均匀性**  
   由于旋转不变性，方向 $\mathbf{U}$ 必然在球面上均匀分布。

#### **为什么没有角度？**

角度 **隐式存在** 于高斯分布的各向同性性质中！

在 2D 情况下：
$$
\begin{aligned}
X &\sim \mathcal{N}(0, 1), \quad Y \sim \mathcal{N}(0, 1) \\
\Theta &= \arctan(Y/X) \sim \text{Uniform}(0, 2\pi)
\end{aligned}
$$

这是隐式生成的角度，无需显式计算。

#### **优点**
✅ **高维推广**：自然适用于任意维度 $d$（如 3D 球面、4D 超球等）  
✅ **理论优雅**：基于深刻的概率理论  
✅ **数值稳定**：高斯采样的数值性质良好  
✅ **各向同性**：自然处理旋转对称性

#### **缺点**
❌ **不直观**：隐藏了角度信息  
❌ **难以限制范围**：生成半圆需要拒绝采样  
❌ **计算稍慢**：需要生成高斯随机数 + 归一化

---

## 🔄 **各向异性变换 (Anisotropy)**

在你的代码中有 `lambda_` 参数：

```python
u[:, 0] = lambda_ * u[:, 0]  # 沿 x 轴拉伸
u_trans = u / np.linalg.norm(u, axis=1, keepdims=True)  # 重新归一化
```

### **效果**
将圆形拉伸成椭圆，然后重新投影回圆形，导致：
- x 方向点密度降低
- y 方向点密度增加

### **在两种方法中的应用**

**极坐标法**：
```python
theta = np.random.uniform(0, 2*np.pi, n)
u_trans = np.hstack([lambda_ * np.cos(theta), np.sin(theta)])
u_trans = u_trans / np.linalg.norm(u_trans, axis=1, keepdims=True)
```

**高斯法**（原代码）：
```python
u = np.random.normal(0, 1, (n, d))
u[:, 0] = lambda_ * u[:, 0]
u_trans = u / np.linalg.norm(u, axis=1, keepdims=True)
```

两者产生**相同的分布**！

---

## 📊 **半圆采样的对比**

### **极坐标法（高效）**
```python
theta = np.random.uniform(0, np.pi, n)  # 只需限制角度范围
u_trans = np.hstack([np.cos(theta), np.sin(theta)])
```
- ✅ 100% 采样效率
- ✅ 无需拒绝采样

### **拒绝采样法（低效）**
```python
u = np.random.normal(0, 1, (n, d))
u_trans = u / np.linalg.norm(u, axis=1, keepdims=True)
valid = u_trans[:, 1] >= 0  # 只保留上半部分
u_trans = u_trans[valid, :]
```
- ❌ ~50% 采样效率（浪费一半样本）
- ✅ 保持与高斯法的统计一致性

---

## 🎓 **高维推广**

### **3D 球面 ($\mathbb{S}^2$)**

**极坐标法（球坐标）**：
```python
# 需要两个角度：θ (方位角) 和 φ (极角)
theta = np.random.uniform(0, 2*np.pi, n)
phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)  # 特殊处理！

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
```

**注意**：$\phi$ 不能均匀采样！必须用 $\cos\phi \sim \text{Uniform}(-1, 1)$ 以保证球面均匀性。

**高斯法**：
```python
u = np.random.normal(0, 1, (n, 3))  # 简单！
u_trans = u / np.linalg.norm(u, axis=1, keepdims=True)
```

### **结论**
- **2D**: 极坐标法更简单
- **3D+**: 高斯法更简单、更不容易出错

---

## 📈 **代码修改建议**

### **当前修改（已实施）**

| 场景 | 方法选择 | 理由 |
|------|---------|------|
| **Full Circle** | 极坐标法 | 显式角度，直观清晰 |
| **Semi-Circle** | 拒绝采样法 | 保持与 Full Circle 统计一致性 |

### **可选修改**

如果追求最高效率，可以将 Semi-Circle 也改为极坐标：

```python
if USE_SEMICIRCLE:
    theta = np.random.uniform(0, np.pi, n)  # 半圆：[0, π]
    u_trans = np.hstack([np.cos(theta), np.sin(theta)])
    # ... 后续处理相同
```

**权衡**：
- ✅ 100% 采样效率
- ❌ 与 Full Circle 采样方式不完全一致（可能影响算法比较）

---

## 🔍 **验证代码**

```python
import numpy as np
import matplotlib.pyplot as plt

n = 1000

# 方法 1: 极坐标
theta1 = np.random.uniform(0, 2*np.pi, n)
x1 = np.cos(theta1)
y1 = np.sin(theta1)

# 方法 2: 高斯归一化
u2 = np.random.normal(0, 1, (n, 2))
u2_norm = u2 / np.linalg.norm(u2, axis=1, keepdims=True)
x2, y2 = u2_norm[:, 0], u2_norm[:, 1]

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(x1, y1, s=1, alpha=0.5)
axes[0].set_title('极坐标法')
axes[0].axis('equal')
axes[1].scatter(x2, y2, s=1, alpha=0.5)
axes[1].set_title('高斯归一化法')
axes[1].axis('equal')
plt.show()

# 统计检验：角度分布应该均匀
angles1 = np.arctan2(y1, x1)
angles2 = np.arctan2(y2, x2)
print(f"方法1 角度均值: {np.mean(angles1):.4f} (期望: 0)")
print(f"方法2 角度均值: {np.mean(angles2):.4f} (期望: 0)")
```

---

## 📚 **参考文献**

1. **Marsaglia, G. (1972)**. "Choosing a Point from the Surface of a Sphere." *Annals of Mathematical Statistics*, 43(2), 645-646.

2. **Muller, M. E. (1959)**. "A note on a method for generating points uniformly on n-dimensional spheres." *Communications of the ACM*, 2(4), 19-20.

3. **Devroye, L. (1986)**. *Non-Uniform Random Variate Generation*. Springer-Verlag, Chapter 9.

---

## ✅ **总结**

| 维度 | 推荐方法 | 原因 |
|------|---------|------|
| **2D 完整圆** | 极坐标法 | 直观、高效、显式角度 |
| **2D 半圆** | 极坐标法 (效率优先) <br> 拒绝采样 (一致性优先) | 根据需求选择 |
| **3D+ 球面** | 高斯归一化法 | 简单、不易出错 |

**修改后的代码现在使用极坐标法作为主方法，并保留高斯法作为注释，方便对比和切换！**
