"""
可视化各向异性变换的效果
演示为什么在极坐标采样中，各向异性变换会改变点的分布
"""
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
n = 500
lambda_ = 2  # 各向异性参数（拉伸倍数）

# 方法1: 均匀角度采样 + 各向异性变换
theta = np.random.uniform(0, 2 * np.pi, n)
x1 = np.cos(theta)
y1 = np.sin(theta)

# 应用各向异性变换并重新归一化
x2 = lambda_ * x1  # 拉伸 x 坐标
y2 = y1
norm = np.sqrt(x2**2 + y2**2)
x3 = x2 / norm  # 重新归一化
y3 = y2 / norm

# 方法2: 不应用各向异性（对比）
x_uniform = np.cos(theta)
y_uniform = np.sin(theta)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 原始均匀分布
axes[0].scatter(x1, y1, s=10, alpha=0.5, c='blue')
axes[0].set_aspect('equal')
axes[0].set_title('步骤1: 均匀角度采样\n圆上均匀分布', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# 图2: 拉伸后的椭圆（未归一化）
axes[1].scatter(x2, y2, s=10, alpha=0.5, c='orange')
axes[1].set_aspect('equal')
axes[1].set_title(f'步骤2: 沿x轴拉伸 (λ={lambda_})\n变成椭圆', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

# 图3: 重新归一化后（非均匀分布）
axes[2].scatter(x3, y3, s=10, alpha=0.5, c='red')
axes[2].set_aspect('equal')
axes[2].set_title('步骤3: 重新投影回单位圆\n⚠️ 不再均匀！', fontsize=14)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')

plt.tight_layout()
plt.savefig('anisotropy_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# 分析角度分布
angles_before = theta
angles_after = np.arctan2(y3, x3)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 原始角度分布（均匀）
axes[0].hist(angles_before, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title('原始角度分布 (均匀)', fontsize=14)
axes[0].set_xlabel('角度 (弧度)')
axes[0].set_ylabel('频数')
axes[0].axhline(y=n/50, color='red', linestyle='--', label='期望均匀值')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 变换后角度分布（非均匀）
axes[1].hist(angles_after, bins=50, alpha=0.7, color='red', edgecolor='black')
axes[1].set_title('变换后角度分布 (非均匀)', fontsize=14)
axes[1].set_xlabel('角度 (弧度)')
axes[1].set_ylabel('频数')
axes[1].axhline(y=n/50, color='blue', linestyle='--', label='期望均匀值')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('angle_distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 70)
print("各向异性变换的效果分析")
print("=" * 70)
print(f"参数设置: λ = {lambda_}")
print(f"样本数量: n = {n}")
print()
print("变换步骤:")
print("  1. 均匀采样角度 θ ∈ [0, 2π]")
print("  2. 生成单位圆上的点 (cos θ, sin θ)")
print(f"  3. 沿 x 轴拉伸 λ 倍: ({lambda_} cos θ, sin θ)")
print("  4. 重新归一化: (λ cos θ, sin θ) / ||(λ cos θ, sin θ)||")
print()
print("结果:")
print("  - 原本均匀的角度分布变得 **非均匀**")
print("  - 靠近 x 轴（θ ≈ 0, π）的点密度 **降低**")
print("  - 靠近 y 轴（θ ≈ π/2, 3π/2）的点密度 **增加**")
print()
print("数学解释:")
print("  新角度 θ' = arctan(y/x) = arctan(sin θ / (λ cos θ))")
print("  当 λ > 1 时，θ' 的分布被 '压缩' 到 y 轴附近")
print("=" * 70)

# 计算密度比
x_region = np.abs(x3) > 0.95  # 靠近 x 轴
y_region = np.abs(y3) > 0.95  # 靠近 y 轴
print(f"\n密度分析 (λ = {lambda_}):")
print(f"  靠近 x 轴的点数: {np.sum(x_region)} ({100*np.sum(x_region)/n:.1f}%)")
print(f"  靠近 y 轴的点数: {np.sum(y_region)} ({100*np.sum(y_region)/n:.1f}%)")
print(f"  密度比 (y/x): {np.sum(y_region) / max(1, np.sum(x_region)):.2f}")
