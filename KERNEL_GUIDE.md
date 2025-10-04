# Kernel Selection Guide for Kernel-EDMD

## Quick Start
Open `test_1_script_nd_sphere_kernel_edmd_full_semi_sphere.py` and modify line ~195:
```python
KERNEL_TYPE = 1  # Change this to 1, 2, 3, or 4
```

---

## Available Kernels

### Kernel 1: RBF/Gaussian Kernel ✅ (Default)
```python
KERNEL_TYPE = 1
```

**Formula**: `k(x,y) = exp(-||x-y||²/(2ε))`

**Bandwidth**: `ε = 0.5 · median(H) / ln(n+1)` (auto-computed)

**Pros**:
- Smooth, infinitely differentiable
- Universal approximator
- Well-studied and reliable

**Cons**:
- Sensitive to bandwidth choice
- Treats all distances in Euclidean space

**Best for**: General-purpose, first choice for most problems

---

### Kernel 2: Spherical/Geodesic Kernel 🌐 (Recommended for circle/sphere)
```python
KERNEL_TYPE = 2
```

**Formula**: `k(x,y) = exp(-d_geodesic(x,y)² / (2θ²))`
- Geodesic distance: `d = arccos(x·y^T)`

**Parameter to tune**:
```python
theta_scale = 0.5  # Try values: 0.1, 0.3, 0.5, 0.8, 1.0
```

**Pros**:
- Respects manifold geometry
- Natural for circle/sphere data
- Uses geodesic distance (shortest path on manifold)

**Cons**:
- Requires normalized inputs
- More expensive than RBF (arccos computation)

**Best for**: Circle, sphere, or other manifold data

**Tuning tips**:
- Small `theta_scale` (0.1-0.3): Local, captures fine details
- Large `theta_scale` (0.8-1.0): Global, smoother approximation

---

### Kernel 3: Matérn Kernel 📈 (Better generalization)
```python
KERNEL_TYPE = 3
```

**Formula (ν=1.5)**: `k(x,y) = (1 + √3·d/ℓ) · exp(-√3·d/ℓ)`

**Length scale**: `ℓ = √median(H)` (auto-computed)

**Smoothness parameter**: `ν = 1.5` (once differentiable)

**Pros**:
- Better generalization than RBF
- Controls function smoothness via ν
- Often outperforms RBF in practice

**Cons**:
- Slightly more expensive
- Less interpretable than RBF

**Best for**: When RBF overfits, or when you need controlled smoothness

**Variants**:
- `nu=1.5`: Once differentiable (default)
- `nu=2.5`: Twice differentiable (smoother)

---

### Kernel 4: Rational Quadratic Kernel 🔍 (Multi-scale)
```python
KERNEL_TYPE = 4
```

**Formula**: `k(x,y) = (1 + ||x-y||²/(2α·ℓ²))^(-α)`

**Parameter to tune**:
```python
alpha = 2.0  # Try values: 1.0, 2.0, 3.0, 5.0, 10.0
```

**Pros**:
- Captures multi-scale features
- Mixture of RBF kernels with different length scales
- Flexible for heterogeneous data

**Cons**:
- Extra parameter to tune (α)
- Can be less stable than RBF

**Best for**: Data with features at multiple scales

**Tuning tips**:
- Small `alpha` (1.0-2.0): More mixture components, captures more scales
- Large `alpha` (5.0+): Closer to RBF kernel

---

## Performance Comparison

| Kernel | Computation Speed | Memory | Generalization | Manifold-Aware |
|--------|------------------|--------|----------------|----------------|
| RBF (1) | ⚡⚡⚡ Fast | Low | Good | ❌ No |
| Spherical (2) | ⚡⚡ Medium | Low | Excellent | ✅ Yes |
| Matérn (3) | ⚡⚡ Medium | Low | Excellent | ❌ No |
| Rational Quad (4) | ⚡⚡ Medium | Low | Very Good | ❌ No |

---

## Recommended Workflow

### Step 1: Baseline
Start with **Kernel 1 (RBF)** as baseline
```python
KERNEL_TYPE = 1
```

### Step 2: Manifold-Specific
Try **Kernel 2 (Spherical)** since you have circle/sphere data
```python
KERNEL_TYPE = 2
theta_scale = 0.5  # Start here, then try 0.3 and 0.8
```

### Step 3: Better Generalization
If RBF overfits, try **Kernel 3 (Matérn)**
```python
KERNEL_TYPE = 3
```

### Step 4: Multi-Scale Features
For complex dynamics, try **Kernel 4 (Rational Quadratic)**
```python
KERNEL_TYPE = 4
alpha = 2.0  # Start here
```

---

## Parameter Tuning Guide

### Auto-Computed Parameters (No tuning needed)
- `epsilon`: RBF bandwidth
- `length_scale`: Matérn length scale

### Manual Tuning Required
- **Kernel 2**: `theta_scale` (line ~274)
  - Affects how "local" vs "global" the kernel is
  - Smaller = more local, larger = more global
  
- **Kernel 4**: `alpha` (line ~282)
  - Controls mixture richness
  - Smaller = more mixture components

### How to Tune
1. Run with default values
2. Check eigenvalue analysis output
3. If too many eigenvalues < 1e-6: increase scale parameter
4. If results too smooth: decrease scale parameter
5. Compare final particle distribution quality

---

## Debugging Tips

### Issue: Too many truncated eigenvalues (>300)
**Solution**: Increase bandwidth/scale
- Kernel 1: Multiply `epsilon` by 2
- Kernel 2: Increase `theta_scale` to 0.8-1.0
- Kernel 3: Multiply `length_scale` by 2
- Kernel 4: Increase `alpha` or `length_scale`

### Issue: Poor convergence
**Solution**: Try Spherical kernel (Kernel 2)
- Better for circle/sphere manifolds

### Issue: Particles don't spread evenly
**Solution**: 
1. Try Matérn kernel (Kernel 3)
2. Or decrease scale parameters (make kernel more local)

---

## Mathematical Details

### RBF Kernel Derivation
```
||x - y||² = ||x||² + ||y||² - 2·x·yᵀ
D²[i,j] = sq_x[i] + sq_y[j] - 2·(X @ Yᵀ)[i,j]
K[i,j] = exp(-D²[i,j] / (2ε))
```

### Spherical Kernel Derivation
```
cos(θ) = x·yᵀ / (||x||·||y||)  (assuming unit vectors)
d_geodesic = arccos(x·yᵀ)  (angle on unit circle)
K[i,j] = exp(-d²[i,j] / (2θ_scale²))
```

### Matérn Kernel (ν=1.5)
```
d = ||x - y||
K = (1 + √3·d/ℓ) · exp(-√3·d/ℓ)
```

### Rational Quadratic
```
K = (1 + ||x-y||² / (2α·ℓ²))^(-α)
As α → ∞: K → exp(-||x-y||²/(2ℓ²))  (RBF)
```

---

## Citation

If you use these kernels in your research:

**RBF**: Schölkopf & Smola, "Learning with Kernels" (2002)

**Spherical**: Jayasumana et al., "Kernel Methods on Riemannian Manifolds with Gaussian RBF Kernels" (2015)

**Matérn**: Rasmussen & Williams, "Gaussian Processes for Machine Learning" (2006)

**Rational Quadratic**: Rasmussen & Williams, "Gaussian Processes for Machine Learning" (2006)

---

## Example Usage

```python
# Try all kernels sequentially
for k in [1, 2, 3, 4]:
    # Modify KERNEL_TYPE in the script
    KERNEL_TYPE = k
    # Run and compare results
```

Compare:
- Eigenvalue distribution
- Final particle distribution
- Convergence quality
- Computation time
