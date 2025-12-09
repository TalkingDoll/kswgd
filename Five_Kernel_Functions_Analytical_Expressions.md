# Analytical Expressions of Five Kernel Functions

## Basic Notation

- **Input points**: $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$
- **Euclidean distance**: $d_{\text{Euclidean}}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\| = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$
- **Squared Euclidean distance**: $D^2(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^T\mathbf{y}$
- **Inner product**: $\langle\mathbf{x}, \mathbf{y}\rangle = \mathbf{x}^T\mathbf{y} = \sum_{i=1}^d x_i y_i$

---

## 1ï¸âƒ£ Kernel Type 1: RBF/Gaussian Kernel

### Analytical Expression

$$
k_{\text{RBF}}(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2\varepsilon}\right)
$$

### Expanded Form

$$
k_{\text{RBF}}(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^T\mathbf{y}}{2\varepsilon}\right)
$$

### Parameter Description

**Bandwidth parameter $\varepsilon > 0$**:
- Controls the "width" or "range of influence" of the kernel
- Smaller $\varepsilon$: more local (only considers nearby neighbors)
- Larger $\varepsilon$: more global (considers distant points)
- **Computation in code**:
  ```python
  H = sq_tar[:, None] + sq_tar[None, :] - 2 * (X_tar @ X_tar.T)  # Pairwise squared distance matrix
  epsilon = 0.5 * np.median(H) / (np.log(n + 1) + 1e-12)
  ```
  where $H_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|^2$

### Properties

1. **Range**: $k(\mathbf{x}, \mathbf{y}) \in (0, 1]$
2. **Symmetry**: $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{y}, \mathbf{x})$
3. **Self-similarity**: $k(\mathbf{x}, \mathbf{x}) = 1$
4. **Smoothness**: Infinitely differentiable ($C^\infty$ class)
5. **Locality**: When $\|\mathbf{x} - \mathbf{y}\| \to \infty$, $k \to 0$ with exponential decay
6. **Positive definiteness**: Corresponding Gram matrix is positive definite (guarantees unique solution)

### Gradient

$$
\nabla_{\mathbf{x}} k_{\text{RBF}}(\mathbf{x}, \mathbf{y}) = -\frac{1}{\varepsilon}(\mathbf{x} - \mathbf{y}) \cdot k_{\text{RBF}}(\mathbf{x}, \mathbf{y})
$$

### Advantages âœ…

- Smooth, infinitely differentiable
- Universal approximator
- Computationally efficient
- Solid theoretical foundation

### Disadvantages âš ï¸

- Very sensitive to bandwidth parameter $\varepsilon$
- Uses Euclidean distance, **does not respect manifold geometry**
- May introduce symmetry bias for circular/spherical data

### Use Cases

- General regression/classification problems in Euclidean space
- Standard Gaussian processes
- Non-manifold data

---

## 2ï¸âƒ£ Kernel Type 2: Spherical/Geodesic Kernel

### Analytical Expression

$$
k_{\text{Spherical}}(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{d_{\text{geodesic}}^2(\mathbf{x}, \mathbf{y})}{2\theta^2}\right)
$$

where **Geodesic Distance on Sphere**:

$$
d_{\text{geodesic}}(\mathbf{x}, \mathbf{y}) = \arccos\left(\frac{\mathbf{x}^T\mathbf{y}}{\|\mathbf{x}\| \cdot \|\mathbf{y}\|}\right)
$$

### Complete Form

$$
k_{\text{Spherical}}(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{1}{2\theta^2} \arccos^2\left(\frac{\langle\mathbf{x}, \mathbf{y}\rangle}{\|\mathbf{x}\| \cdot \|\mathbf{y}\|}\right)\right)
$$

### Unit Sphere Form

For **normalized vectors on unit sphere** $\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}$, $\hat{\mathbf{y}} = \frac{\mathbf{y}}{\|\mathbf{y}\|}$:

$$
k_{\text{Spherical}}(\hat{\mathbf{x}}, \hat{\mathbf{y}}) = \exp\left(-\frac{\arccos^2(\hat{\mathbf{x}}^T\hat{\mathbf{y}})}{2\theta^2}\right)
$$

where $\hat{\mathbf{x}}^T\hat{\mathbf{y}} = \cos(\alpha)$, $\alpha$ is the angle between two vectors.

### Geometric Meaning of Geodesic Distance

For unit circle or unit sphere:
- **Geodesic distance = shortest path length along the surface between two points**
- On unit circle/sphere, geodesic distance is the **angle (in radians)**
- Value range: $d_{\text{geodesic}} \in [0, \pi]$

**2D Unit Circle Example**:
```
Point A = (1, 0), Point B = (0, 1)
- Euclidean distance (straight line): âˆš2 â‰ˆ 1.414
- Geodesic distance (circular arc): arccos(0) = Ï€/2 â‰ˆ 1.571
```

**3D Unit Sphere Example**:
```
North Pole N = (0, 0, 1), Equator Point E = (1, 0, 0)
- Euclidean distance: âˆš2 â‰ˆ 1.414
- Geodesic distance: Ï€/2 â‰ˆ 1.571 (1/4 circle along meridian)
```

### Parameter Description

**Scale parameter $\theta > 0$**:
- Controls the "width" of the kernel
- Smaller $\theta$: more local (only considers points with similar angles)
- Larger $\theta$: more global (considers points with large angular separation)
- **Setting in code**:
  ```python
  theta_scale = 0.3  # Adjustable range [0.1, 1.0]
  ```

### Properties

1. **Range**: $k(\mathbf{x}, \mathbf{y}) \in (0, 1]$
2. **Symmetry**: $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{y}, \mathbf{x})$
3. **Self-similarity**: $k(\mathbf{x}, \mathbf{x}) = 1$ (because $\arccos(1) = 0$)
4. **Rotation invariance**: Depends only on angle, independent of coordinate system rotation
5. **Manifold awareness**: Respects spherical geometry, distance measured along surface
6. **Boundary sensitivity**: Naturally sensitive to boundaries of semi-circles/semi-spheres

### Special Values ($\theta = 0.3$)

| Angle | Geodesic Distance | Kernel Value | Similarity |
|-------|------------------|--------------|-----------|
| 0Â° (parallel) | 0 | 1.000 | Complete similarity |
| 30Â° | 0.524 rad | 0.400 | High similarity |
| 45Â° | 0.785 rad | 0.070 | Medium similarity |
| 90Â° (orthogonal) | 1.571 rad | $1.8 \times 10^{-8}$ | Almost dissimilar |
| 180Â° (opposite) | 3.142 rad | $\approx 0$ | Complete dissimilarity |

### Advantages âœ…

- **Respects manifold geometric structure** (Most important!)
- Naturally senses boundary constraints (for semi-circle/semi-sphere problems)
- Rotation invariance
- Suitable for circular, spherical, and other manifold data
- Infinitely differentiable

### Disadvantages âš ï¸

- Requires normalization of inputs to unit sphere
- arccos computation slightly slower than simple Euclidean distance
- Only applicable to manifold data

### Use Cases â­

- **Circular/spherical data**
- **Semi-circle/semi-sphere data (with boundary constraints)**
- Directional data (e.g., wind direction, line of sight)
- Earth surface distance calculations
- **This project: Kernel EDMD on semi-circle**

---

## 3ï¸âƒ£ Kernel Type 3: MatÃ©rn Kernel

### Analytical Expression ($\nu = 1.5$, once differentiable)

$$
k_{\text{MatÃ©rn}}(\mathbf{x}, \mathbf{y}) = \left(1 + \frac{\sqrt{3} \cdot d(\mathbf{x}, \mathbf{y})}{\ell}\right) \exp\left(-\frac{\sqrt{3} \cdot d(\mathbf{x}, \mathbf{y})}{\ell}\right)
$$

where $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$ is the Euclidean distance.

### Simplified Notation

Let $r = \frac{\sqrt{3} \cdot d}{\ell}$, then:

$$
k_{\text{MatÃ©rn}}(\mathbf{x}, \mathbf{y}) = (1 + r) e^{-r}
$$

### Other Smoothness Versions

**$\nu = 0.5$ (non-differentiable, exponential kernel)**:
$$
k(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{d}{\ell}\right)
$$

**$\nu = 2.5$ (twice differentiable)**:
$$
k(\mathbf{x}, \mathbf{y}) = \left(1 + \frac{\sqrt{5} \cdot d}{\ell} + \frac{5d^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5} \cdot d}{\ell}\right)
$$

**General Form** (arbitrary $\nu > 0$):
$$
k_{\text{MatÃ©rn}}(\mathbf{x}, \mathbf{y}) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} \cdot d}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} \cdot d}{\ell}\right)
$$

where:
- $\Gamma(\cdot)$ is the Gamma function
- $K_\nu(\cdot)$ is the modified Bessel function of the second kind

### Parameter Description

**Length scale $\ell > 0$**:
- Controls the decay rate of the kernel
- **Computation in code**:
  ```python
  length_scale = np.sqrt(np.median(H))
  ```

**Smoothness parameter $\nu > 0$**:
- Controls the differentiability of the function
- $\nu = 0.5$: non-differentiable (rough)
- $\nu = 1.5$: once differentiable (medium smooth)
- $\nu = 2.5$: twice differentiable (smoother)
- $\nu \to \infty$: converges to RBF kernel (infinitely smooth)
- **Default in code**: $\nu = 1.5$

### Properties

1. **Range**: $k(\mathbf{x}, \mathbf{y}) \in (0, 1]$
2. **Symmetry**: $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{y}, \mathbf{x})$
3. **Self-similarity**: $k(\mathbf{x}, \mathbf{x}) = 1$
4. **Differentiability**: $\lceil \nu \rceil - 1$ times differentiable ($\lceil \cdot \rceil$ is ceiling function)
5. **Relationship with RBF**: Converges to RBF kernel as $\nu \to \infty$

### Relationship with RBF

$$
\lim_{\nu \to \infty} k_{\text{MatÃ©rn}}(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{d^2}{2\ell^2}\right) = k_{\text{RBF}}(\mathbf{x}, \mathbf{y})
$$

### Advantages âœ…

- **Controllable function smoothness** (via $\nu$)
- Better generalization ability than RBF
- Better matches characteristics of many physical processes
- More robust on noisy data

### Disadvantages âš ï¸

- Slightly more complex computation than RBF
- Requires choosing appropriate $\nu$ parameter
- Still uses Euclidean distance (not suitable for manifold data)

### Use Cases

- Regression problems requiring smoothness control
- Modeling noisy time series
- Physical system modeling (temperature, pressure, etc.)
- Problems with prior knowledge about smoothness

---

## 4ï¸âƒ£ Kernel Type 4: Rational Quadratic Kernel

### Analytical Expression

$$
k_{\text{RQ}}(\mathbf{x}, \mathbf{y}) = \left(1 + \frac{\|\mathbf{x} - \mathbf{y}\|^2}{2\alpha\ell^2}\right)^{-\alpha}
$$

### Expanded Form

$$
k_{\text{RQ}}(\mathbf{x}, \mathbf{y}) = \left(1 + \frac{D^2(\mathbf{x}, \mathbf{y})}{2\alpha\ell^2}\right)^{-\alpha}
$$

where:
$$
D^2(\mathbf{x}, \mathbf{y}) = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^T\mathbf{y}
$$

### Parameter Description

**Scale mixture parameter $\alpha > 0$**:
- Controls the mixture ratio of large and small scales
- Smaller $\alpha$: more focus on large-scale variations
- Larger $\alpha$: closer to RBF kernel
- **Setting in code**: $\alpha = 2.0$ (adjustable range [1.0, 5.0])

**Length scale $\ell > 0$**:
- Controls the overall scale of the kernel
- **Computation in code**:
  ```python
  length_scale = np.sqrt(np.median(H))
  ```

### Properties

1. **Range**: $k(\mathbf{x}, \mathbf{y}) \in (0, 1]$
2. **Symmetry**: $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{y}, \mathbf{x})$
3. **Self-similarity**: $k(\mathbf{x}, \mathbf{x}) = 1$
4. **Smoothness**: Infinitely differentiable ($C^\infty$)
5. **Scale mixture**: Equivalent to weighted sum of infinitely many RBF kernels with different scales

### Scale Mixture Interpretation ðŸ”‘

Rational Quadratic kernel can be represented as a **mixture of infinitely many RBF kernels with different bandwidths**:

$$
k_{\text{RQ}}(\mathbf{x}, \mathbf{y}) = \int_0^\infty k_{\text{RBF}}(\mathbf{x}, \mathbf{y}; \tau) \cdot p(\tau \mid \alpha) \, d\tau
$$

where $p(\tau \mid \alpha)$ is a Gamma distribution.

**Physical Meaning**:
- Automatically captures features at multiple scales
- Can see both local details and global structure
- No need to manually select and combine multiple kernels

### Relationship with RBF

As $\alpha \to \infty$:

$$
\lim_{\alpha \to \infty} k_{\text{RQ}}(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{d^2}{2\ell^2}\right) = k_{\text{RBF}}(\mathbf{x}, \mathbf{y})
$$

### Gradient

$$
\nabla_{\mathbf{x}} k_{\text{RQ}}(\mathbf{x}, \mathbf{y}) = -\frac{\alpha}{\ell^2} \cdot \frac{(\mathbf{x} - \mathbf{y})}{1 + \frac{D^2}{2\alpha\ell^2}} \cdot k_{\text{RQ}}(\mathbf{x}, \mathbf{y})
$$

### Advantages âœ…

- Automatically captures **multi-scale features** (Most important!)
- Insensitive to length scale choice (good robustness)
- Infinitely differentiable
- Can serve as efficient replacement for multiple RBF kernels

### Disadvantages âš ï¸

- One more parameter ($\alpha$) to tune compared to RBF
- Slightly more complex computation
- Still uses Euclidean distance (not suitable for manifold data)

### Use Cases

- Data contains features at multiple scales
- Uncertain about appropriate length scale
- Need to capture both local and global patterns simultaneously
- Complex nonlinear regression problems

---

## 5ï¸âƒ£ Kernel Type 5: Polynomial Kernel

### Analytical Expression

$$
k_{\text{Polynomial}}(\mathbf{x}, \mathbf{y}) = (\gamma \cdot \langle\mathbf{x}, \mathbf{y}\rangle + c_0)^d
$$

### Expanded Form

$$
k_{\text{Polynomial}}(\mathbf{x}, \mathbf{y}) = \left(\gamma \sum_{i=1}^{D} x_i y_i + c_0\right)^d
$$

where $D$ is the dimension of input space.

### Parameter Description

**Degree $d \in \mathbb{N}$**:
- Polynomial degree (typically 2-5)
- Higher $d$: captures higher-order feature interactions
- Lower $d$: simpler, more efficient
- **Setting in code**: $d = 3$ (cubic kernel)

**Coefficient $c_0 \geq 0$**:
- Independent term (also called "offset" or "bias")
- $c_0 = 0$: Homogeneous polynomial kernel
- $c_0 = 1$: Inhomogeneous polynomial kernel (more common)
- **Setting in code**: $c_0 = 1.0$

**Scaling factor $\gamma > 0$**:
- Controls the influence of inner product
- Larger $\gamma$: more emphasis on similarity
- Smaller $\gamma$: more conservative
- **Default in code**: $\gamma = 1/D$ (inverse of dimension)

### Special Cases

**Linear kernel** ($d=1, c_0=0$):
$$
k_{\text{Linear}}(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T\mathbf{y}
$$

**Quadratic kernel** ($d=2, c_0=1, \gamma=1$):
$$
k_{\text{Quadratic}}(\mathbf{x}, \mathbf{y}) = (1 + \mathbf{x}^T\mathbf{y})^2
$$

**Cubic kernel** ($d=3, c_0=1, \gamma=1$):
$$
k_{\text{Cubic}}(\mathbf{x}, \mathbf{y}) = (1 + \mathbf{x}^T\mathbf{y})^3
$$

### Feature Space Interpretation ðŸ”‘

The polynomial kernel implicitly maps data to a feature space containing all monomials up to degree $d$:

**Example** (2D quadratic kernel):
$$
\phi(\mathbf{x}) = [1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, \sqrt{2}x_1 x_2, x_2^2]^T
$$

Then:
$$
k(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x})^T \phi(\mathbf{y}) = (1 + x_1 y_1 + x_2 y_2)^2
$$

**Dimension of feature space**:
$$
\dim(\phi) = \binom{D + d}{d} = \frac{(D+d)!}{D! \cdot d!}
$$

For $D=2, d=3$: $\dim = 10$ features  
For $D=10, d=3$: $\dim = 286$ features

### Properties

1. **Range**: Depends on parameters
   - For unit sphere data: $k(\mathbf{x}, \mathbf{y}) \in [(c_0 - \gamma)^d, (c_0 + \gamma)^d]$
   - When $\gamma = 1/D, c_0 = 1$: bounded and well-behaved
2. **Symmetry**: $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{y}, \mathbf{x})$
3. **Self-similarity**: $k(\mathbf{x}, \mathbf{x}) = (\gamma \|\mathbf{x}\|^2 + c_0)^d$
4. **Non-stationary**: Depends on $\mathbf{x}$ and $\mathbf{y}$ separately, not just $\mathbf{x} - \mathbf{y}$
5. **Global behavior**: Does not decay to zero at infinity (unlike RBF)
6. **Smoothness**: Infinitely differentiable for $d \geq 1$

### Gradient

$$
\nabla_{\mathbf{x}} k_{\text{Polynomial}}(\mathbf{x}, \mathbf{y}) = d \gamma (\gamma \langle\mathbf{x}, \mathbf{y}\rangle + c_0)^{d-1} \mathbf{y}
$$

### Special Property for Manifold Data

For data on **unit sphere** where $\|\mathbf{x}\| = \|\mathbf{y}\| = 1$:
- Inner product bounded: $\langle\mathbf{x}, \mathbf{y}\rangle \in [-1, 1]$
- Kernel is bounded: $k(\mathbf{x}, \mathbf{y}) \in [(c_0 - \gamma)^d, (c_0 + \gamma)^d]$
- **No numerical overflow issues** (unlike unbounded polynomial kernels)

**Example** ($d=3, \gamma=0.5, c_0=1$):
- Maximum value: $(1 + 0.5)^3 = 3.375$
- Minimum value: $(1 - 0.5)^3 = 0.125$

### Advantages âœ…

- **Computationally efficient**: Only requires inner product computation
- **Explicit feature interactions**: Captures all polynomial terms up to degree $d$
- **Non-stationary**: Can model data with varying local properties
- **Interpretable**: Clear correspondence to polynomial feature expansion
- **Bounded for manifold data**: Safe from numerical instability on unit sphere

### Disadvantages âš ï¸

- **Sensitive to degree choice**: Too high $d$ â†’ overfitting, too low $d$ â†’ underfitting
- **Global nature**: Does not decay to zero (no locality)
- **Parameter sensitivity**: Poor choice of $\gamma$ or $c_0$ can cause numerical issues
- **Not suitable for all problems**: Works best when data has polynomial structure

### Comparison with Other Kernels

| Aspect | Polynomial | RBF | Spherical |
|--------|-----------|-----|-----------|
| **Locality** | Global | Local | Local |
| **Decay** | No decay | Exponential | Exponential |
| **Stationarity** | Non-stationary | Stationary | Stationary |
| **Parameters** | 3 ($d, \gamma, c_0$) | 1 ($\varepsilon$) | 1 ($\theta$) |
| **Computation** | Very fast | Fast | Medium |
| **Manifold-aware** | No | No | Yes |

### Use Cases

- **Support Vector Machines (SVM)**: Classic choice for classification
- **Data with polynomial structure**: When you know interactions are polynomial
- **Computer vision**: Image classification tasks
- **Text classification**: Document similarity
- **Low-dimensional problems**: Works best in 2D-10D
- **When locality is NOT desired**: Global patterns important

### Practical Tips

1. **Degree selection**:
   - $d=2$: Good starting point, captures pairwise interactions
   - $d=3$: Most common choice, good balance
   - $d \geq 4$: Use with caution, prone to overfitting

2. **Parameter tuning**:
   - Start with $\gamma = 1/D, c_0 = 1$
   - If kernel values too large: decrease $\gamma$
   - If kernel values too small: increase $c_0$

3. **Normalization**:
   - For manifold data: inputs already normalized
   - For general data: consider standardization

4. **Numerical stability**:
   - Use float64 for high degrees ($d \geq 4$)
   - Monitor condition number of Gram matrix

---

## ðŸ“ Application in Kernel EDMD

### Gram Matrix (Kernel Matrix)

For datasets $X = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$ and $Y = \{\mathbf{y}_1, \ldots, \mathbf{y}_m\}$:

$$
K \in \mathbb{R}^{n \times m}, \quad K_{ij} = k(\mathbf{x}_i, \mathbf{y}_j)
$$

### Koopman Operator Approximation

In Kernel EDMD, the Koopman operator is approximated as:

$$
\mathcal{K}_{\text{Koopman}} \approx K_{xy} (K_{xx} + \gamma I)^{-1}
$$

where:
- $K_{xx} = k(X_{\text{tar}}, X_{\text{tar}})$: Auto-kernel matrix of current state $(n \times n)$
- $K_{xy} = k(X_{\text{tar}}, X_{\text{tar next}})$: Cross-kernel matrix $(n \times n)$
- $\gamma > 0$: Tikhonov regularization parameter (in code $\gamma = 10^{-6}$)
- $I$: Identity matrix

### Role of Kernel Functions

1. **Feature mapping**: Maps data from original space to high-dimensional feature space
2. **Similarity measure**: Quantifies "similarity" between two data points
3. **Manifold learning**: For Spherical kernel, learns geometric structure on manifold
4. **Boundary constraint**: For Spherical kernel, naturally encodes boundary information
5. **Feature interactions**: For Polynomial kernel, explicitly captures polynomial interactions

---

## ðŸ’¡ Practical Tips

### Parameter Tuning Tips

1. **Bandwidth/Scale parameters**:
   - Start from median distance of data
   - Too small: overfitting, only considers neighbors
   - Too large: underfitting, too global

2. **$\theta$ for Spherical kernel**:
   - Semi-circle problem recommendation: $\theta \in [0.2, 0.4]$
   - Need more local: decrease $\theta$
   - Need more global: increase $\theta$

3. **$\nu$ for MatÃ©rn kernel**:
   - Smooth data: $\nu = 2.5$
   - Medium data: $\nu = 1.5$
   - Rough data: $\nu = 0.5$

4. **$\alpha$ for Rational Quadratic kernel**:
   - Small scale important: $\alpha \in [1, 2]$
   - Large scale important: $\alpha \in [3, 5]$
   - Balanced: $\alpha = 2$

5. **$d$ for Polynomial kernel**:
   - Start with $d = 2$ or $d = 3$
   - Higher $d$: risk of overfitting
   - Use cross-validation to select

---

## ðŸ“š References

1. **RBF Kernel**: 
   - SchÃ¶lkopf, B., & Smola, A. J. (2002). *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. MIT Press.

2. **Spherical/Geodesic Kernel**: 
   - Feragen, A., Lauze, F., & Hauberg, S. (2015). "    " *CVPR 2015*.

3. **MatÃ©rn Kernel**: 
   - Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

4. **Rational Quadratic Kernel**: 
   - Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

5. **Polynomial Kernel**:
   - SchÃ¶lkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press. Chapter 2.
   - Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*. Cambridge University Press.

6. **Kernel EDMD**:  
   - Williams, M. O., Rowley, C. W., & Kevrekidis, I. G. (2015). "A kernel-based method for data-driven Koopman spectral analysis." *Journal of Computational Dynamics*, 2(2), 247â€“265. DOI: 10.3934/jcd.2015005 

7. **KSWGD (Kernelized Wasserstein Gradient Flow)**:  
   - Chewi, S., Le Gouic, T., Lu, C., Maunu, T., & Rigollet, P. (2020). "SVGD as a kernelized Wasserstein gradient flow of the chi-squared divergence." *Advances in Neural Information Processing Systems (NeurIPS 2020).

---

## Appendix: Code Implementation Tips

### Numerical Stability Tricks

1. **Avoid arccos domain errors**:
   ```python
   cos_sim = np.clip(cos_sim, -1.0, 1.0)  # Ensure within [-1, 1]
   ```

2. **Avoid division by zero**:
   ```python
   X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
   ```

3. **Avoid log(0) and exp(large number)**:
   ```python
   # Use numerically stable tricks when computing log-sum-exp
   log_K = -D2 / (2 * epsilon)
   max_log_K = np.max(log_K, axis=1, keepdims=True)
   K = np.exp(log_K - max_log_K)  # Subtract max to avoid overflow
   ```

4. **For Polynomial kernel, avoid overflow**:
   ```python
   # Use log-space computation for very high degrees
   if degree > 10:
       log_inner = np.log(np.abs(gamma * inner_prod + coef0))
       K = np.sign(gamma * inner_prod + coef0) * np.exp(degree * log_inner)
   ```
