# Theoretical Framework: Kernel EDMD + Manifold-Constrained Langevin Dynamics + LAWGD

**Document Created**: October 5, 2025  
**Applicable Code**: `test_1_script_nd_sphere_kernel_edmd_full_semi_sphere.py`  
**Purpose**: Complete mathematical and analytical foundations (implementation-independent)

---

## 🎓 **Overview of Theoretical Framework**

This code integrates three core theoretical components:
1. **Kernel EDMD (Extended Dynamic Mode Decomposition)**
2. **Manifold-Constrained Langevin Dynamics**
3. **LAWGD (Langevin-Adjusted Wasserstein Gradient Descent)**

---

## 📐 **1. Kernel EDMD Theory**

### **Core Concept**
Learn the **Koopman operator** $\mathcal{K}$, which describes the evolution of dynamical systems:
$$
\mathcal{K}: \mathcal{M} \to \mathcal{M}
$$
$$
\mathbf{x}_{t+\Delta t} = \mathcal{K}(\mathbf{x}_t)
$$

### **Mathematical Formulation**
Via kernel methods, the Koopman operator is approximated in the RKHS (Reproducing Kernel Hilbert Space) as:
$$
\mathcal{K} \approx K_{xy} (K_{xx} + \gamma I)^{-1}
$$

where:
- $K_{xx} = [k(\mathbf{x}_i, \mathbf{x}_j)]_{i,j=1}^n \in \mathbb{R}^{n \times n}$: Gram matrix of current states
- $K_{xy} = [k(\mathbf{x}_i, \mathbf{y}_j)]_{i,j=1}^n$: Cross-Gram matrix from current to future states
- $\gamma > 0$: Tikhonov regularization parameter
- $k(\cdot, \cdot)$: Kernel function (RBF, Spherical, Matérn, Rational Quadratic)

### **Theoretical Foundation**

#### **From DMD to Kernel EDMD**

**1. Dynamic Mode Decomposition (DMD)**  
For linear systems $\mathbf{x}_{k+1} = A\mathbf{x}_k$, DMD approximates:
$$
A \approx Y X^\dagger = Y X^T (X X^T)^{-1}
$$
where $X = [\mathbf{x}_1, \ldots, \mathbf{x}_n]$ and $Y = [\mathbf{y}_1, \ldots, \mathbf{y}_n]$.

**2. Extended DMD (EDMD)**  
For nonlinear systems, lift to higher-dimensional feature space via dictionary functions $\boldsymbol{\psi}: \mathbb{R}^d \to \mathbb{R}^p$:
$$
\boldsymbol{\psi}(\mathbf{x}_{k+1}) \approx K \boldsymbol{\psi}(\mathbf{x}_k)
$$
The Koopman operator is approximated as:
$$
K \approx \Psi_Y \Psi_X^\dagger = \Psi_Y \Psi_X^T (\Psi_X \Psi_X^T + \gamma I)^{-1}
$$

**3. Kernel EDMD**  
Replace explicit dictionaries with kernel functions via the **representer theorem**:
$$
k(\mathbf{x}, \mathbf{y}) = \langle \boldsymbol{\psi}(\mathbf{x}), \boldsymbol{\psi}(\mathbf{y}) \rangle_{\mathcal{H}}
$$

This leads to:
$$
\mathcal{K} = K_{xy} (K_{xx} + \gamma I)^{-1}
$$

### **Koopman Operator Theory**

**Definition**: For a dynamical system $\mathbf{x}_{k+1} = F(\mathbf{x}_k)$, the Koopman operator acts on observables $g: \mathcal{M} \to \mathbb{R}$:
$$
(\mathcal{K} g)(\mathbf{x}) = g(F(\mathbf{x}))
$$

**Key Properties**:
1. **Linearity**: $\mathcal{K}$ is linear even for nonlinear $F$
2. **Spectrum**: Eigenvalues $\lambda_i$ encode stability and oscillatory modes
3. **Invariant subspaces**: Eigenfunctions span finite-dimensional approximations

**Spectral Decomposition**:
$$
\mathcal{K} = \sum_{i=1}^r \lambda_i \phi_i \otimes \psi_i
$$
where $\{\phi_i\}$ are Koopman eigenfunctions and $\{\psi_i\}$ are dual basis functions.

### **Regularization and Stability**

The Tikhonov regularization term $\gamma I$ ensures:
1. **Numerical stability**: Prevents ill-conditioning when $K_{xx}$ is nearly singular
2. **Generalization**: Reduces overfitting to training data
3. **Bias-variance trade-off**: 
   - Small $\gamma$: Low bias, high variance (overfitting)
   - Large $\gamma$: High bias, low variance (underfitting)

**Optimal choice** (from theory):
$$
\gamma_{\text{opt}} \sim \frac{\sigma^2}{n}
$$
where $\sigma^2$ is noise variance and $n$ is sample size.

---

## 🌀 **2. Manifold-Constrained Langevin Dynamics**

### **Standard Langevin SDE**

In Euclidean space $\mathbb{R}^d$, Langevin dynamics is defined as:
\begin{equation*}
d\mathbf{X}_t = \nabla \log \pi(\mathbf{X}_t) \, dt + \sqrt{2D} \, d\mathbf{W}_t
\end{equation*}

where:
- $\nabla \log \pi(\mathbf{x})$: **Score function** (gradient of log target density $\pi$)
- $D$: Diffusion coefficient (in code: $D=1$, so $\sqrt{2D} = \sqrt{2}$)
- $d\mathbf{W}_t$: Standard Wiener process (Brownian motion)

### **Fokker-Planck Equation**

The evolution of the probability density $\rho_t(\mathbf{x})$ is governed by:
$$
\frac{\partial \rho_t}{\partial t} = \nabla \cdot \left( \rho_t \nabla \log \pi + D \nabla \rho_t \right) = D \nabla \cdot \left( \rho_t \nabla \log \frac{\pi}{\rho_t} \right)
$$

**Key Result**: As $t \to \infty$, $\rho_t \to \pi$ (stationary distribution).

### **Manifold-Constrained Version**

When data is constrained to a manifold $\mathcal{M}$ (e.g., unit sphere $\mathbb{S}^{d-1}$), the SDE is modified:
$$
d\mathbf{X}_t = \mathbf{P}_{\mathbf{X}_t} \nabla \log \pi(\mathbf{X}_t) \, dt + \sqrt{2D} \, \mathbf{P}_{\mathbf{X}_t} \, d\mathbf{W}_t
$$
$$
\text{subject to: } \mathbf{X}_t \in \mathcal{M}
$$

where $\mathbf{P}_{\mathbf{x}}$ is the **tangent space projection operator** at point $\mathbf{x}$:
$$
\mathbf{P}_{\mathbf{x}} = I - \mathbf{n}(\mathbf{x}) \otimes \mathbf{n}(\mathbf{x})^T
$$
and $\mathbf{n}(\mathbf{x})$ is the unit normal vector to the manifold at $\mathbf{x}$.

**For unit sphere** $\mathbb{S}^{d-1} = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\| = 1\}$:
$$
\mathbf{n}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|}
$$

### **Geometric Interpretation**

**Tangent Bundle**: At each point $\mathbf{x} \in \mathcal{M}$, the tangent space $T_{\mathbf{x}}\mathcal{M}$ consists of all vectors tangent to the manifold:
$$
T_{\mathbf{x}}\mathcal{M} = \{\mathbf{v} \in \mathbb{R}^d : \langle \mathbf{v}, \mathbf{n}(\mathbf{x}) \rangle = 0\}
$$

**Projection Operation**: $\mathbf{P}_{\mathbf{x}}$ removes the normal component:
$$
\mathbf{P}_{\mathbf{x}} \mathbf{v} = \mathbf{v} - \langle \mathbf{v}, \mathbf{n}(\mathbf{x}) \rangle \mathbf{n}(\mathbf{x})
$$

This ensures all dynamics occur within the tangent space, maintaining the manifold constraint.

### **Boundary Conditions**

For semi-sphere $\mathcal{M}_{\text{semi}} = \{\mathbf{x} \in \mathbb{S}^{d-1} : x_d \geq 0\}$, apply **reflecting boundary condition**:
$$
\text{if } \mathbf{X}_t \notin \mathcal{M}_{\text{semi}}: \quad \mathbf{X}_t \to R(\mathbf{X}_t)
$$

where $R$ is the mirror reflection operator about the boundary hyperplane $\{x_d = 0\}$:
$$
R(\mathbf{x}) = \mathbf{x} - 2 \langle \mathbf{x}, \mathbf{e}_d \rangle \mathbf{e}_d
$$

**For 2D semi-circle** (upper half):
$$
R(x, y) = \begin{cases}
(x, y) & \text{if } y \geq 0 \\
(x, -y) & \text{if } y < 0
\end{cases}
$$

### **Theoretical Properties**

1. **Ergodicity**: Under suitable conditions, the process explores the entire manifold
2. **Stationary Distribution**: As $t \to \infty$, $\mathbf{X}_t \sim \pi|_{\mathcal{M}}$ (restricted to manifold)
3. **Geometric Preservation**: Maintains Riemannian geometric structure
4. **Detailed Balance**: For symmetric $\pi$, the process satisfies detailed balance

### **Discretization: Euler-Maruyama Scheme**

The continuous SDE is discretized as:
$$
\mathbf{X}_{k+1} = \Pi_{\mathcal{M}}\left( \mathbf{X}_k + \Delta t \cdot \mathbf{P}_{\mathbf{X}_k} \nabla \log \pi(\mathbf{X}_k) + \sqrt{2D\Delta t} \cdot \mathbf{P}_{\mathbf{X}_k} \boldsymbol{\xi}_k \right)
$$

where:
- $\boldsymbol{\xi}_k \sim \mathcal{N}(0, I)$: Standard Gaussian noise
- $\Pi_{\mathcal{M}}$: Projection operator onto manifold (e.g., normalization for sphere)
- $\Delta t$: Time step size

**Convergence**: Euler-Maruyama has **strong order 0.5** and **weak order 1.0** convergence.

---

## 🎯 **3. Score Function Estimation (KDE Method)**

### **Problem Statement**

Langevin dynamics requires $\nabla \log \pi(\mathbf{x})$, but $\pi(\mathbf{x})$ is unknown (only samples available).

### **Solution: Kernel Density Estimation (KDE)**

Estimate density using Gaussian kernel:
$$
\hat{\pi}(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^n \frac{1}{(2\pi h^2)^{d/2}} \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_i\|^2}{2h^2}\right)
$$

**Score Function** (gradient of log density):
$$
\nabla \log \hat{\pi}(\mathbf{x}) = \frac{\nabla \hat{\pi}(\mathbf{x})}{\hat{\pi}(\mathbf{x})}
$$

**Explicit Formula**:
$$
\nabla \log \hat{\pi}(\mathbf{x}) = \frac{\sum_{i=1}^n w_i (\mathbf{x}_i - \mathbf{x})}{\sum_{i=1}^n w_i} \cdot \frac{1}{h^2}
$$

where the weights are:
$$
w_i = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_i\|^2}{2h^2}\right)
$$

### **Bandwidth Selection**

**Silverman's Rule of Thumb**:
$$
h = \left(\frac{4}{d+2}\right)^{1/(d+4)} n^{-1/(d+4)} \hat{\sigma}
$$

where $\hat{\sigma}$ is the sample standard deviation.

**Simplified Version** (used in code):
$$
h \approx \sqrt{\text{median}\left(\{\|\mathbf{x}_i - \mathbf{x}_j\|^2 : i \neq j\}\right)}
$$

This is a **robust estimator** less sensitive to outliers than mean-based methods.

### **Theoretical Justification**

**Stein's Identity**: For smooth functions $f$ with $\pi f \to 0$ at infinity:
$$
\mathbb{E}_{\pi}\left[f(\mathbf{x}) \nabla \log \pi(\mathbf{x})\right] = -\mathbb{E}_{\pi}[\nabla f(\mathbf{x})]
$$

This forms the basis for **score matching** methods.

**Consistency**: Under regularity conditions:
$$
\nabla \log \hat{\pi}(\mathbf{x}) \xrightarrow{n \to \infty} \nabla \log \pi(\mathbf{x})
$$

**Convergence Rate**: For smooth densities in $d$ dimensions:
$$
\left\|\nabla \log \hat{\pi} - \nabla \log \pi\right\|_2 = O_p\left(n^{-\frac{2}{d+4}}\right)
$$

---

## 🚀 **4. LAWGD (Langevin-Adjusted Wasserstein Gradient Descent)**

### **Core Concept**

Combine Langevin dynamics with Wasserstein gradient flow to transport particles from source distribution $\rho_0$ to target distribution $\pi$.

### **Wasserstein Distance**

The **Wasserstein-2 distance** (also called Kantorovich-Rubinstein distance) between two distributions $\rho$ and $\pi$ is:
$$
W_2(\rho, \pi) = \inf_{\gamma \in \Gamma(\rho, \pi)} \left( \int_{\mathcal{M} \times \mathcal{M}} \|\mathbf{x} - \mathbf{y}\|^2 \, d\gamma(\mathbf{x}, \mathbf{y}) \right)^{1/2}
$$

where $\Gamma(\rho, \pi)$ is the set of all joint distributions with marginals $\rho$ and $\pi$.

**Interpretation**: Optimal cost of transporting mass from $\rho$ to $\pi$ with quadratic cost.

### **Wasserstein Gradient Flow**

Minimize functional $\mathcal{F}[\rho]$ (e.g., KL divergence) via gradient flow:
$$
\min_{\rho} \mathcal{F}[\rho] = \int_{\mathcal{M}} \rho(\mathbf{x}) \log \frac{\rho(\mathbf{x})}{\pi(\mathbf{x})} \, d\mathbf{x}
$$

The **gradient flow** is described by the continuity equation:
$$
\frac{\partial \rho_t}{\partial t} + \nabla \cdot (\rho_t \mathbf{v}_t) = 0
$$

where the velocity field satisfies:
$$
\mathbf{v}_t = -\nabla \frac{\delta \mathcal{F}}{\delta \rho}
$$

### **JKO Scheme (Jordan-Kinderlehrer-Otto)**

Discrete-time formulation:
$$
\rho_{k+1} = \arg\min_{\rho} \left\{ \frac{1}{2\tau} W_2^2(\rho, \rho_k) + \mathcal{F}[\rho] \right\}
$$

This is a **proximal gradient descent** in Wasserstein space.

### **Langevin Adjustment**

Add diffusion term to enhance exploration:
$$
\frac{\partial \rho_t}{\partial t} = \nabla \cdot \left( \rho_t \nabla \frac{\delta \mathcal{F}}{\delta \rho} + D \nabla \rho_t \right)
$$

**Combined Flow**:
$$
\frac{\partial \rho_t}{\partial t} = D \nabla \cdot \left( \rho_t \nabla \log \frac{\pi}{\rho_t} \right)
$$

This is exactly the **Fokker-Planck equation** for Langevin dynamics!

### **Particle Approximation**

Represent distributions via $N$ particles $\{\mathbf{x}_i^{(k)}\}_{i=1}^N$:
$$
\rho_k = \frac{1}{N} \sum_{i=1}^N \delta_{\mathbf{x}_i^{(k)}}
$$

Update particles via:
$$
\mathbf{x}_i^{(k+1)} = \mathbf{x}_i^{(k)} + \Delta t \cdot \mathbf{v}(\mathbf{x}_i^{(k)}) + \sqrt{2D \Delta t} \cdot \boldsymbol{\xi}_i
$$

where the velocity field $\mathbf{v}$ is determined by the Koopman operator:
$$
\mathbf{v}(\mathbf{x}) = \frac{\mathcal{K}(\mathbf{x}) - \mathbf{x}}{\Delta t}
$$

### **Theoretical Guarantees**

**Convergence**: Under suitable conditions (log-Sobolev inequality, etc.):
$$
W_2(\rho_t, \pi) \leq e^{-\lambda t} W_2(\rho_0, \pi)
$$

with rate $\lambda > 0$ depending on:
- Curvature of the manifold
- Smoothness of $\pi$
- Diffusion coefficient $D$

**Entropy Decay**: The relative entropy (KL divergence) decays:
$$
\mathcal{F}[\rho_t] \leq e^{-2\lambda t} \mathcal{F}[\rho_0]
$$

---

## 🔗 **5. Theoretical Integration**

### **Complete Mathematical Description**

#### **Phase 1: Data Generation (Langevin SDE on Manifold)**

Generate training pairs $(\mathbf{X}_t, \mathbf{X}_{t+\Delta t})$ via:
$$
\mathbf{X}_{t+\Delta t} = \Pi_{\mathcal{M}_{\text{semi}}}\left( \mathbf{X}_t + \Delta t \cdot \mathbf{P}_{\mathbf{X}_t} \nabla \log \pi(\mathbf{X}_t) + \alpha \sqrt{2D\Delta t} \cdot \mathbf{P}_{\mathbf{X}_t} \boldsymbol{\xi} \right)
$$

where:
- $\mathcal{M}_{\text{semi}} = \{\mathbf{x} \in \mathbb{S}^{d-1} : x_d \geq 0\}$: Semi-sphere manifold
- $\Pi_{\mathcal{M}_{\text{semi}}}$: Projection + reflection operator
- $\alpha > 1$: Noise amplification factor (to enhance boundary exploration)

**Purpose**: Generate data pairs that include sufficient boundary crossings for learning.

#### **Phase 2: Koopman Learning (Kernel EDMD)**

Learn the Koopman operator by solving:
$$
\mathcal{K} = \arg\min_{\mathcal{L} \in \mathcal{H}} \sum_{i=1}^n \left\| \mathcal{L}(\mathbf{x}_i) - \mathbf{x}_i' \right\|^2 + \gamma \|\mathcal{L}\|_{\mathcal{H}}^2
$$

**Closed-form solution** (representer theorem):
$$
\mathcal{K} = K_{xy} (K_{xx} + \gamma I)^{-1}
$$

**Purpose**: Learn the dynamical flow field $\mathbf{X}_t \mapsto \mathbf{X}_{t+\Delta t}$ from data.

#### **Phase 3: Particle Transport (LAWGD)**

Transport particles from initial distribution $\rho_0$ to target distribution $\pi$ via:
$$
\mathbf{x}_i^{(k+1)} = \mathcal{K}(\mathbf{x}_i^{(k)})
$$

**No artificial boundary enforcement** during transport! The learned Koopman operator should naturally keep particles within the physical domain.

**Purpose**: Test if the learned dynamics respects boundary constraints.

### **Key Theoretical Insight**

**Separation of Concerns**:
1. **Training Data Generation**: Include boundary physics (reflection)
2. **Operator Learning**: Learn from data (kernel methods)
3. **Inference**: Trust the learned operator (no manual constraints)

**Philosophy**: The algorithm learns **from** data with boundaries, not **with** artificial constraints during learning.

---

## 📚 **Mathematical Foundations**

### **Key Theorems and Results**

#### **Representer Theorem**

For RKHS $\mathcal{H}$ with kernel $k$, the solution to:
$$
\min_{f \in \mathcal{H}} \sum_{i=1}^n L(y_i, f(\mathbf{x}_i)) + \gamma \|f\|_{\mathcal{H}}^2
$$

has the form:
$$
f^*(\mathbf{x}) = \sum_{i=1}^n \alpha_i k(\mathbf{x}, \mathbf{x}_i)
$$

**Implication**: Infinite-dimensional optimization reduces to finite-dimensional problem.

#### **Koopman Spectral Theorem**

For measure-preserving dynamical system, the Koopman operator has:
- **Spectrum on unit circle**: $|\lambda_i| = 1$ (conservative systems)
- **Eigenfunctions**: Form orthonormal basis in $L^2(\mathcal{M}, \mu)$
- **Discrete spectrum**: For systems with mixing properties

#### **Fokker-Planck-Kolmogorov Theorem**

For SDE $d\mathbf{X}_t = \mathbf{b}(\mathbf{X}_t) dt + \sigma(\mathbf{X}_t) d\mathbf{W}_t$, the density $\rho_t$ satisfies:
$$
\frac{\partial \rho_t}{\partial t} = -\nabla \cdot (\mathbf{b} \rho_t) + \frac{1}{2} \nabla \cdot \nabla \cdot ((\sigma \sigma^T) \rho_t)
$$

**For Langevin**: $\mathbf{b} = \nabla \log \pi$, $\sigma \sigma^T = 2D I$ gives:
$$
\frac{\partial \rho_t}{\partial t} = D \nabla \cdot (\rho_t \nabla \log(\pi/\rho_t))
$$

#### **Log-Sobolev Inequality**

If $\pi$ satisfies log-Sobolev inequality with constant $C_{\text{LS}}$:
$$
\text{Ent}(\rho | \pi) \leq \frac{C_{\text{LS}}}{2} \int \left|\nabla \log \frac{\rho}{\pi}\right|^2 \rho \, d\mathbf{x}
$$

then Langevin dynamics converges exponentially:
$$
W_2(\rho_t, \pi) \leq e^{-t/C_{\text{LS}}} W_2(\rho_0, \pi)
$$

---

## 📊 **Theoretical Modules Summary**

| Module | Mathematical Basis | Key Theorems |
|--------|-------------------|--------------|
| **Kernel EDMD** | Functional analysis, Koopman theory | Representer theorem, Spectral theory |
| **Langevin Dynamics** | Stochastic differential equations | Fokker-Planck equation, Ergodicity |
| **Manifold Constraint** | Differential geometry, Riemannian geometry | Tangent bundle, Exponential map |
| **Score Matching** | Nonparametric statistics | Stein's identity, Denoising score matching |
| **Wasserstein Gradient** | Optimal transport theory | Benamou-Brenier formula, JKO scheme |
| **KDE** | Nonparametric estimation | Parzen window, Kernel smoothing |

---

## 🎯 **Theoretical Innovations**

### **1. Geometry-Aware Kernels**

Use **geodesic distance** (Spherical kernel) instead of Euclidean distance:
$$
k_{\text{Spherical}}(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\arccos^2(\langle \hat{\mathbf{x}}, \hat{\mathbf{y}} \rangle)}{2\theta^2}\right)
$$

**Advantage**: Respects manifold geometry, naturally encodes boundary information.

### **2. Boundary Learning via Enhanced Noise**

Amplify diffusion coefficient to increase boundary crossing rate:
$$
d\mathbf{X}_t = \nabla_{\text{tan}} \log \pi(\mathbf{X}_t) dt + \alpha \sqrt{2D} \, d\mathbf{W}_{\text{tan}}
$$

with $\alpha > 1$ (e.g., $\alpha = 3$).

**Theoretical Justification**: Ensures $O(10\%-30\%)$ of training pairs involve boundary reflection, providing sufficient gradient information for learning boundary dynamics.

### **3. Data-Driven vs. Constraint-Driven**

**Key Distinction**:
- **Training Data**: Generated with boundary constraints (reflection)
- **Inference**: No artificial constraints, purely data-driven

**Theorem (Informal)**: If the training distribution $\rho_{\text{train}}$ has sufficient support near boundaries, then the learned Koopman operator $\mathcal{K}_{\text{learned}}$ approximates the true dynamics $\mathcal{K}_{\text{true}}$ with bounded error:
$$
\left\|\mathcal{K}_{\text{learned}} - \mathcal{K}_{\text{true}}\right\|_{\text{op}} \leq C \cdot \left(\frac{1}{\sqrt{n}} + \text{boundary coverage}\right)
$$

### **4. Spectral Decomposition for Stability**

After computing $\mathcal{K} = K_{xy} (K_{xx} + \gamma I)^{-1}$, perform SVD:
$$
K_{xy} = U \Sigma V^T
$$

**Truncation**: Keep only top $r$ singular values where $\sigma_i > \sigma_{\max} / \kappa$ for condition number $\kappa$.

**Stabilized Operator**:
$$
\mathcal{K}_{\text{stable}} = U_r \Sigma_r V_r^T (K_{xx} + \gamma I)^{-1}
$$

---

## 📖 **Core Mathematical Objects**

### **Manifold Structure**
- **Semi-sphere**: $\mathcal{M}_{\text{semi}} = \mathbb{S}^{d-1}_+ = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\|=1, x_d \geq 0\}$
- **Tangent space**: $T_{\mathbf{x}}\mathcal{M} = \{\mathbf{v} : \langle \mathbf{v}, \mathbf{x} \rangle = 0\}$
- **Riemannian metric**: Inherited from $\mathbb{R}^d$

### **Probability Measures**
- **Target distribution**: $\pi: \mathcal{M} \to \mathbb{R}_+$ with $\int_{\mathcal{M}} \pi(\mathbf{x}) d\mathbf{x} = 1$
- **Empirical measure**: $\hat{\pi}_n = \frac{1}{n} \sum_{i=1}^n \delta_{\mathbf{x}_i}$
- **Wasserstein space**: $(\mathcal{P}_2(\mathcal{M}), W_2)$ is a metric space

### **Function Spaces**
- **Koopman operator domain**: $\mathcal{K}: L^2(\mathcal{M}, \mu) \to L^2(\mathcal{M}, \mu)$
- **RKHS**: $\mathcal{H}_k = \overline{\text{span}\{k(\cdot, \mathbf{x}) : \mathbf{x} \in \mathcal{M}\}}$
- **Sobolev space**: $W^{s,p}(\mathcal{M})$ for regularity conditions

### **Operators**
- **Koopman operator**: $\mathcal{K}: \mathcal{M} \to \mathcal{M}$
- **Projection operator**: $\mathbf{P}_{\mathbf{x}}: T_{\mathbf{x}}\mathbb{R}^d \to T_{\mathbf{x}}\mathcal{M}$
- **Reflection operator**: $R: \mathcal{M} \cup \partial\mathcal{M} \to \mathcal{M}$

### **Kernel Functions**
$$
k: \mathcal{M} \times \mathcal{M} \to \mathbb{R}
$$

Must satisfy:
1. **Symmetry**: $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{y}, \mathbf{x})$
2. **Positive definiteness**: Gram matrix $K$ is positive semi-definite
3. **Continuity**: $k$ is continuous in both arguments

### **Score Function**
$$
s(\mathbf{x}) = \nabla \log \pi(\mathbf{x}) = \frac{\nabla \pi(\mathbf{x})}{\pi(\mathbf{x})}
$$

**Properties**:
- **Divergence-free condition**: $\int \nabla \cdot (s \pi) d\mathbf{x} = 0$
- **Score matching objective**: $\mathbb{E}\left[\|s - s_{\theta}\|^2\right]$ for parameterized $s_{\theta}$

---

## 🔬 **Advanced Topics**

### **Convergence Analysis**

**Theorem (Kernel EDMD Consistency)**: Under regularity conditions, as $n \to \infty$ and $\gamma \to 0$:
$$
\left\|\mathcal{K}_n - \mathcal{K}_{\text{true}}\right\|_{\text{op}} = O_p\left(\sqrt{\frac{\log n}{n}} + \gamma\right)
$$

**Proof sketch**: Combine concentration inequalities for empirical kernel matrices with operator norm bounds.

### **Computational Complexity**

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Kernel matrix $K_{xx}$ | $O(n^2 d)$ | Pairwise evaluations |
| Matrix inversion | $O(n^3)$ | Via Cholesky or SVD |
| Score estimation (KDE) | $O(n^2 d)$ | All pairwise distances |
| Particle update | $O(Nn)$ | $N$ particles, $n$ training points |

**GPU Acceleration**: Kernel matrix operations are highly parallelizable.

### **Error Decomposition**

Total error in learned Koopman operator:
$$
\varepsilon_{\text{total}} = \underbrace{\varepsilon_{\text{approx}}}_{\text{function class}} + \underbrace{\varepsilon_{\text{estimation}}}_{\text{finite samples}} + \underbrace{\varepsilon_{\text{optimization}}}_{\text{numerical}}
$$

- **Approximation error**: Depends on RKHS richness
- **Estimation error**: $O(n^{-1/2})$ rate
- **Optimization error**: Machine precision ($\sim 10^{-16}$ for float64)

---

## 🎓 **Appendix: Mathematical Prerequisites**

### **Required Background**

1. **Functional Analysis**:
   - Hilbert spaces, inner products, operators
   - Reproducing kernel Hilbert spaces (RKHS)
   - Spectral theory, eigenvalue decomposition

2. **Differential Geometry**:
   - Manifolds, tangent spaces, tangent bundles
   - Riemannian metrics, geodesics
   - Exponential map, parallel transport

3. **Probability Theory**:
   - Measure theory, probability measures
   - Stochastic processes, Brownian motion
   - Martingales, stochastic calculus (Itô calculus)

4. **Optimal Transport**:
   - Wasserstein distances, Kantorovich duality
   - Gradient flows in probability space
   - JKO scheme, continuity equations

5. **Numerical Analysis**:
   - Discretization of SDEs (Euler-Maruyama, etc.)
   - Matrix decompositions (SVD, eigendecomposition)
   - Regularization techniques (Tikhonov, spectral filtering)

---

## 📝 **Notation Summary**

| Symbol | Meaning |
|--------|---------|
| $\mathcal{M}$ | Manifold (unit sphere or semi-sphere) |
| $\mathbf{x}, \mathbf{y}$ | Points on manifold |
| $\pi(\mathbf{x})$ | Target probability density |
| $\rho_t(\mathbf{x})$ | Time-evolving density |
| $\mathcal{K}$ | Koopman operator |
| $k(\mathbf{x}, \mathbf{y})$ | Kernel function |
| $K_{xx}, K_{xy}$ | Gram matrices |
| $\mathbf{P}_{\mathbf{x}}$ | Tangent space projection operator |
| $\nabla \log \pi$ | Score function |
| $d\mathbf{W}_t$ | Wiener process (Brownian motion) |
| $W_2(\rho, \pi)$ | Wasserstein-2 distance |
| $\mathcal{H}_k$ | RKHS associated with kernel $k$ |
| $\gamma$ | Regularization parameter |
| $D$ | Diffusion coefficient |
| $\Delta t$ | Time step size |

---

**Document End**

This document provides the complete theoretical and analytical foundations for understanding the code, independent of implementation details.
