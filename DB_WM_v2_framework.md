# Deep Basis World Models (DB-WM): Scalable Gaussian Process Dynamics from Visual Observations with Observability and Controllability Guarantees


## 1. Motivation: The Scalability Gap

The previous DK-WM framework (v1) proposed learning dynamics in the weight space of a deep kernel GP. The core operation was:

$$
w_t = K_\mathcal{C}^{-1}\, \kappa(z_t), \quad K_\mathcal{C} \in \mathbb{R}^{M \times M},
$$

where $K_\mathcal{C}$ is the Gram matrix at $M$ dictionary centers. This inversion costs O(M³), which becomes prohibitive for:

- **Satellite imagery**: A single Landsat tile at 30m resolution over a ~4,500 km² study area yields ~5 million pixels. Even downsampled, $n$ can reach $10^4$–$10^5$.
- **Fluid dynamics**: CFD snapshots at Re = 1000 contain 95,000 velocity data points per snapshot.
- **Robotic manipulation**: Real-time planning requires sub-second inference.

The DBK framework (Zhu, Yuchi & Xie, 2026) resolves this by constructing the kernel directly as an inner product of neural basis functions, yielding an **explicit low-rank structure** that reduces GP inference to O(nr²) without any matrix inversion of size $n$ or $M$.

---

## 2. The Deep Basis World Model (DB-WM)

### 2.1 Core Construction: DBK as Latent State Representation

**Definition 2.1 (Deep Basis Map).** Let $\phi_\theta: \mathcal{O} \to \mathbb{R}^r$ be a neural network mapping observations (pixel images, satellite tiles, or sensor readings) to an $r$-dimensional basis representation:

$$
\phi_\theta(o) = [\phi_1(o), \ldots, \phi_r(o)]^\top,
$$

where each $\phi_i: \mathcal{O} \to \mathbb{R}$ is a scalar-valued basis function parameterized by the shared network $\phi_\theta$.

**Definition 2.2 (Deep Basis Kernel).** The DBK is defined as:

$$
k(o, o') = \sum_{i=1}^{r} \phi_i(o)\,\phi_i(o') = \langle \phi_\theta(o),\; \phi_\theta(o') \rangle. \tag{DBK}
$$

By construction, $k$ is positive semi-definite (Schölkopf & Smola, 2002). The kernel matrix at $n$ observations is $K_{XX} = \Phi_X \Phi_X^\top$, where $\Phi_X \in \mathbb{R}^{n \times r}$ with rows $\{\phi_\theta(o_i)^\top\}_{i=1}^n$, and has rank at most $r$.

**Theorem 2.1 (Universal Approximation of Kernels via DBK, Mercer).** Any continuous, symmetric, positive semi-definite kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ on a compact set $\mathcal{X} \subset \mathbb{R}^d$ admits the expansion $k(x,x') = \lim_{r \to \infty} \sum_{i=1}^r \lambda_i \psi_i(x) \psi_i(x')$, where $\lambda_i \geq 0$ are eigenvalues and $\psi_i$ are orthonormal eigenfunctions. By the universal approximation theorem, a deep basis can parameterize $\phi_i(x) = \sqrt{\lambda_i}\,\psi_i(x)$ arbitrarily well. Therefore, DBKs can approximate any continuous kernel as $r$ and network capacity increase.

### 2.2 Weight-Space Dynamics

The weight-space view of the DBK gives the generative model:

$$
f_t(x) = \langle w_t,\; \phi_\theta(x) \rangle, \quad w_t \in \mathbb{R}^r, \quad w_0 \sim \mathcal{N}(0, I_r).
$$

The temporal evolution of the spatiotemporal field is captured by linear dynamics in the latent basis-weight space:

$$\boxed{\;w_{t+1} \;=\; \hat{A}\,w_t \;+\; B_p\,p_t \;+\; B_u\,u_t \;+\; \eta_t\;}$$

with $w_t\in\mathbb{R}^r$ the basis weights, $\hat A\in\mathbb{R}^{r\times r}$ the autonomous transition operator, $B_p\in\mathbb{R}^{r}$ the precipitation-forcing direction, $B_u\in\mathbb{R}^{r}$ the irrigation-forcing direction, $p_t\ge 0$ observed precipitation, $u_t\ge 0$ irrigation magnitude, and $\eta_t\sim\mathcal N(0,Q)$ process noise. Stacking the forcing into a single input matrix $B=[\,B_p\ \ B_u\,]\in\mathbb{R}^{r\times\ell}$ and raw input $u_t^{\text{raw}}=[\,p_t,\ u_t\,]^\top\in\mathbb{R}^{\ell}$ ($\ell=2$ here), the model is the standard input-affine LTI form $w_{t+1}=\hat A w_t + B\,u_t^{\text{raw}} + \eta_t$. The application sections specialize $B$ as needed (e.g., $B\,a_t$ with an $\ell$-dimensional action $a_t$ for robotic control).

**Remark 2.1 (Input-affine forcing, not an action-conditioned operator).** We model forcing *additively*, $w_{t+1}=\hat A w_t + B\,u_t^{\text{raw}}+\eta_t$, with a **single, input-independent** operator $\hat A$. We deliberately reject the action-conditioned alternative $w_{t+1}=\hat A(a_t)\,w_t+\eta_t$, in which the transition operator itself is reparameterized by the input/action. The grounds are both structural and theoretical:

1. **Clean disambiguation.** The input-affine form separates the *autonomous* dynamics ($\hat A$) from *controllable* actuation ($B_u u_t$) and *uncontrollable exogenous disturbances* ($B_p p_t$). This separation is exactly what makes the observability/controllability analysis (Propositions 4.2–4.3) and the disambiguated identification (Algorithm 3, Remark 6.1) meaningful.
2. **Closed-form identifiability.** Additive forcing is the discrete-time image of an input-affine control system and is identifiable in closed form by matrix least squares (Algorithm 3). A general action-conditioned operator is not.

Indexing by a *constant physical parameter held fixed within a trajectory* (e.g., a per-Reynolds operator $\hat A_{\text{Re}}$, Section 8.2) is **not** action conditioning: each such operator is fixed along its trajectory and retains a well-defined spectrum.

### 2.3 Observation Model

At each time step, pixel observations provide measurements of the latent state:

$$
y_t = \Phi_{X_t}\, w_t + \zeta_t, \quad \zeta_t \sim \mathcal{N}(0, \sigma_\epsilon^2 I_N), \tag{DB-WM Observation}
$$

where $\Phi_{X_t} \in \mathbb{R}^{N \times r}$ is the feature matrix computed by evaluating the basis map at $N$ observation locations. In the visual setting, a single pixel frame $o_t$ yields $\phi_\theta(o_t) \in \mathbb{R}^r$, so $N = 1$ and $\Phi_{X_t} = \phi_\theta(o_t)^\top \in \mathbb{R}^{1 \times r}$.

### 2.4 Two-Stage Architecture for Multi-Dimensional Inputs

The DBK paper's two-stage architecture (backbone + expansion layer) naturally handles inputs of arbitrary dimension:

**Stage 1: Backbone** $g_\theta: \mathbb{R}^d \to \mathbb{R}^h$

The backbone maps raw inputs to an $h$-dimensional intermediate representation. For different input modalities:

| Input type | Dimension $d$ | Backbone architecture |
|---|---|---|
| 1D time series | $d = 1$ | MLP / ResNet |
| 2D spatial (lat/lon) | $d = 2$ | MLP / ResNet (as in DBK's GA/NM experiments) |
| 2D imagery (single band) | $d = H \times W$ | CNN / ViT |
| Multi-band satellite imagery | $d = C \times H \times W$ | CNN / ViT with multi-channel input |
| 3D volumetric (CFD) | $d = H \times W \times D$ | 3D CNN |

**Stage 2: Expansion Layer** $\text{expand}: \mathbb{R}^h \to \mathbb{R}^r$

The expansion layer lifts the $h$-dimensional representation to $r$ basis functions. Two variants (following Zhu et al., 2026):

- **DB-WM-SwiGLU**: $\phi_\theta(o) = \text{diag}(s) \cdot \sigma_{\text{SwiGLU}}(W\, g_\theta(o) + b)$, where $s \in \mathbb{R}^r$ is a learnable scale, $W \in \mathbb{R}^{r \times h}$, $b \in \mathbb{R}^r$, and $\sigma_{\text{SwiGLU}}$ is the SwiGLU activation. This is the GBLL case (no inducing points).

- **DB-WM-RBF**: $\phi_\theta(o) = \tilde{K}_{ZZ}^{-1/2}\, \tilde{k}_Z(g_\theta(o))$, where $Z = \{z_i\}_{i=1}^r \subset \mathbb{R}^h$ are learnable inducing points in the latent space and $\tilde{k}$ is an RBF base kernel. This is the sparse DKL case.

**Remark 2.2 (Multi-Dimensional Inputs Are Not a Limitation).** The DBK paper's 1D synthetic experiments are for illustration; their mobile internet quality estimation uses 2D spatial inputs ($d = 2$, lat/lon coordinates) with $n > 500{,}000$ data points and $r = 1024$ basis functions. The backbone $g_\theta$ handles the input dimensionality, and the expansion layer always outputs $r$-dimensional bases regardless of $d$. For satellite imagery at resolution $H \times W$ with $C$ spectral bands, a standard CNN or ViT backbone processes the $C \times H \times W$ input and outputs $h$-dimensional features, which the expansion layer maps to $r$ basis functions. The spatiotemporal field at time $t$ is then $f_t(x) = \langle w_t, \phi_\theta(o_t) \rangle$ for any input modality.

### 2.5 Identification of the Latent Dynamics: Least Squares in the Learned Basis

The transition operator $\hat A$ and input matrix $B$ are identified by **regularized matrix least squares** on the sequence of basis weights $\{w_t\}$. This subsection makes precise why this replaces — and subsumes — a Dynamic Mode Decomposition, and why the substitution is purely a matter of description that leaves the resulting operator (and every downstream guarantee) unchanged.

**A reduced-order DMD is two operations.** Given raw snapshots $x_1,\dots,x_m\in\mathbb{R}^n$, a projected DMD (i) computes a truncated SVD / POD of the snapshots to obtain an $r$-dimensional coordinate system, then (ii) fits an $r\times r$ operator by least squares *in those reduced coordinates*. The SVD is not the essence of the dynamics; it is a **data-driven reduction** performed because $n$ is too large to fit an $n\times n$ operator directly.

**The deep basis is the reduction.** In DB-WM the map $\phi_\theta:\mathcal O\to\mathbb R^r$ takes each high-dimensional observation directly into an $r$-dimensional (learned, nonlinear) coordinate system — the weight space, $w_t=\phi_\theta(o_t)$ (visual case, $N=1$) or $w_t=\Lambda_X^{-1}\Phi_X^\top y_t$ (multi-observation GP case). Step (i) is therefore *already performed by the encoder*. Only step (ii) — the least-squares operator fit — remains:

$$
\hat A \;=\; \underset{A\in\mathbb R^{r\times r}}{\arg\min}\ \|W_+ - A\,W_-\|_F^2 + \mu\|A\|_F^2 \;=\; W_+ W_-^\top\big(W_- W_-^\top + \mu I_r\big)^{-1},
$$

with $W_-=[w_1,\dots,w_{T-1}]$, $W_+=[w_2,\dots,w_T]\in\mathbb R^{r\times(T-1)}$. Because $r\le 1024$, forming and inverting the $r\times r$ normal-equations matrix $W_-W_-^\top+\mu I_r$ is inexpensive — $O(Tr^2+r^3)$ — and no separate truncating SVD of any $n$-dimensional object is ever formed.

**The operator is the reduced-order (economized) operator.** Proposition 4.4 shows that, within the learned coordinates, this least-squares fit has the *identical nonzero spectrum and modes* as the reduced operator a projected DMD would return if handed the same $r$-dimensional weight sequence; when the excited data spans $\mathbb R^r$ the two operators are related by an orthonormal similarity. The internal SVD of that projected DMD reduces to a redundant orthonormal whitening of already-reduced coordinates — it removes no dimension. This is exactly why applying a DMD to the weights returns the same operator one obtains by direct least squares: the deep basis has already done the only reduction that matters.

**Bridge to prior terminology (stated once).** The autonomous least-squares fit above is the estimator that the Koopman literature calls the empirical **finite-section (Galerkin) approximation** of the Koopman operator restricted to a dictionary of observables (Williams, Kevrekidis & Rowley, 2015; Korda & Mezić, 2018) — here the "dictionary" is the learned basis $\{\phi_i\circ g_\theta\}$. In a linear/reduced coordinate basis this estimator coincides numerically with the reduced operator of a projected DMD. We adopt the least-squares / finite-section description as primary. The input-augmented version (Section 4.5, Corollary 4.3) is the finite-section of the **Koopman operator with inputs** (Proctor, Brunton & Kutz, 2018); the least-squares fit of $[\hat A\ B]$ is "control on the learned reduced-order dynamics model."

---

## 3. Scalable Inference and Training

### 3.1 Complexity Analysis

All inference operations exploit the low-rank structure $K_{XX} = \Phi_X \Phi_X^\top$ via the **Woodbury identity**:

$$
\Sigma_X^{-1} = (\Phi_X \Phi_X^\top + \sigma_\epsilon^2 I_n)^{-1} = \frac{1}{\sigma_\epsilon^2}\left(I_n - \Phi_X \Lambda_X^{-1} \Phi_X^\top\right),
$$

where $\Lambda_X = \Phi_X^\top \Phi_X + \sigma_\epsilon^2 I_r \in \mathbb{R}^{r \times r}$.

| Operation | Original E-GP | DK-WM v1 | **DB-WM v2** |
|---|---|---|---|
| Gram matrix | $K \in \mathbb{R}^{M \times M}$, O(M²) | $K_\mathcal{C} \in \mathbb{R}^{M \times M}$, O(M²) | $\Lambda_X \in \mathbb{R}^{r \times r}$, **O(nr²)** |
| Matrix inversion | O(M³) | O(M³) | **O(r³)** (with $r \ll n$) |
| Per-step dynamics | O(M²) | O(M²) | **O(r²)** |
| System ID ($\hat{A}$) | O(TM²) | O(TM²) | **O(Tr²)** |
| Kalman filter update | O(M²N) | O(M²N) | **O(r²N)** |
| Log marginal likelihood | O(n³) | O(M³) | **O(nr²)** |
| Mini-batch training (dPPGP) | — | — | **O(br²)** per iteration |
| Prediction | O(M²) | O(M²) | **O(r²)** per test point |

**Proposition 3.1 (Scalable Exact Inference).** For a DB-WM with $n$ training observations, $r$ basis functions, and per-input computation cost $c_\phi$ for the basis map, the exact GP posterior can be evaluated in O(n(r² + c_\phi)) time and O(n(r + c_\phi^{\text{space}})) space.

*Proof.* The posterior mean and variance are:
$$
\hat{\mu}_f(o^*) = \phi_\theta(o^*)^\top \Lambda_X^{-1} \Phi_X^\top y, \quad \hat{\sigma}_f^2(o^*) = \sigma_\epsilon^2\, \phi_\theta(o^*)^\top \Lambda_X^{-1} \phi_\theta(o^*),
$$
where $\Lambda_X = \Phi_X^\top \Phi_X + \sigma_\epsilon^2 I_r$. Computing $\Phi_X^\top \Phi_X$ costs O(nr²); inverting $\Lambda_X$ costs O(r³); computing $\Phi_X^\top y$ costs O(nr). Total: O(nr² + r³) = O(nr²) when $r \ll n$. $\square$

### 3.2 Training Objective: dPPGP for Dynamics

We adopt the decoupled Parametric Predictive GP (dPPGP) objective from Zhu et al. (2026), extended to the temporal dynamics setting. Given a variational posterior $q(w) = \mathcal{N}(w; m, LL^\top)$, the predictive mean and variance at each time step are:

$$
\hat{\mu}_f(o_t) = \langle m, \phi_\theta(o_t) \rangle, \quad \hat{\sigma}_f^2(o_t) = \|L^\top \phi_\theta(o_t)\|^2.
$$

**Definition 3.1 (DB-WM Training Loss).** The complete training objective combines dynamics prediction with dPPGP calibration:

$$
\mathcal{L}_{\text{DB-WM}} = \underbrace{\frac{1}{bT}\sum_{i,t} \|w_{t+1}^{(i)} - \hat{A}\, w_t^{(i)} - B_p\, p_t^{(i)} - B_u\, u_t^{(i)}\|^2}_{\text{Dynamics prediction (input-affine)}} + \lambda_1 \underbrace{\mathcal{L}_{\text{dPPGP}}}_{\text{Calibrated GP}} + \lambda_2 \underbrace{\mathcal{L}_{\text{spec}}}_{\text{Spectral reg.}},
$$

where the dPPGP loss is:

$$
\mathcal{L}_{\text{dPPGP}} = \frac{1}{b}\sum_{(o,y)} -\log \mathcal{N}\!\left(y;\, \hat{\mu}_f(o),\, \hat{\sigma}_f^2(o) + \sigma_\epsilon^2\right) + \alpha\, \mathcal{L}_{\text{trace}} + \frac{\beta}{n}\, D_{\text{KL}}\!\left(\mathcal{N}(m, LL^\top) \,\|\, \mathcal{N}(0, I_r)\right),
$$

with trace regularizer:

$$
\mathcal{L}_{\text{trace}} = \frac{1}{b}\sum_{o} \frac{\tilde{k}_b - \|\phi_\theta(o)\|^2}{2\sigma_\epsilon^2}, \quad \tilde{k}_b = \max_{o \in \text{batch}} \|\phi_\theta(o)\|^2.
$$

where $\mathcal L_{\text{spec}}$ is:
$$
\mathcal L_{\text{spec}} \;=\; \big(\max(0,\ \rho(\hat A)-\rho_{\max})\big)^2,\qquad \rho(\hat A)=\text{spectral radius}.
$$
For LST (near-conservative thermal field) set $\rho_{\max}=1$ to allow persistent diurnal/seasonal modes without exponential blow-up; for strictly dissipative settings set $\rho_{\max}<1$.

**Remark 3.1 (Why dPPGP over GP Log-Marginal Likelihood).** Theorem 2 of Zhu et al. (2026) shows that naive MML training of expressive low-rank kernels leads to **rank-1 degeneracy**: the optimal kernel collapses to $K^*_{XX} = f_{\text{gt}} f_{\text{gt}}^\top$ with $\sigma_\epsilon^{*2} = \sigma_{\text{gt}}^2$, collapsing the entire function space onto a single direction. The dPPGP objective avoids this by: (i) including $\hat{\sigma}_f^2$ in the predictive likelihood (forcing the model to learn input-dependent uncertainty), and (ii) the trace regularizer encouraging uniform prior variance across inputs, preventing rank collapse.

**Remark 3.2 (Anti-Collapse: dPPGP vs. SIGReg vs. GP Prior).** The three approaches to preventing representation collapse are:

| Method | Anti-collapse mechanism | Hyperparameters | Guarantees |
|---|---|---|---|
| LeWM (SIGReg) | Enforce Gaussian distribution via Epps–Pulley test | 1 ($\lambda$) | Provable (Cramér–Wold) |
| DK-WM v1 (GP prior) | GP log-marginal likelihood penalizes collapsed Gram | 1 ($\lambda_1$) | Principled (Bayesian) |
| **DB-WM v2 (dPPGP)** | Trace regularization + predictive likelihood + KL | 2 ($\alpha, \beta$) | **Principled + calibrated** |

The dPPGP is superior because it directly targets test-time predictive calibration (Theorem 2 and 3 of Zhu et al., 2026), whereas SIGReg and GP priors optimize proxies that may not align with test-time performance.

---

## 4. Theoretical Guarantees

### 4.1 Observability in the DBK Weight Space

The observation model (DB-WM Observation) is a linear system with state $w_t \in \mathbb{R}^r$, transition $\hat{A} \in \mathbb{R}^{r \times r}$, and observation matrix $\Phi_X \in \mathbb{R}^{N \times r}$. All kernel observer theorems apply directly.

**Definition 4.1 (DBK Shadedness).** The feature matrix $\Phi_X \in \mathbb{R}^{N \times r}$ is **shaded** if for each column $j \in \{1, \ldots, r\}$, there exists at least one row $i$ such that $(\Phi_X)_{ij} = \phi_j(o_i) \neq 0$. That is, every basis function is "activated" by at least one observation.

**Theorem 4.1 (Observability of DB-WM).** Consider the DB-WM system:
$$
w_{t+1} = \hat{A}\, w_t + \eta_t, \quad y_t = \Phi_{X_t}\, w_t + \zeta_t.
$$
Suppose:
- (A1) $\hat{A} \in \mathbb{R}^{r \times r}$ has a full-rank Jordan decomposition with distinct eigenvalues.
- (A2) $\Phi_X$ is shaded (Definition 4.1).
- (A3) The time instances $\Upsilon = \{\tau_1, \ldots, \tau_L\}$ have $|\Upsilon| \geq r$ distinct values.

Then the system is **observable**: the generalized observability matrix

$$
O_\Upsilon = \begin{bmatrix} \Phi_X \hat{A}^{\tau_1} \\ \vdots \\ \Phi_X \hat{A}^{\tau_L} \end{bmatrix} \in \mathbb{R}^{NL \times r}
$$

has rank $r$, and the state $w_0$ can be uniquely recovered from observations $\{y_{\tau_1}, \ldots, y_{\tau_L}\}$.

*Proof.* Identical to Proposition 1 of the kernel observer paper, with the substitution $K \to \Phi_X$ and $M \to r$. The DBK feature matrix $\Phi_X$ plays the exact role of the kernel observation matrix. The shadedness condition ensures at least one nonzero row via elementary row operations, and the distinct-eigenvalue assumption provides the Vandermonde structure needed for linear independence. $\square$

**Corollary 4.1 (Sensor Lower Bound).** The minimum number of observations required for observability is $\ell$, the cyclic index of $\hat{A} \in \mathbb{R}^{r \times r}$, where $\ell = \max_{1 \leq i \leq s} \text{gm}(\lambda_i)$ and $\text{gm}(\lambda_i)$ is the geometric multiplicity of eigenvalue $\lambda_i$.

*Proof.* Direct application of Proposition 2 from the kernel observer paper to the $r$-dimensional system. $\square$

**Proposition 4.1 (Automatic Shadedness for SwiGLU Expansion).** If the expansion layer uses SwiGLU activation (DB-WM-SwiGLU variant), then $\phi_j(o) = s_j \cdot \sigma_{\text{SwiGLU}}(w_j^\top g_\theta(o) + b_j)$. Since SwiGLU is strictly nonzero almost everywhere ($\sigma_{\text{SwiGLU}}(x) = Swish(xW + b) ⊗ (xV + c) \neq 0$ for $x \neq 0$), the feature matrix $\Phi_X$ is shaded with probability 1 under any continuous input distribution, provided $g_\theta$ is non-degenerate.

*Proof.* For continuous inputs $o$ and a trained backbone $g_\theta$ that is not constant, the pre-activation $w_j^\top g_\theta(o) + b_j$ is a continuous random variable. The set $\{x : \sigma_{\text{SwiGLU}}(x) = 0\} = \{0\}$ has measure zero, so $\phi_j(o) \neq 0$ almost surely. $\square$

### 4.2 Observability Under Known Inputs

**Proposition 4.2 (Observability).** For $w_{t+1}=\hat A w_t+Bu_t^{\text{raw}}+\eta_t,\ y_t=\Phi_X w_t+\zeta_t$ with $\hat A$ having a full-rank Jordan form with distinct eigenvalues and $\Phi_X$ shaded, the pair $(\Phi_X,\hat A)$ is observable; the generalized observability matrix $O_\Upsilon=[(\Phi_X\hat A^{\tau_1})^\top\cdots]^\top$ has rank $r$ for $|\Upsilon|\ge r$. *Crucially, observability depends on $(\Phi_X,\hat A)$ only — the input term $Bu_t^{\text{raw}}$ is a known signal and does not affect observability.*

### 4.3 Controllability

**Proposition 4.3 (DB-WM Controllability).** The system

$$
w_{t+1} = \hat{A}\, w_t + B_u\, u_t + \eta_t
$$

is **controllable** if the controllability Gramian

$$
W_c(0,T)=\sum_{k=0}^{T-1}\hat A^{k}B_uB_u^\top(\hat A^k)^\top\succ0.
$$

Again, all operations are O(r²) per step. For scalar irrigation this is a rank-1-per-step accumulation; full rank requires $T\ge r$ and $B_u$ not orthogonal to any left-eigenspace of $\hat A$ — checkable numerically post-identification. Precipitation, being uncontrollable, is correctly excluded from the controllability analysis (it is a disturbance; in $W_c$ it would be meaningless).

### 4.4 Multi-Step Prediction Error Bounds

**Theorem 4.2 (Open-Loop Prediction Error).** Let $\rho = \|\hat{A}\|_2$ be the spectral norm. The $T$-step open-loop prediction error satisfies:

$$
\|\hat{w}_T - w_T^*\| \leq \rho^T\, \varepsilon_{\text{enc}} + \frac{\rho^T - 1}{\rho - 1}\, \varepsilon_{\text{dyn}}\quad(\rho\neq 1),\qquad =\ \varepsilon_{\text{enc}} + T\,\varepsilon_{\text{dyn}}\quad(\rho=1),
$$

where $\varepsilon_{\text{enc}} = \sup_t \|w_t^{\text{encoded}} - w_t^{\text{true}}\|$ and $\varepsilon_{\text{dyn}} = \sup_t \|w_{t+1}^* - \hat{A}\, w_t^* - B_p\, p_t - B_u\, u_t\|$ is the one-step dynamics residual under the *known* forcing. For near-conservative fields (e.g., LST) the design choice $\rho_{\max}=1$ gives $\rho\approx 1$, so open-loop error grows *linearly* in the horizon $T$ — the honest characterization of a multi-day forecast, rather than the optimistic geometric decay that $\rho<1$ would imply.

**Theorem 4.3 (Observer-Corrected Bound, Kalman Filter).** When pixel observations $o_t$ are available, the visual kernel observer (Kalman filter on $\mathbb{R}^r$) provides state estimates with steady-state error covariance $P_\infty$ satisfying the **discrete algebraic Riccati equation** in $\mathbb{R}^{r \times r}$:

$$
P_\infty = \hat{A}\, P_\infty\, \hat{A}^\top + Q - \hat{A}\, P_\infty\, \Phi_X^\top (\Phi_X P_\infty \Phi_X^\top + \sigma_\epsilon^2 I_N)^{-1} \Phi_X\, P_\infty\, \hat{A}^\top.
$$

If $(\Phi_X, \hat{A})$ is observable and $(Q^{1/2}, \hat{A})$ is stabilizable, then $P_\infty$ exists, is unique, and provides bounded estimation error **independent of the prediction horizon**. Because the dynamics are input-affine, the known forcing term $B_p p_t + B_u u_t$ shifts the state *mean* through the predict step but leaves the error covariance recursion — and hence $P_\infty$ — unchanged.

The Riccati equation is $r \times r$, so solving it costs O(r³) — negligible compared to the O(M³) of the original framework.

### 4.5 Koopman Operator Connection

**Theorem 4.4 (Least-Squares Koopman Approximation in the Learned Basis).** The transition matrix $\hat{A}$ obtained by minimizing the dynamics prediction loss is precisely the empirical **finite-section (Galerkin) approximation** of the Koopman operator restricted to the span of the learned basis functions $\{\phi_i \circ g_\theta\}_{i=1}^r$ — i.e., the least-squares fit:

$$
\hat{A} = \underset{A \in \mathbb{R}^{r \times r}}{\arg\min} \sum_t \|\phi_\theta(o_{t+1}) - A\, \phi_\theta(o_t)\|^2 = \left(\sum_t \phi_\theta(o_t)\, \phi_\theta(o_t)^\top\right)^{-1} \left(\sum_t \phi_\theta(o_t)\, \phi_\theta(o_{t+1})^\top\right).
$$

This is the least-squares solution in $\mathbb{R}^{r \times r}$, computable in O(Tr²) time. It is the estimator that converges to the Galerkin projection of the Koopman operator onto the learned dictionary as $T\to\infty$ (Korda & Mezić, 2018).

*Proof.* Setting the gradient of $\sum_t \|\phi_\theta(o_{t+1}) - A \phi_\theta(o_t)\|^2$ with respect to $A$ to zero yields $A \sum_t \phi_\theta(o_t) \phi_\theta(o_t)^\top = \sum_t \phi_\theta(o_{t+1}) \phi_\theta(o_t)^\top$, which is the finite-section Koopman regression in the learned basis. $\square$

**Proposition 4.4 (Redundancy of the Truncation Step: Least Squares Equals the Reduced-Order Operator).** Let $W_-=[w_1,\dots,w_{T-1}]$, $W_+=[w_2,\dots,w_T]\in\mathbb{R}^{r\times(T-1)}$ be the weight snapshots in the learned basis, and let $\hat A = W_+W_-^\top(W_-W_-^\top+\mu I_r)^{-1}$ be the regularized least-squares operator ($\mu\to0^+$ gives the exact fit). Then:

1. **(Redundant reduction)** Let $W_-=U\Sigma V^\top$ be a thin SVD with $U\in\mathbb{R}^{r\times\rho}$, $\rho=\operatorname{rank}(W_-)\le r$. The operator a projected DMD would fit in the orthonormal coordinates $\tilde w=U^\top w$ is $\tilde A = U^\top W_+ V\Sigma^{-1}$. With $\mu=0$, $\hat A$ and $\tilde A$ have **identical nonzero eigenvalues and corresponding modes**; when $\rho=r$ they are related by the orthonormal similarity $\tilde A = U^\top \hat A\, U$. Thus the SVD only re-expresses the already-$r$-dimensional weights in an orthonormal frame and discards no coordinate.
2. **(Direct computability)** $\hat A$ is obtained from the $r\times r$ normal equations in $O(Tr^2+r^3)$ without ever forming or reducing any $n$-dimensional snapshot.

Hence, because $\phi_\theta$ already supplies the reduction to $\mathbb{R}^r$, the truncating SVD/POD stage of a raw-space DMD is **redundant**; the reduced-order (economized) operator is recovered exactly by matrix least squares.

*Proof.* The normal equations of $\min_A \|W_+ - AW_-\|_F^2 + \mu\|A\|_F^2$ give $\hat A(W_-W_-^\top+\mu I_r)=W_+W_-^\top$, hence the closed form in (2); cost is dominated by $W_-W_-^\top$ ($O(Tr^2)$) and its $r\times r$ inverse ($O(r^3)$). For (1), take $\mu=0$, so $\hat A = W_+W_-^{+} = W_+ V\Sigma^{-1}U^\top$ (the minimum-norm least-squares / exact-DMD form; Tu et al., 2014). Factor $\hat A = BC$ with $B=W_+V\Sigma^{-1}\in\mathbb{R}^{r\times\rho}$ and $C=U^\top\in\mathbb{R}^{\rho\times r}$; then the reduced operator is $\tilde A = CB = U^\top W_+ V\Sigma^{-1}$. Because the nonzero spectra of $BC$ and $CB$ coincide, $\hat A$ and $\tilde A$ share all nonzero eigenvalues (and $\hat A$'s remaining eigenvalues are zero, corresponding to weight directions unexcited by the data, i.e. $\ker W_-^\top$). If $\rho=r$, then $U$ is orthogonal, $\hat A\,U = W_+V\Sigma^{-1}$, and $U^\top\hat A\,U = U^\top W_+ V\Sigma^{-1}=\tilde A$ — an orthonormal similarity. In either case the SVD contributes only the orthonormal frame $U$ within $\mathbb{R}^r$ and no dimensional truncation. $\square$

**Remark 4.1 (Proof Invariance Under the Identification Method).** Every guarantee in Section 4 — observability (Theorem 4.1, Proposition 4.2), the sensor lower bound (Corollary 4.1), automatic shadedness (Proposition 4.1), controllability (Proposition 4.3), the open-loop prediction bound (Theorem 4.2), the Kalman observer bound (Theorem 4.3), the Koopman-mode extraction (Corollary 4.2), and the Koopman-with-inputs identity (Corollary 4.3) — is stated entirely in terms of the operator $\hat A\in\mathbb{R}^{r\times r}$, the feature/observation matrix $\Phi_X$, the input matrix $B$, and the noise covariances. **None of their hypotheses or proofs references the procedure used to obtain $\hat A$.** Consequently, replacing any Dynamic-Mode-Decomposition description of the identification with the equivalent matrix-least-squares fit (Proposition 4.4) leaves every theorem, corollary, and proof unchanged. The identification method affects only *which* $\hat A$ is estimated from finite data — a statistical/consistency question governed by persistence of excitation (Remark 6.1) — not the validity of the structural guarantees that hold for the resulting operator.

**Corollary 4.2 (Koopman Mode Extraction).** The eigendecomposition $\hat{A} = P \Lambda P^{-1}$ yields Koopman eigenvalues $\{\lambda_j\}_{j=1}^r$ and modes $s_j(o) = \sum_k p_{j,k}\, \phi_k(o)$, with frequencies $f_j = |\text{Im}(\log \lambda_j)| / (2\pi \Delta t)$ and growth rates $\gamma_j = \text{Re}(\log \lambda_j) / \Delta t$. Cost: O(r³) for the eigendecomposition.

**Corollary 4.3 (Koopman with Inputs via Least-Squares Control Identification).** When forcing is present, the joint least-squares solution $[\hat{A}\ B] = W_+ \Omega^\top (\Omega \Omega^\top + \mu I)^{-1}$ (Algorithm 3, with $\Omega = [W_-^\top\ \Upsilon^\top]^\top$) is the empirical finite-section (Galerkin) approximation of the **Koopman operator with inputs** (Proctor, Brunton & Kutz, 2018, KIC), restricted to the span of the learned basis $\{\phi_i \circ g_\theta\}$. Because the basis map has already lifted the observations into $\mathbb{R}^r$, this finite-section estimate is obtained *directly* from the normal equations in the weight space — no separate spectral/SVD reduction of the snapshots is required (Proposition 4.4). Theorem 4.4 is the autonomous ($\ell=0$) special case. The eigenpairs of $\hat{A}$ are the input-*decoupled* Koopman modes of the autonomous dynamics — well-defined precisely because $\hat{A}$ is input-independent. This is exactly the property the rejected $\hat A(a_t)$ form destroys (no single $\hat A$, hence no single spectrum; cf. Remark 2.1).

*Proof.* The block least-squares fit of $[\hat A\ B]$ against the augmented regressor $\Omega=[W_-^\top\ \Upsilon^\top]^\top$ is the finite-section of the Koopman generator on the input-augmented state, by the same Galerkin argument as Theorem 4.4 applied to the extended observable space (Proctor, Brunton & Kutz, 2018). Setting $\ell=0$ (empty $\Upsilon$) recovers exactly the normal equations of Theorem 4.4. Input-decoupling of $\hat A$ follows from the block structure: the $(\hat A)$-block of the solution regresses $W_+$ onto the state channel of $\Omega$ after accounting for the input channel, yielding a single input-independent operator whose eigenpairs are therefore well-defined. $\square$

---

## 5. Resolution of E-GP Limitations (Updated)

| # | Limitation | Resolution in DB-WM v2 | Complexity |
|---|---|---|---|
| 1 | Linear dynamics | Koopman lifting via learned deep basis (Theorem 4.4) | O(Tr²) |
| 2 | Full sensor coverage | Visual encoder provides virtual measurements | O(br²) per batch |
| 3 | Approximation bias | Mercer truncation bound (Theorem 2.1) + dPPGP calibration | Controlled |
| 4 | Abstract measurement map | Encoder $\phi_\theta$ is the measurement map by construction | Eliminated |
| 5 | Stringent observability | Automatic shadedness for SwiGLU (Prop. 4.1) | Guaranteed |
| 6 | Frequency resolution | Spectral mixture expansion or regime conditioning | O(Qr²) |
| 7 | Error accumulation | Kalman filter in $\mathbb{R}^r$ with O(r²) updates (Theorem 4.3) | **O(r²)** per step |
| 8 | Kernel stationarity | Deep basis is automatically nonstationary (Prop. 5.5 of v1) | Intrinsic |
| **9** | **O(M³) scalability** | **DBK low-rank structure: O(nr²) time, O(nr) space** | **Resolved** |

---

## 6. Algorithms

### Algorithm 1: DB-WM Training (dPPGP + Dynamics)

```
Input: Trajectories {(o_1^(i), u_1^(i,raw), ..., o_T^(i))}_{i=1}^B   // u_t^raw = [p_t, u_t]^T ∈ R^ℓ
       Deep basis map φ_θ: O → R^r (backbone g_θ + expansion layer)
       Transition matrix Â ∈ R^{r×r}, input matrix B = [B_p B_u] ∈ R^{r×ℓ}
       Variational params (m ∈ R^r, L ∈ R^{r×r} lower triangular)
       Hyperparams: α (trace reg.), β (KL reg.), λ_1 (GP weight), λ_2 (dynamics weight)

Initialize:
  φ_θ ← random init (or pretrained backbone)
  Â ← identity + small perturbation;  B ← 0 (zero-init forcing directions)
  m ← 0, L ← (1/√r) · I_r

For each training iteration:
  1. Sample mini-batch of b trajectory segments of length T

  2. Compute basis features:
     For each (i, t): φ_t^(i) = φ_θ(o_t^(i)) ∈ R^r       // O(b·T·c_φ)

  3. Compute dynamics loss:
     For each (i, t):
       w_t^(i) = φ_t^(i)                                    // Direct basis weights
       ŵ_{t+1}^(i) = Â · w_t^(i) + B · u_t^(i,raw)          // O(r²) per step (input-affine)
     L_dyn = (1/bT) Σ_{i,t} ||w_{t+1}^(i) - ŵ_{t+1}^(i)||²

  4. Compute dPPGP loss:
     μ̂_f(o) = ⟨m, φ_θ(o)⟩                                  // O(r)
     σ̂²_f(o) = ||L^T φ_θ(o)||²                              // O(r²)
     k̃_b = max_{o in batch} ||φ_θ(o)||²
     L_trace = (1/b) Σ_o (k̃_b - ||φ_θ(o)||²) / (2σ²_ε)
     L_dPPGP = (1/b) Σ_{(o,y)} -log N(y; μ̂_f, σ̂²_f + σ²_ε)
               + α·L_trace + (β/n)·D_KL(N(m,LL^T) || N(0,I_r))

  5. Total loss:
     L = λ_2 · L_dyn + λ_1 · L_dPPGP

  6. Backpropagate and update all parameters
     // Total per-iteration cost: O(bT(r² + c_φ))

Output: Trained DB-WM (φ_θ, Â, B_p, B_u, m, L, σ²_ε)
```

### Algorithm 2: Visual Basis Observer (Kalman Filter in R^r)

```
Input: Trained DB-WM, observations {o_t}, known inputs {u_t^raw = [p_t, u_t]^T}
       Process-noise covariance Q ∈ R^{r×r}, input matrix B = [B_p B_u] ∈ R^{r×ℓ},
       measurement noise σ²_ε,  (optional) model-error inflation Γ_dyn

Initialize:
  w_{0|0} = φ_θ(o_0)          // Encode initial state: O(c_φ)
  P_{0|0} = σ²_ε · I_r        // Initial covariance: O(r²)

For each time step t = 1, 2, ...:
  === PREDICT ===   // O(r²)
  w_{t|t-1} = Â · w_{t-1|t-1} + B · u_{t-1}^raw          // known forcing enters the predict mean
  P_{t|t-1} = Â · P_{t-1|t-1} · Â^T + Q + Γ_dyn          // + optional dynamics-residual inflation

  === UPDATE (if observation o_t available) ===   // O(r²)
  φ_t = φ_θ(o_t) ∈ R^r                          // Basis features
  y_t = φ_t                                       // "Measurement" = encoded observation

  Innovation: ν_t = y_t - w_{t|t-1}              // forcing already subtracted via predict mean
  Innovation covariance: S_t = P_{t|t-1} + σ²_ε · I_r    // R^{r×r}
  Kalman gain: L_t = P_{t|t-1} · S_t^{-1}                // O(r³)

  w_{t|t} = w_{t|t-1} + L_t · ν_t                        // Corrected state
  P_{t|t} = (I_r - L_t) · P_{t|t-1}                      // Corrected covariance

  === PLAN (optional; only if irrigation u is *chosen* rather than observed) ===
  w_g = φ_θ(o_g)
  Admissible set (mutual exclusivity as a continuous box constraint):
    U_{t+k} = {0}        if p_{t+k} > 0   (rain forecast: no irrigation)
            = [0, u_max] if p_{t+k} = 0
  Optimize u_{t:t+H} via CEM (clip samples to U_{t+k}):
    Roll out: ŵ_{t+k+1} = Â · ŵ_{t+k} + B_p · p_{t+k} + B_u · u_{t+k}   // p known forecast; u decision
    Cost: C = Σ_k γ^k [ ||ŵ_{t+k} - w_g||² + β_plan · tr(P_{t+k}) + λ_u · u_{t+k}² ]
  Execute first K irrigation actions, then replan (cadence K).

Output: State estimates {w_t}, covariances {P_t}, controls {u_t}
  // Total per-step cost: O(r³ + c_φ) — dominated by S_t^{-1} in R^{r×r}
```

### Algorithm 3: Scalable System Identification (Least-Squares Operator Fitting in the Deep-Basis Weight Space)

The transition operator and forcing directions are identified by **regularized matrix least squares** applied directly in the $r$-dimensional latent weight space — equivalently, **control on the learned reduced-order dynamics model**. Because the deep basis $\phi_\theta$ has *already* performed the Koopman lifting/reduction into $\mathbb{R}^r$ ($r \le 1024$), the additional truncating SVD used by DMD-type reductions is **redundant** (Proposition 4.4): the normal equations are tractable in closed form and return the same reduced-order operator. No pixel-space (double-)SVD is needed. The autonomous least-squares fit of Theorem 4.4 is recovered as the no-forcing special case (and as Stage I below).

```
Input: Encoded weights {w_t = Λ_X^{-1} Φ_X^T y_t}_{t=1}^T,
       raw inputs {u_t^raw = [p_t, u_t]^T ∈ R^ℓ}_{t=1}^{T-1},  ridge μ

1. Form data matrices:
   W_- = [w_1, ..., w_{T-1}] ∈ R^{r×(T-1)}
   W_+ = [w_2, ..., w_T]     ∈ R^{r×(T-1)}
   Υ   = [u_1^raw, ..., u_{T-1}^raw] ∈ R^{ℓ×(T-1)}

--- Regime A: JOINT identification (Â and B both unknown; dense, high-SNR forcing) ---
2A. Stack augmented regressor: Ω = [W_- ; Υ] ∈ R^{(r+ℓ)×(T-1)}
3A. Solve regularized least squares:
    [Â  B] = W_+ · Ω^T · (Ω·Ω^T + μ·I_{r+ℓ})^{-1}      // O(T(r+ℓ)² + (r+ℓ)³)
    where Ω·Ω^T = [[W_-W_-^T, W_-Υ^T],[ΥW_-^T, ΥΥ^T]]
    // This is the finite-section Koopman-with-inputs fit (Corollary 4.3):
    // one r×r block for the autonomous operator, one r×ℓ block for control.

--- Regime B: TWO-STAGE identification (DEFAULT; sparse forcing, statistically cleaner) ---
2B. Stage I — autonomous operator from QUIESCENT transitions (p_t = u_t = 0):
      Â = W_+^0 · (W_-^0)^T · (W_-^0 (W_-^0)^T + μ·I_r)^{-1}   // = autonomous LS fit (Theorem 4.4)
3B. Stage II — forcing directions from residuals on ALL transitions:
      ΔW = [Δw_1, ..., Δw_{T-1}],   Δw_t = w_{t+1} - Â·w_t
      [B_p  B_u] = ΔW · Υ^T · (Υ·Υ^T + μ·I_ℓ)^{-1} ∈ R^{r×ℓ}   // O(Tr²)

4. Optional spectral safety: clip eigenvalues of Â to ρ_max
   (Â = P·diag(clip(λ_i, |λ_i| ≤ ρ_max))·P^{-1}; ρ_max = 1 for near-conservative fields)

5. Process-noise covariance (closes the Kalman loop; residual, not regressor):
   η̂_t = w_{t+1} - Â·w_t - B_p·p_t - B_u·u_t
   Q̂ = (1/(T-2)) Σ_t η̂_t η̂_t^T + ν·I_r     // small ν > 0 for conditioning

Output: Â ∈ R^{r×r}, B = [B_p B_u] ∈ R^{r×ℓ}, Q̂ ∈ R^{r×r}
   // Total cost: O(T(r+ℓ)² + (r+ℓ)³) — linear in trajectory length, closed-form
```

**Remark 6.1 (Why least-squares control identification, two-stage by default).** The least-squares control identification *disambiguates* the autonomous dynamics from the forcing — the single property that makes $\hat A$'s spectrum (Corollary 4.2) physically meaningful and that the rejected $\hat A(a_t)$ form forfeits (Remark 2.1). Identifiability requires **persistence of excitation**, $\operatorname{rank}[W_-^\top\ \Upsilon^\top]^\top = r + \ell$; the dangerous failure mode is forcing collinear with an autonomous mode (e.g., monsoon rain aligned with the seasonal swing), which makes $\hat A$ and $B_p$ inseparable. Two-stage identification sidesteps this by fitting $\hat A$ on quiescent transitions only, then $B$ on the residuals — the recommended default whenever forcing is sparse (as in agricultural LST). Joint identification is preferred only when the forcing is dense and high-SNR. Identification is closed-form (no SGD): in a two-phase schedule, $\phi_\theta$ is trained first via dPPGP, then frozen and the dates encoded before this algorithm runs; in an end-to-end schedule, $\hat A, B_p, B_u$ instead become differentiable parameters updated by the dynamics loss of Definition 3.1.

---

## 7. The Complete Structural Isomorphism

### 7.1 The World Model Interpretation

The DB-WM is a complete World Model in the JEPA sense with GP-theoretic guarantees:

1. **Encoder**: $\phi_\theta(o_t)$ maps pixels to $r$-dimensional basis weights (analogous to LeWM's ViT encoder mapping pixels to $d$-dimensional latent codes).

2. **Predictor**: the input-affine map $w_{t+1} = \hat{A}\, w_t + B\, u_t^{\text{raw}}$ with fixed $\hat{A} \in \mathbb{R}^{r \times r}$ and input matrix $B = [B_p\ B_u]$ predicts next-step weights (analogous to LeWM's transformer predictor, but with forcing entering additively per the input-affine model rather than reparameterizing the operator — Remark 2.1). The linearity is justified by Koopman theory (Theorem 4.4, Corollary 4.3).

3. **Anti-collapse**: dPPGP's trace regularization and KL divergence (analogous to LeWM's SIGReg, but with stronger calibration guarantees per Theorem 2 of Zhu et al.).

4. **Planning**: CEM in $\mathbb{R}^r$ with uncertainty-aware cost function (analogous to LeWM's CEM in latent space, but with calibrated predictive intervals).

5. **Unique to DB-WM**: Formal observability (Theorem 4.1), controllability (Proposition 4.3), bounded observer error (Theorem 4.3), Koopman mode interpretability (Corollary 4.2), and calibrated uncertainty quantification (dPPGP).

---

## 8. Application-Specific Considerations

### 8.1 Satellite Imagery (LST/NDVI/Surface Reflectance)

For study area (~4,500 km², UTM Zone 43N) with multi-temporal Landsat/MODIS data:

- **Input**: $o_t \in \mathbb{R}^{C \times H \times W}$ (multi-band satellite image at time $t$)
- **Backbone**: CNN or ViT processing the spatial-spectral structure, $g_\theta: \mathbb{R}^{C \times H \times W} \to \mathbb{R}^h$ with $h = 256$
- **Expansion**: SwiGLU or RBF, producing $\phi_\theta(o_t) \in \mathbb{R}^r$ with $r = 512$–$1024$
- **Dynamics**: $w_{t+1} = \hat{A}\, w_t + B_p\, p_t + B_u\, u_t$ — fixed autonomous thermal operator $\hat{A}$ (diurnal/seasonal Koopman modes), precipitation $p_t$ as a *known exogenous disturbance* (enters the Kalman predict step, not a decision variable), and irrigation $u_t$ as an optional controllable actuator; daily/weekly/monthly time steps. Set $B_p = B_u = 0$ to recover the pure-temporal special case.
- **Prediction**: "What will the LST field be at $t+1$?" → $f_{t+1}(x) = \langle \hat{A} w_t + B_p p_t + B_u u_t,\ \phi_\theta(x) \rangle$, using the known precipitation forecast $p_t$.
- **Uncertainty**: Predictive variance $\hat{\sigma}_f^2(o^*) = \|L^\top \phi_\theta(o^*)\|^2$ provides calibrated confidence intervals for each pixel
- **Scalability**: With $n = 100{,}000$ pixels and $r = 512$: O(nr²) ≈ O(100K × 260K) ≈ 2.6 × 10¹⁰ FLOPs for full inference — feasible on a single GPU

### 8.2 Fluid Dynamics (Vortex Shedding)

For cylinder flow at Re = 100–1000:

- **Input**: $o_t \in \mathbb{R}^{1 \times H \times W}$ (velocity field snapshot as image)
- **Backbone**: CNN, $g_\theta: \mathbb{R}^{H \times W} \to \mathbb{R}^{256}$
- **Expansion**: RBF inducing points in latent space, $r = 600$ (matching E-GP's 600 kernel centers)
- **Dynamics**: a *separate fixed* operator $\hat{A}_{\text{Re}}$ identified per Reynolds regime — legitimate indexing by a constant physical parameter held fixed within each trajectory, which is *not* the rejected per-step action conditioning $\hat A(a_t)$ (Remark 2.1): each $\hat{A}_{\text{Re}}$ retains a well-defined Koopman spectrum.
- **Koopman analysis**: Extract vortex shedding frequencies from eigenvalues of $\hat{A}$ (Corollary 4.2), and compare against the spectral (Koopman) modes reported by E-GP.

### 8.3 Robotic Manipulation (2D/3D Control)

- **Input**: $o_t \in \mathbb{R}^{3 \times 224 \times 224}$ (RGB camera image)
- **Backbone**: ViT-Tiny (~5M params), $g_\theta: \mathbb{R}^{3 \times 224 \times 224} \to \mathbb{R}^{192}$
- **Expansion**: SwiGLU, $r = 128$
- **Dynamics**: $w_{t+1} = \hat{A}\, w_t + B\, a_t$ — input-affine control with fixed $\hat{A} \in \mathbb{R}^{r \times r}$ and learned input matrix $B \in \mathbb{R}^{r \times \ell}$ for the $\ell$-dimensional action $a_t$, identified by matrix least squares (Algorithm 3). Here the action is a genuine per-step decision variable, so it enters additively through $B$ — never by reparameterizing $\hat{A}$ (Remark 2.1).
- **Planning**: CEM with 300 candidates, 30 iterations → O(300 × 30 × 128²) ≈ 1.5 × 10⁸ FLOPs per planning step (~1ms on GPU)

---

## 9. Conclusion

The Deep Basis World Model (DB-WM) resolves the final critical limitation of the DK-WM v1 framework: the O(M³) computational bottleneck from kernel Gram matrix inversion. By adopting the Deep Basis Kernel (DBK) construction of Zhu et al. (2026), where the kernel is defined as an inner product of neural basis functions $k(o, o') = \langle \phi_\theta(o), \phi_\theta(o') \rangle$, all operations reduce to O(nr²) time and O(nr) space via the Woodbury identity.

The key theoretical results are preserved and strengthened:

- **Observability** (Theorem 4.1): Transfers directly with the DBK feature matrix $\Phi_X$ replacing the kernel observation matrix $K$, and the transition matrix $\hat{A}$ now in $\mathbb{R}^{r \times r}$ instead of $\mathbb{R}^{M \times M}$.
- **Controllability** (Proposition 4.3): Gramian computation is O(r²) per step.
- **Koopman approximation** (Theorem 4.4, Corollary 4.3): least-squares finite-section (Galerkin) Koopman regression in the learned basis for the autonomous case, extended to the **Koopman-with-inputs** least-squares fit (control on the learned reduced-order dynamics model) when forcing is present, cost O(T(r+ℓ)²).
- **Identification & control** (Algorithm 3, Remark 2.1): dynamics are **input-affine** $w_{t+1} = \hat{A} w_t + B u_t^{\text{raw}}$, never action-conditioned; $[\hat{A}\ B]$ is identified in closed form by regularized matrix least squares, cleanly separating the fixed autonomous operator (with a well-defined Koopman spectrum) from controllable actuation and *uncontrollable exogenous disturbances*.
- **Proof invariance** (Remark 4.1): all structural guarantees depend only on the resulting operator $\hat A$, $\Phi_X$, and $B$ — not on the identification procedure — so describing the identification as matrix least squares rather than DMD changes no theorem and no proof.
- **Observer error bound** (Theorem 4.3): Kalman filter operates in $\mathbb{R}^r$, all updates O(r²); known forcing shifts the mean but not the steady-state covariance.
- **Anti-collapse** (dPPGP): Principled calibration with trace regularization, addressing the rank-1 degeneracy identified in Theorem 2 of the DBK paper.
- **Multi-dimensional inputs**: Handled by the backbone architecture ($d$-agnostic), with the expansion layer always producing $r$-dimensional bases.

The framework now scales to the data regimes required for satellite remote sensing (100K+ pixels), fluid dynamics (95K velocity points), and real-time robotic control (sub-second planning), while retaining the formal guarantees from kernel observer theory that distinguish it from purely neural World Models.

---

## References

- Zhu, Y., Yuchi, H. S., & Xie, Y. (2026). Scalable Deep Basis Kernel Gaussian Processes. arXiv:2505.18526v2.
- Williams, M. O., Kevrekidis, I. G., & Rowley, C. W. (2015). A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition. *Journal of Nonlinear Science*, 25(6), 1307–1346. *(Least-squares Koopman regression in a dictionary — the finite-section estimator used here in the learned basis.)*
- Korda, M., & Mezić, I. (2018). On Convergence of the Empirical Koopman Operator Approximation to the Koopman Operator. *Journal of Nonlinear Science*, 28, 687–710. *(Convergence of the least-squares dictionary fit to the Galerkin/finite-section projection of the Koopman operator.)*
- Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2018). Generalizing Koopman Theory to Allow for Inputs and Control (KIC). SIAM J. Applied Dynamical Systems. *(Koopman operator with inputs — the theory underlying the least-squares $[\hat A\ B]$ fit, Corollary 4.3.)*
- Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2016). Dynamic Mode Decomposition with Control. SIAM J. Applied Dynamical Systems, 15(1), 142–161. *(Origin of the input/autonomous-dynamics disambiguation principle; the linear-observable special case of the least-squares control fit.)*
- Tu, J. H., Rowley, C. W., Luchtenburg, D. M., Brunton, S. L., & Kutz, J. N. (2014). On Dynamic Mode Decomposition: Theory and Applications. *Journal of Computational Dynamics*, 1(2), 391–421. *(Exact / minimum-norm least-squares operator $W_+W_-^{+}$ used in Proposition 4.4.)*
- Maes, L., Le Lidec, Q., et al. (2026). LeWorldModel: Stable End-to-End JEPA from Pixels. arXiv:2603.19312.
- The Kernel Observers / E-GP paper (2021). IEEE Control Systems Magazine.
- Wilson, A. G., et al. (2016). Deep Kernel Learning. AISTATS.
- Hensman, J., et al. (2013). Gaussian Processes for Big Data. UAI.
- Jankowiak, M., et al. (2020). Parametric Gaussian Process Regressors. ICML.
- Rasmussen, C. E. & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.
- Anderson, B. D. O. & Moore, J. B. (1979). Optimal Filtering. Prentice Hall.
