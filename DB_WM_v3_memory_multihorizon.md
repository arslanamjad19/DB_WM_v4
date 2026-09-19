# DB-WM v3 — Section 2 Revised: Memory-Augmented, Multi-Horizon Deep Basis World Model (DB-WM-MH)

*Drop-in replacement for §2.2–2.5 of `DB_WM_v2_framework.md`, plus the induced revisions to §4.1–4.4. Numbering follows the v2 document; new items are marked ★.*

---

## 2.2 Weight-Space Dynamics with Memory

### 2.2.0 ★ Why memory is not an ad hoc extension: the Mori–Zwanzig argument

The v2 model asserts a Markov transition on the deep-basis weights,
$w_{t+1} = \hat A w_t + Bu_t^{\text{raw}} + \eta_t$ with $\eta_t$ white. This is exactly the
finite-section (Galerkin) Koopman approximation on the learned dictionary
$\mathcal D_r = \operatorname{span}\{\phi_i\circ g_\theta\}_{i=1}^r$ (v2 Thm 4.4). The Mori–Zwanzig (MZ)
formalism states what the *residual* of that projection must look like.

**Proposition 2.5 ★ (MZ structure of the finite-section residual).** Let $\mathcal K$ be the Koopman
operator of the underlying dynamics on a Hilbert space $\mathcal H$ of observables, let
$P:\mathcal H\to\mathcal D_r$ be the orthogonal projection onto the learned dictionary, and let $Q=I-P$.
Then the exact evolution of the projected observables obeys the discrete Dyson / MZ identity

$$
P\mathcal K^{t+1}g \;=\; \underbrace{P\mathcal K P\,\big(P\mathcal K^{t}g\big)}_{\text{Markov term}}
\;+\; \underbrace{\sum_{j=1}^{t} P\mathcal K Q\,(Q\mathcal K Q)^{\,j-1} Q\mathcal K\,\big(P\mathcal K^{t-j}g\big)}_{\text{memory term}}
\;+\; \underbrace{P\mathcal K Q (Q\mathcal K Q)^{t} g}_{\text{orthogonal (noise) term}} .
$$

Consequently the *exact* weight dynamics are non-Markovian: writing $A_0 := P\mathcal K P$ restricted to
$\mathcal D_r$ and $A_j := P\mathcal K Q(Q\mathcal K Q)^{j-1}Q\mathcal K$,

$$
w_{t+1} \;=\; A_0 w_t \;+\; \sum_{j\ge 1} A_j\, w_{t-j} \;+\; F_t ,
$$

with $F_t$ orthogonal to $\mathcal D_r$. The v2 model is the $L=1$ truncation, which forces the entire
memory series into the white-noise term $\eta_t$.

**Corollary 2.5.1 ★ (Misspecification certificate).** If the memory kernel $\{A_j\}_{j\ge1}$ is not
identically zero, the v2 innovation sequence is *autocorrelated*. Hence the Kalman filter of v2
Algorithm 2 is not the minimum-variance filter for the process, and $\hat Q$ from v2 Algorithm 3 step 5
absorbs coloured memory into a white covariance, producing systematically **miscalibrated**
predictive intervals.

> **Falsifiable test (run this before anything else).** Fit the v2 model, compute
> $\hat\eta_t = w_{t+1}-\hat Aw_t-Bu_t^{\text{raw}}$, and apply a multivariate Ljung–Box / portmanteau test
> to $\{\hat\eta_t\}$ at lags $1..12$. Rejection ⟹ Prop. 2.5 bites and the whole memory extension is
> *empirically motivated*, not merely postulated. Non-rejection ⟹ $L=1$ suffices and the extension is
> pure variance inflation. This is the pivotal experiment of the paper.

*Remark.* Under invertible dynamics, delay observables $\phi_i\circ F^{-j}$ are themselves legitimate
Koopman observables, so lifting the dictionary from $\{\phi_i\}$ to $\{\phi_i\circ F^{-j}\}_{j=0}^{L-1}$ is a
**Krylov/Hankel enrichment of the Koopman dictionary** (Hankel-DMD, HAVOK). Memory augmentation is
therefore *internal* to the Koopman framing of v2, not an escape from it.

---

### 2.2.1 ★ The lifted weight state

**Definition 2.3 ★ (Memory-lifted weight state).** For memory order $L\ge1$ define

$$
\bar w_t \;=\; \big[\,w_t^\top,\; w_{t-1}^\top,\;\dots,\;w_{t-L+1}^\top\,\big]^\top \in \mathbb R^{Lr}.
$$

**Definition 2.4 ★ (Memory-augmented input-affine dynamics).**

$$
\boxed{\;w_{t+1}\;=\;\sum_{j=0}^{L-1} A_j\, w_{t-j} \;+\; B_p\,p_t \;+\; B_u\,u_t \;+\;\eta_t\;}
\tag{DB-WM-M}
$$

equivalently, in companion (Markov) realization on $\mathbb R^{Lr}$,

$$
\bar w_{t+1} \;=\; \mathcal A\,\bar w_t \;+\; \mathcal B\, u_t^{\text{raw}} \;+\; \mathcal E\,\eta_t ,
$$

$$
\mathcal A=\begin{bmatrix}
A_0 & A_1 & \cdots & A_{L-2} & A_{L-1}\\
I_r & 0 & \cdots & 0 & 0\\
0 & I_r & \cdots & 0 & 0\\
\vdots & & \ddots & & \vdots\\
0 & 0 & \cdots & I_r & 0
\end{bmatrix}\in\mathbb R^{Lr\times Lr},\quad
\mathcal B=\begin{bmatrix}B\\0\\\vdots\\0\end{bmatrix},\quad
\mathcal E=\begin{bmatrix}I_r\\0\\\vdots\\0\end{bmatrix}.
$$

Write $\mathcal S=[\,I_r\ 0\ \cdots\ 0\,]\in\mathbb R^{r\times Lr}$ for the current-block selector, so
$w_t=\mathcal S\bar w_t$. The v2 model is recovered exactly at $L=1$.

Remark 2.1 of v2 (input-affine forcing, never an action-conditioned operator) carries over verbatim:
$\mathcal A$ is input-independent, hence retains a single well-defined spectrum.

---

### 2.2.2 ★ Structural consequences of the lift

Four results are needed because **the lift does not leave §4 untouched.**

**Lemma 2.6 ★ (Spectrum of the lift).** $\lambda\in\sigma(\mathcal A)$ iff
$\det\!\big(\lambda^L I_r-\sum_{j=0}^{L-1}\lambda^{L-1-j}A_j\big)=0$, and the associated eigenvector is
$\xi=[\lambda^{L-1}v^\top,\lambda^{L-2}v^\top,\dots,v^\top]^\top$ where $v$ spans the corresponding null space.

*Proof.* The shift rows give $\xi_{i-1}=\lambda\xi_i$, hence $\xi_i=\lambda^{L-i}v$ with $v:=\xi_L$;
substituting into the top block row gives the matrix polynomial. $\square$

**Consequence.** The v2 spectral regularizer $\mathcal L_{\text{spec}}=(\max(0,\rho(\hat A)-\rho_{\max}))^2$ and the
eigenvalue clipping of Algorithm 3 step 4 must act on the **roots of the matrix polynomial**, not on
$\sigma(A_0)$. Clipping $\sigma(\mathcal A)$ directly destroys the companion structure; use instead the
sufficient condition $\sum_{j=0}^{L-1}\|A_j\|_2\le\rho_{\max}$, or a stable-by-construction
parameterization (§2.2.4).

**Lemma 2.7 ★ (Process-noise reachability is automatic).** $(\mathcal A,\mathcal E)$ is controllable for every
choice of $\{A_j\}$.

*Proof.* $\mathcal A^{k}\mathcal E$ has $I_r$ in block $k+1$ and zeros below, so
$[\mathcal E,\mathcal A\mathcal E,\dots,\mathcal A^{L-1}\mathcal E]$ is block upper-triangular with identity diagonal blocks,
hence rank $Lr$. $\square$

This matters: the lifted process noise $\mathcal Q=\mathcal E Q\mathcal E^\top$ is **singular** (rank $r$ of $Lr$), so
the naive reading of v2 Thm 4.3 ("$(Q^{1/2},\hat A)$ stabilizable") appears to fail. Lemma 2.7 restores it.

**Theorem 2.8 ★ (Observability / detectability of the lifted DB-WM).** Let the measurement be
$y_t=\Phi_{X_t}\bar w_t^{\,(1)}+\zeta_t$, i.e. $\mathcal C=[\,\Phi_X\ 0\ \cdots\ 0\,]$ (only the current frame is
observed). Then:

1. $(\mathcal C,\mathcal A)$ is **observable** iff (i) $A_{L-1}$ is nonsingular, and (ii) for every root $\lambda$ of the
   matrix polynomial of Lemma 2.6 and every associated null vector $v$, $\Phi_X v\neq 0$.
2. $(\mathcal C,\mathcal A)$ is **detectable** iff condition (ii) holds for all roots with $|\lambda|\ge 1$.

*Proof.* PBH: rank deficiency of $[\mathcal A-\lambda I;\,\mathcal C]$ ⟺ ∃ eigenvector $\xi\neq0$ with $\mathcal C\xi=0$.
By Lemma 2.6, $\mathcal C\xi=\lambda^{L-1}\Phi_X v$. For $\lambda\neq0$ this vanishes iff $\Phi_Xv=0$, giving (ii).
For $\lambda=0$, $\xi=[0,\dots,0,v^\top]^\top$ with $A_{L-1}v=0$, and $\mathcal C\xi=0$ identically; such $\xi$ exists
iff $A_{L-1}$ is singular, giving (i). Detectability drops the $|\lambda|<1$ modes. $\square$

**Two corollaries with real teeth:**

- **Over-lagging destroys observability.** If the true memory order is $L^\ast<L$ then $A_{L-1}=0$ and the
  lifted system is *unobservable* (though still detectable, since the offending modes sit at $\lambda=0$).
  Memory order must therefore be selected, not maximized. This is a hard constraint that VLWM's
  "sample $k$ up to $K_{\max}$" heuristic has no analogue of.
- **Shadedness is no longer sufficient.** v2 Def. 4.1 requires each basis function to be activated by some
  observation. Theorem 2.8(ii) requires more: $\ker\Phi_X$ must avoid the eigen-directions of the *matrix
  polynomial*. In the visual case $\Phi_X=\phi_\theta(o_t)^\top$ with $N=1$ this is a genuine restriction;
  in the full-state visual case $w_t=\phi_\theta(o_t)$ (so $\mathcal C=\mathcal S$) it holds automatically.

**Proposition 2.9 ★ (State augmentation ≡ fixed-lag smoothing).** The Kalman filter on
$(\mathcal A,\mathcal C,\mathcal Q,\sigma_\epsilon^2 I)$ produces, in its lower blocks, the fixed-lag smoothed estimates
$\hat w_{t-j\,|\,t}$, $j=1,\dots,L-1$. Hence the lifted filter is *not* merely a bookkeeping device: it
retro-corrects past weights using present observations, which is exactly the mechanism that fills
cloud/revisit gaps in the LST application.

---

### 2.2.3 ★ Multi-horizon (direct) prediction

**Definition 2.5 ★ (Horizon operator family).** For $h=1,\dots,H$,

$$
\boxed{\;w_{t+h}\;=\;\Theta_h\,\bar w_t\;+\;\sum_{i=0}^{h-1}B^{(h)}_{i}\,u^{\text{raw}}_{t+i}\;+\;\varepsilon^{(h)}_t,\qquad
\varepsilon^{(h)}_t\sim\mathcal N(0,\Sigma_h).\;}
\tag{DB-WM-MH}
$$

The block $[\,B^{(h)}_0,\dots,B^{(h)}_{h-1}\,]$ is the finite-horizon **input-to-state Toeplitz (Markov-parameter)
block** of subspace identification. It is the linear-Gaussian counterpart of VLWM's "action-as-token"
device: feeding the whole segment $u_{t:t+h-1}$ *is* multiplying by the Toeplitz block, with no
architectural machinery required.

**Semigroup-consistent (iterated) values.** With $\mathcal A$ built from $\Theta_1$,

$$
\Theta^{\text{it}}_h=\mathcal S\mathcal A^{h},\qquad
B^{(h),\text{it}}_i=\mathcal S\mathcal A^{h-1-i}\mathcal B,\qquad
\Sigma^{\text{it}}_h=\sum_{i=0}^{h-1}\mathcal S\mathcal A^{i}\mathcal E\,Q\,\mathcal E^\top(\mathcal A^{i})^\top\mathcal S^\top .
$$

**Definition 2.6 ★ (Semigroup defect).** $D_h:=\Theta_h-\mathcal S\mathcal A^{h}\in\mathbb R^{r\times Lr}$, $h\ge2$.

**Theorem 2.10 ★ (Obstruction: direct multi-horizon prediction is an admission of misspecification).**
Suppose $\{w_t\}$ is such that $\bar w_t$ is Markov and $\mathbb E[w_{t+h}\mid\mathcal F_t]=\Theta_h\bar w_t$ for
$h=1,\dots,H$. Then necessarily $\Theta_h=\mathcal S\mathcal A^{h}$ for all $h$, i.e. $D_h\equiv0$.

*Proof.* $\mathcal A$ is fully determined by $\Theta_1$ (top row) and the definition of $\bar w$ (shift rows), and
$\mathbb E[\bar w_{t+1}\mid\bar w_t]=\mathcal A\bar w_t$. By the tower property,
$\mathbb E[w_{t+h}\mid\bar w_t]=\mathbb E\big[\mathbb E[w_{t+h}\mid\bar w_{t+1}]\mid\bar w_t\big]
=\Theta_{h-1}\mathcal A\bar w_t$. Induction from $\Theta_1=\mathcal S\mathcal A$ gives $\Theta_h=\mathcal S\mathcal A^{h}$. $\square$

**Interpretation.** $\|D_h\|$ is an *estimable, interpretable diagnostic*: it is nonzero exactly to the extent
that the process fails to be an order-$L$ linear Markov model — memory beyond $L$, nonlinearity, or
non-stationarity. Under MZ (Prop. 2.5), $\|D_h\|\to0$ as $L$ grows past the memory-kernel decay length.
**Plotting $\|D_h\|$ against $L$ is therefore a direct empirical measurement of the Mori–Zwanzig memory
depth of the thermal field** — a result with standalone physical value, independent of any forecasting gain.

---

### 2.2.4 ★ Structured parameterizations (the identifiability constraint)

Unstructured (DB-WM-MH) has $H\cdot L r^2$ parameters. At $r=512$, $L=7$, $H=3$: $5.5\times10^6$, against
$T\sim10^3$ usable dates. The regression is rank-deficient by construction
($\operatorname{rank}(\bar W\bar W^\top)\le T-L\ll Lr$). Structure is **mandatory**, not optional.

**(S1) Scalar lag weighting.** $A_j=\alpha_j A$, parameters $r^2+L$.

**Proposition 2.11 ★ (Spectral decoupling under S1).** With $A=P\Lambda P^{-1}$, $\Lambda=\mathrm{diag}(\mu_i)$
and $a(\lambda)=\sum_{j}\alpha_j\lambda^{L-1-j}$, the characteristic polynomial factorizes as
$\det(\lambda^LI-a(\lambda)A)=\prod_{i=1}^{r}\big(\lambda^{L}-a(\lambda)\mu_i\big)$.
Hence each v2 Koopman mode $\mu_i$ splits into exactly $L$ lifted modes, computable in
$O(r^3+rL^3)$ instead of $O(L^3r^3)$, and **v2 Corollary 4.2 (Koopman mode extraction) survives the lift
with an explicit mode-splitting interpretation**: a memory kernel redistributes each Koopman mode into a
cluster of $L$ delay modes whose spread encodes the thermal inertia of that mode.

Observability under S1 reduces to $\alpha_{L-1}\neq0$ and $A$ nonsingular, plus condition (ii) per mode.

**(S2) Per-mode scalar AR($L$) in the Koopman eigenbasis.** Diagonalize $A_0$ once, then let each mode
carry its own scalar memory: $\tilde w^{(i)}_{t+1}=\sum_j \alpha^{(i)}_j\tilde w^{(i)}_{t-j}+\dots$
Parameters: $rL$ only. Physically the most defensible for LST — each Koopman mode gets its own relaxation
time, which is exactly what a surface energy balance predicts.

**(S3) Reduced-rank memory.** $A_j=U C_j V^\top$ with $\operatorname{rank}\le q$: $O(Lqr)$ parameters,
covered by reduced-rank-regression / nuclear-norm VAR theory.

**(S4) Physics-constrained memory kernel.** For a diffusive surface, the MZ kernel of semi-infinite
conduction relaxes like $j^{-1/2}$; constrain $\alpha_j$ to a two-parameter relaxation family
($\alpha_j\propto j^{-\gamma}e^{-j/\tau}$). Two parameters replace $L$.

**(S5) Continuous-time generator.** $\mathcal A=\exp(\Delta t\,\mathcal L)$ so that $\Theta_h=\mathcal S\exp(h\Delta t\,\mathcal L)$
for *any real* $h$. Exactly semigroup-consistent, hence $D_h\equiv0$ by construction, and it handles
irregular revisit intervals and cloud gaps natively. This is the rigorous version of "variable-length"
prediction and should be the framework's default backbone.

---

### 2.2.5 ★ Identification: semigroup-shrunk multi-horizon least squares

Let $Z_t=[\bar w_t^\top,\,u_t^\top,\dots,u_{t+h-1}^\top]^\top$, $G_h=[\Theta_h,\,B^{(h)}_0,\dots,B^{(h)}_{h-1}]$, and
$\Pi=\mathrm{diag}(I_{Lr},0)$ the selector of the $\Theta$ block. Define

$$
\hat G_h \;=\; \arg\min_{G}\; \big\|W_{+h}-G Z\big\|_F^2 \;+\;\mu\|G\|_F^2\;+\;\nu_h\big\|G\Pi-[\,\mathcal S\mathcal A^{h}\ \ 0\,]\big\|_F^2 ,
$$

which has the **closed form**

$$
\boxed{\;\hat G_h=\Big(W_{+h}Z^\top+\nu_h[\,\mathcal S\mathcal A^{h}\ \ 0\,]\Big)\Big(ZZ^\top+\mu I+\nu_h\Pi\Big)^{-1}.\;}
$$

With $\Theta_1$ fixed at the v2 one-step ridge solution, the whole family $\{\hat G_h\}_{h=2}^H$ is obtained in
closed form — preserving the "no SGD, closed-form identification" property of v2 Algorithm 3.
$\nu_h\to\infty$ recovers the pure iterated (recursive) predictor; $\nu_h=0$ recovers the pure direct
(VLWM-style) predictor.

**Theorem 2.12 ★ (Strict dominance of horizon shrinkage).** In the isotropic-design idealization
$ZZ^\top=T\sigma_z^2 I$ and $\mu=0$, the estimator is the convex combination

$$
\hat\Theta_h=(1-\omega_h)\,\hat\Theta^{\text{dir}}_h+\omega_h\,\mathcal S\hat{\mathcal A}^{h},
\qquad \omega_h=\frac{\nu_h}{T\sigma_z^2+\nu_h}\in(0,1).
$$

Let $V_h=\mathbb E\|\hat\Theta_h^{\text{dir}}-\Theta_h^{\circ}\|_F^2$ (variance of the direct fit) and
$b_h^2=\|\mathcal S\mathcal A_\circ^{h}-\Theta_h^{\circ}\|_F^2$ (squared bias of the iterated fit, $=\|D_h^\circ\|_F^2$).
Then the risk $\mathcal R(\omega)=(1-\omega)^2V_h+\omega^2 b_h^2$ is uniquely minimized at

$$
\omega_h^\star=\frac{V_h}{V_h+b_h^2}\in(0,1)\quad\text{whenever } 0<b_h^2<\infty,\ V_h>0,
$$

with $\mathcal R(\omega_h^\star)=\dfrac{V_hb_h^2}{V_h+b_h^2}<\min\{V_h,\,b_h^2\}=\min\{\mathcal R(0),\mathcal R(1)\}$.
Hence the shrunk estimator **strictly dominates both the recursive and the direct predictor** at every finite
$T$ with nonzero, finite semigroup defect. $\square$

*Remarks.* (a) $\nu_h$ is selectable by GCV/SURE or by maximizing the innovation likelihood of the resulting
filter. (b) Theorem 2.12 is the principled explanation of VLWM's own Table 1 finding that "no single
planner/chunking strategy is universally optimal": their P1/P2/P3 are crude discrete samples of the
continuous shrinkage path $\omega\in[0,1]$. (c) A curriculum on $\gamma_h$ (horizon weights) and $\nu_h$ is
the exact analogue of VLWM §3.3, with an interpretation the original lacks: annealing from the
MLE-consistent one-step criterion to the task-aligned multi-horizon criterion.

---

### 2.2.6 ★ Observer with multi-horizon emission

**Filtering must remain one-step.** By Theorem 2.10, only $\Theta_1$ is compatible with a recursive Bayesian
update. The filter therefore runs on $(\mathcal A,\mathcal C)$:

$$
\bar w_{t|t-1}=\mathcal A\bar w_{t-1|t-1}+\mathcal B u^{\text{raw}}_{t-1},\qquad
P_{t|t-1}=\mathcal A P_{t-1|t-1}\mathcal A^\top+\mathcal E Q\mathcal E^\top+\Gamma_{\text{dyn}},
$$
$$
K_t=P_{t|t-1}\mathcal C^\top\big(\mathcal C P_{t|t-1}\mathcal C^\top+\sigma_\epsilon^2 I_N\big)^{-1},\quad
\bar w_{t|t}=\bar w_{t|t-1}+K_t\big(y_t-\mathcal C\bar w_{t|t-1}\big),\quad
P_{t|t}=(I-K_t\mathcal C)P_{t|t-1}.
$$

Existence/uniqueness of $P_\infty$ (v2 Thm 4.3) holds under Lemma 2.7 (stabilizability) + Theorem 2.8(2)
(detectability). Cost rises from $O(r^3)$ to $O(L^3r^3)$ per step unless the companion sparsity is exploited;
use a square-root/information form and note that $\mathcal A$ costs only $O(Lr^2)$ to apply.

**Forecasting uses the direct family, with directly estimated covariance.**

$$
\hat w_{t+h|t}=\hat\Theta_h\,\bar w_{t|t}+\sum_{i=0}^{h-1}\hat B^{(h)}_i\,\hat u^{\text{raw}}_{t+i},
$$
$$
P_{t+h|t}=\hat\Theta_h P_{t|t}\hat\Theta_h^\top \;+\; \hat\Sigma_h \;+\;
\sum_{i=0}^{h-1}\hat B^{(h)}_i\,\Sigma_{u,t+i}\,\hat B^{(h)\top}_i .
\tag{★}
$$

Three things in (★) are absent from v2 and matter:

**Proposition 2.13 ★ (Iterated covariance under-covers).** Let $b_h=\Theta^\circ_h-\mathcal S\hat{\mathcal A}^{h}$ be the
$h$-step bias of the iterated predictor. Then

$$
\operatorname{Cov}\!\big(w_{t+h}-\hat w^{\text{it}}_{t+h|t}\big)
=\Sigma_h^{\text{it}}+b_h\operatorname{Cov}(\bar w_t)b_h^\top \;\succeq\; \Sigma^{\text{it}}_h,
$$

so nominal $1-\alpha$ intervals built from $\Sigma^{\text{it}}_h$ have coverage strictly below $1-\alpha$ whenever
$b_h\neq0$. The direct residual covariance $\hat\Sigma_h=\widehat{\operatorname{Cov}}(\varepsilon^{(h)})$ estimates the
*full* $h$-step error covariance and is asymptotically calibrated. $\square$

> This is the sharpest defensible claim available: **in an observer-corrected latent world model, direct
> multi-horizon prediction is primarily an uncertainty-calibration device, not an accuracy device.** It is
> novel, it is testable at $H=3$ with PIT histograms and empirical coverage, and it does not depend on
> winning an accuracy race that a 3-step horizon may be too short to win.

**Input uncertainty.** The third term in (★) is unavoidable for $h\ge2$: $p_{t+1},p_{t+2}$ are *forecast*
precipitation, not observed. For a 3-day thermal forecast this term is likely to dominate; omitting it
(as the v2 formulation implicitly does, treating $p$ as known) makes the forecast over-confident by
construction.

**Multi-origin fusion.** At time $t+2$, two predictions of $w_{t+3}$ exist: $\hat w_{t+3|t}$ (h=3) and
$\hat w_{t+3|t+2}$ (h=1). Under a correctly specified model the later dominates (it conditions on strictly more
information) and fusion is worthless. Under misspecification they are not nested and the minimum-variance
GLS combination (MinT-style reconciliation) strictly improves on both. **The value of multi-origin fusion is
exactly the semigroup defect $D_h$** — the same quantity that Theorem 2.12 shrinks toward zero. This closes
the framework: one estimable quantity governs the estimator, the calibration, and the fusion gain.

---

### 2.2.7 ★ Revised open-loop bound (replacing v2 Thm 4.2)

**Theorem 2.14 ★.** Let $\rho=\rho(\mathcal A)$ and let $\varepsilon_{\text{dyn}}$ be the one-step residual bound.
Then the *iterated* $h$-step error obeys the v2 bound with $\hat A\to\mathcal A$:
$\|\hat w^{\text{it}}_{t+h}-w_{t+h}\|\le\rho^h\varepsilon_{\text{enc}}+\tfrac{\rho^h-1}{\rho-1}\varepsilon_{\text{dyn}}$
($=\varepsilon_{\text{enc}}+h\varepsilon_{\text{dyn}}$ at $\rho=1$), whereas the *direct* $h$-step error obeys

$$
\|\hat w^{\text{dir}}_{t+h}-w_{t+h}\|\;\le\;\|\Theta_h\|\,\varepsilon_{\text{enc}}+\varepsilon^{(h)}_{\text{dyn}},
$$

with $\varepsilon^{(h)}_{\text{dyn}}$ the *directly measured* $h$-step residual, which is **not** the sum of one-step
residuals. The design choice $\rho_{\max}=1$ that v2 adopts for near-conservative thermal fields is precisely
the regime in which the iterated bias grows linearly and unboundedly in $h$, and is therefore the regime in
which the direct family is theoretically favoured. **The extension is motivated by v2's own Theorem 4.2.**

---

## Estimation pipeline (revised Algorithm 3)

```
Input: weights {w_t}, raw inputs {u_t^raw}, memory order L, horizon H,
       ridge mu, shrinkage {nu_h}, horizon weights {gamma_h}

0. Whiteness pretest: fit L=1 model, Ljung-Box on residuals.
   If not rejected -> STOP, L=1 suffices. (Report this either way.)

1. Build lifted design  Wbar = [wbar_L, ..., wbar_{T-H}]  in R^{Lr x (T-L-H+1)}
   and targets W_{+h} = [w_{L+h}, ..., w_{T-H+h}] for h = 1..H.

2. Choose a structured parameterization (S1-S5). Unstructured is
   infeasible unless T >> L*r.

3. h = 1: closed-form ridge  ->  Theta_1  ->  build companion A.
   Enforce rho(A) <= rho_max via sum_j ||A_j|| <= rho_max
   (Lemma 2.6: do NOT clip sigma(A_0)).

4. h = 2..H: closed-form semigroup-shrunk ridge
      G_h = (W_{+h} Z^T + nu_h [S A^h , 0]) (Z Z^T + mu I + nu_h Pi)^{-1}

5. Semigroup defect  D_h = Theta_h - S A^h  ;  record ||D_h||_F.
   Sweep L and plot ||D_h|| vs L  ->  MZ memory depth of the field.

6. Direct residual covariances  Sigma_h = Cov(w_{t+h} - Theta_h wbar_t - Toeplitz u).
   (Do NOT construct Sigma_h by propagating Q.)

7. Select nu_h by GCV / innovation likelihood on a held-out block.

8. Check Theorem 2.8: A_{L-1} nonsingular?  roots with |lambda|>=1 avoid ker Phi_X?

Output: {Theta_h, B^(h), Sigma_h}, A, Q, D_h diagnostics
```

---

## What this buys, stated without inflation

| Claim | Status |
|---|---|
| Memory lift is derivable and Koopman-internal | **Proved** (Prop 2.5, Lemma 2.6–2.7, Thm 2.8) |
| Direct multi-horizon is derivable in closed form | **Proved** (Def 2.5, §2.2.5) |
| Direct family cannot be a single Markov model | **Proved** (Thm 2.10) — this is a *limitation*, stated honestly |
| Shrinkage strictly beats both extremes | **Proved** (Thm 2.12) — main theoretical contribution |
| Iterated covariance under-covers | **Proved** (Prop 2.13) — main empirical contribution |
| Memory depth is measurable via $\|D_h\|$ vs $L$ | **New diagnostic**, cheap to run |
| Accuracy gain at $H=3$ | **Unproven and possibly small.** Do not lead with this. |

---

## Open items (deliberately not resolved here)

1. **Uniform observability under intermittent sampling.** With cloud gaps, $\Phi_{X_t}$ is time-varying and
   observations are missing. The right object is *uniform complete observability* of the lifted pair over
   windows, and the right question is: **what is the maximum tolerable gap length as a function of $L$,
   $\rho(\mathcal A)$, and the spectral gap?** This is the highest-value open theorem for the LST application
   (cf. Kalman filtering with intermittent observations and its critical arrival rate).
2. **Encoder–horizon interaction.** If $\phi_\theta$ is trained jointly with a multi-horizon loss, the long-horizon
   terms reward *predictable* (hence uninformative) features — an anti-collapse pressure that dPPGP was not
   designed for. Either freeze $\phi_\theta$ (v2's two-phase schedule) or add a horizon-invariance penalty and
   monitor the effective rank of $\Phi$ per horizon.
3. **Gauss–Newton refinement of $\Theta_1$** under the multi-horizon loss (currently fixed at the one-step
   solution, which is a defensible but suboptimal coordinate-descent truncation).
