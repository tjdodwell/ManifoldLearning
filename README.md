# ManifoldLearning

Goal oriented dimension reduction on the Grassmann manifold with ridge surrogate models and heteroscedastic GP residual modelling.

This repository contains research code for manifold learning methods built around:

- Grassmann manifold optimisation
- Ridge surrogate modelling
- Gaussian process residual modelling (via GPyTorch)
- Torch based implementation

---

## Project Structure

This project uses a `src/` layout.

```
ManifoldLearning/
  src/manifold_learning/
  examples/
  tests/
  pyproject.toml
```

The Python package lives in:

```
src/manifold_learning
```


## Installation (Recommended)

### Install uv

If you do not already have `uv`, install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal afterwards.

### Clone the repository

```bash
git clone <REPO_URL>
cd ManifoldLearning
```

### Create the environment and install

```bash
uv venv
uv pip install -e .
```

This installs the package in editable mode so local changes are immediately reflected.

# Stochastic Grassmann Optimisation for Goal Oriented Dimension Reduction

This note outlines the mathematics for stochastic optimisation on the Grassmann manifold for goal oriented dimension reduction. The goal is to identify a low dimensional linear subspace of the input space that minimises the best achievable approximation error of a quantity of interest (QoI).

## 1. Problem setup

Let $x \in \mathbb R^d$ denote a high dimensional input and let $f(x) \in \mathbb R$ denote a scalar QoI. We seek a reduced representation of the form

$$
f(x) \approx \hat f(x) = p_\alpha(M^\top x),
$$

where

- $M \in \mathbb R^{d \times r}$ with $r \ll d$ has orthonormal columns, $M^\top M = I_r$
- $z = M^\top x \in \mathbb R^r$ are reduced coordinates
- $p_\alpha : \mathbb R^r \to \mathbb R$ is a surrogate parameterised by $\alpha$

A common and convenient case is a surrogate that is linear in parameters:

$$
p_\alpha(z) = \phi(z)^\top \alpha,
$$

where $\phi : \mathbb R^r \to \mathbb R^p$ is a fixed feature or basis map (polynomials, RBF features, splines, etc.) and $\alpha \in \mathbb R^p$.

Given data $\{(x_i,f_i)\}_{i=1}^N$, define the design matrix

$$
\Phi(M) \in \mathbb R^{N \times p}, \qquad \Phi_{i:}(M) = \phi(M^\top x_i)^\top,
$$

and the stacked outputs $f = (f_1,\dots,f_N)^\top$.

## 2. Eliminating the surrogate coefficients (variable projection)

Consider the joint least squares problem

$$
\min_{\alpha,M} \| f - \Phi(M)\alpha \|_2^2.
$$

For fixed $M$, the minimiser in $\alpha$ is the (optionally regularised) least squares solution

$$
\alpha^\star(M) = \arg\min_\alpha \| f - \Phi(M)\alpha \|_2^2
= (\Phi^\top \Phi + \lambda I)^{-1}\Phi^\top f,
$$

where $\lambda \ge 0$ is a ridge parameter (recommended in practice for numerical stability and smoothness).

Substituting $\alpha^\star$ gives the reduced objective that depends only on the subspace:

$$
J(M) = \| f - \Phi(M)\alpha^\star(M) \|_2^2.
$$

Equivalently, using the ridge projection operator

$$
\Pi_{\Phi(M)}^{(\lambda)} = \Phi(M)\big(\Phi(M)^\top \Phi(M) + \lambda I\big)^{-1}\Phi(M)^\top,
$$

we can write

$$
J(M) = \| f - \Pi_{\Phi(M)}^{(\lambda)} f \|_2^2.
$$

This is the key reformulation: the problem becomes an optimisation over subspaces, rather than over both $(\alpha,M)$.

## 3. Grassmann manifold viewpoint

Only the subspace spanned by the columns of $M$ matters. If $Q \in O(r)$ is any orthogonal matrix, then $M$ and $MQ$ represent the same subspace and yield the same reduced coordinates up to rotation.

Therefore the natural search space is the Grassmann manifold $\mathcal G(r,d)$, the set of $r$ dimensional subspaces of $\mathbb R^d$.

A point on $\mathcal G(r,d)$ can be represented by any $M \in \mathbb R^{d \times r}$ with $M^\top M = I_r$.

### Tangent space and projection

The tangent space at $M$ is

$$
T_M\mathcal G(r,d) = \{ \Delta \in \mathbb R^{d \times r} : M^\top \Delta = 0 \}.
$$

Given any matrix $G \in \mathbb R^{d \times r}$, its projection to the tangent space is

$$
\mathrm{Proj}_M(G) = (I - MM^\top)G.
$$

## 4. Stochastic objective and mini batches

The population objective can be written as

$$
\mathcal J(M) = \mathbb E\big[(f(x) - \hat f_M(x))^2\big],
$$

where $\hat f_M(x) = \phi(M^\top x)^\top \alpha^\star(M)$ and $\alpha^\star(M)$ is the least squares or ridge solution induced by the distribution of $(x,f)$.

In practice we work with mini batches. Let $\mathcal B$ be a batch of size $B$:

$$
\mathcal B = \{(x_i,f_i)\}_{i=1}^B.
$$

Define

- $z_i = M^\top x_i$
- $\Phi_{\mathcal B}(M) \in \mathbb R^{B \times p}$ with rows $\phi(z_i)^\top$
- $f_{\mathcal B} = (f_1,\dots,f_B)^\top$

The batch objective is

$$
J_{\mathcal B}(M) = \min_\alpha \| f_{\mathcal B} - \Phi_{\mathcal B}(M)\alpha \|_2^2.
$$

A stochastic optimisation algorithm uses $J_{\mathcal B}(M)$ and its gradient as a noisy estimate of the full objective and gradient.

## 5. Euclidean gradient of the reduced objective

For a given batch, compute the ridge solution

$$
\alpha^\star = (\Phi_{\mathcal B}^\top \Phi_{\mathcal B} + \lambda I)^{-1}\Phi_{\mathcal B}^\top f_{\mathcal B}.
$$

Define residuals

$$
e = f_{\mathcal B} - \Phi_{\mathcal B}\alpha^\star \in \mathbb R^B.
$$

The reduced batch objective is $J_{\mathcal B}(M) = \|e\|_2^2$.

Using the variable projection principle, we can differentiate $J_{\mathcal B}(M)$ without explicitly differentiating $\alpha^\star$ (because $\alpha^\star$ is the minimiser of the inner problem). The differential is

$$
\mathrm d J_{\mathcal B} = -2\, e^\top (\mathrm d\Phi_{\mathcal B})\, \alpha^\star.
$$

This leads to a Euclidean gradient of the form

$$
\nabla_M J_{\mathcal B}(M) = -2 \sum_{i=1}^B e_i \, \nabla_M \big(\phi(M^\top x_i)^\top \alpha^\star\big).
$$

### Chain rule expansion

Let $z_i = M^\top x_i$. Define the scalar surrogate prediction

$$
s_i(M) = \phi(z_i)^\top \alpha^\star.
$$

Then

$$
\nabla_M s_i(M) = x_i \, (\nabla_{z}\, s_i)^\top,
$$

and since $s_i = \phi(z_i)^\top \alpha^\star$,

$$
\nabla_z s_i = J_\phi(z_i)^\top \alpha^\star \in \mathbb R^r,
$$

where $J_\phi(z_i) \in \mathbb R^{p \times r}$ is the Jacobian of $\phi$ evaluated at $z_i$.

Combining these,

$$
\nabla_M s_i(M) = x_i \, \big(J_\phi(z_i)^\top \alpha^\star\big)^\top.
$$

Therefore the Euclidean batch gradient can be written as

$$
G_{\mathcal B}(M) \equiv \nabla_M J_{\mathcal B}(M)
= -2 \sum_{i=1}^B e_i \, x_i \, \big(J_\phi(z_i)^\top \alpha^\star\big)^\top.
$$

This is the main working expression used in implementation.

## 6. Riemannian (Grassmann) gradient

To obtain a gradient that respects the Grassmann geometry, project the Euclidean gradient onto the tangent space:

$$
\mathrm{grad}\,J_{\mathcal B}(M) = (I - MM^\top) \, G_{\mathcal B}(M).
$$

This removes components that only change the choice of basis within the same subspace, ensuring invariance under $M \mapsto MQ$.

## 7. Stochastic Grassmann gradient descent

A basic stochastic Grassmann descent step takes the form

$$
M^{+} = \mathrm{Retr}_M\big(-\eta_k \, \mathrm{grad}\,J_{\mathcal B}(M)\big),
$$

where $\eta_k > 0$ is a stepsize and $\mathrm{Retr}_M(\cdot)$ is a retraction mapping back onto the manifold.

### QR (or polar) retraction

A common and simple retraction is

1. Take an unconstrained step:
$$
\widetilde M = M - \eta_k \, \mathrm{grad}\,J_{\mathcal B}(M).
$$

2. Re orthonormalise columns:
$$
\widetilde M = QR, \qquad M^{+} = Q.
$$

This yields a valid point on the Stiefel manifold, and the induced subspace update is a valid Grassmann retraction.

### Geodesic update (optional)

If $\Delta = -\eta_k \, \mathrm{grad}\,J_{\mathcal B}(M)$ and $\Delta$ has compact SVD $\Delta = U\Sigma V^\top$, then the Grassmann geodesic update is

$$
M^{+} = M V\cos(\Sigma)V^\top + U\sin(\Sigma)V^\top.
$$

This is more exact but typically more expensive than QR.

## 8. Stepsize conditions for stochastic convergence

For classical stochastic approximation, a standard sufficient condition is

$$
\sum_{k=0}^\infty \eta_k = \infty,
\qquad
\sum_{k=0}^\infty \eta_k^2 < \infty.
$$

A typical choice is $\eta_k = \eta_0 (1+k)^{-\gamma}$ with $\gamma \in (1/2,1]$.

In practice one may use adaptive schemes, but the above is the clean theoretical baseline.

## 9. Algorithm summary

Given an initial $M_0 \in \mathbb R^{d \times r}$ with $M_0^\top M_0 = I_r$, iterate for $k=0,1,2,\dots$:

1. Sample a mini batch $\mathcal B_k = \{(x_i,f_i)\}_{i=1}^B$
2. Compute projected coordinates $z_i = M_k^\top x_i$ and construct $\Phi_{\mathcal B_k}(M_k)$
3. Solve ridge regression for $\alpha^\star$
$$
\alpha^\star = \big(\Phi^\top \Phi + \lambda I\big)^{-1}\Phi^\top f_{\mathcal B_k}
$$
4. Compute residuals
$$
e = f_{\mathcal B_k} - \Phi \alpha^\star
$$
5. Compute Euclidean gradient
$$
G = -2 \sum_{i=1}^B e_i \, x_i \, \big(J_\phi(z_i)^\top \alpha^\star\big)^\top
$$
6. Project to Grassmann tangent space
$$
\Delta = (I - M_k M_k^\top)\, G
$$
7. Retract (QR)
$$
\widetilde M = M_k - \eta_k \Delta, \qquad \widetilde M = QR, \qquad M_{k+1} = Q
$$

Stop when $J_{\mathcal B}(M)$ stabilises, or after a fixed budget, then train a final surrogate in the reduced space using the full dataset.

## 10. Practical notes

- Ridge regularisation ($\lambda > 0$) is recommended to avoid rank deficiency in $\Phi^\top \Phi$ and to keep the objective smoother as $M$ varies.
- Input scaling matters. Normalising or whitening $x$ often improves stability and interpretability of the learned directions.
- The reduced dimension $r$ can be selected by monitoring validation error as a function of $r$, or by inspecting the marginal gain in $J(M)$ as $r$ increases.
- This approach assumes a globally useful linear subspace. If the QoI exhibits regime changes, local subspaces or mixtures of subspaces may be more appropriate.



