<!-- Written by Dr. Francesco Fedele for CEE4803 Art & generative AI - (c) Georgia Tech, Spring 2026 -->

<div align="center">

# Modern Hopfield Networks  
A Log-Sum-Exp Generalization of the Classical Hopfield Model

</div>


## Introduction

The classical Hopfield network uses a quadratic energy based on pairwise interactions and stores only ~0.14N patterns.  
The modern Hopfield network (Ramsauer et al., 2020) replaces the quadratic term $\sum_\mu (\mathbf{s}\cdot\boldsymbol{\xi}^\mu)^2$  
with a *log-sum-exp* (LSE) term, yielding exponential storage capacity and a direct equivalence to self-attention.  
In the following we derive both models, highlight the role of the quadratic regularizer, define the state-dependent matrix $J(\mathbf{s})$,  
and show the transformer connection.

## 1. Classical Hopfield Network

The energy of the classical Hopfield network is

$$
E(\mathbf{s}) = -\frac{1}{2} \mathbf{s}^\top W \mathbf{s}
$$

with binary states $\mathbf{s} \in \{-1,+1\}^N$ and the Hebbian weight matrix

$$
W = \frac{1}{N}\sum_{\mu=1}^P \boldsymbol{\xi}^\mu (\boldsymbol{\xi}^\mu)^\top \qquad (\text{patterns } \boldsymbol{\xi}^\mu \in \{-1,+1\}^N).
$$

Substituting the Hebbian rule into the energy gives a revealing form:

$$
\begin{aligned}
E(\mathbf{s}) 
&= -\frac{1}{2N} \mathbf{s}^\top \Bigl( \sum_{\mu=1}^P \boldsymbol{\xi}^\mu (\boldsymbol{\xi}^\mu)^\top \Bigr) \mathbf{s} \\
&= -\frac{1}{2N} \sum_{\mu=1}^P (\mathbf{s} \cdot \boldsymbol{\xi}^\mu)^2.
\end{aligned}
$$

This shows that the classical energy is *minus the sum of squared dot products* with all stored patterns.  
The network tries to maximize the (squared) similarity to the memories. Because the interactions are purely quadratic (pairwise),  
the capacity is severely limited ($\approx 0.14N$ random patterns before catastrophic crosstalk).

Dynamics are usually synchronous sign updates:

$$
s_i \leftarrow \mathrm{sign}\Bigl( \sum_j W_{ij} s_j \Bigr).
$$

## 2. Modern Hopfield Network — Log-Sum-Exp Generalization

The modern formulation keeps the insight that the network should maximize similarity to stored patterns,  
but replaces the *sum of squares* above with a much sharper *log-sum-exp* function.

The modern energy is

$$
E(\mathbf{s}) = \frac{1}{2} \|\mathbf{s}\|^2 - \frac{1}{\beta} \log \sum_{\mu=1}^P \exp\bigl(\beta \boldsymbol{\xi}^\mu \cdot \mathbf{s}\bigr),
$$

or equivalently, with memory matrix $X = [\boldsymbol{\xi}^1 \dots \boldsymbol{\xi}^P] \in \mathbb{R}^{N\times P}$:

$$
E(\mathbf{s}) = \frac{1}{2} \|\mathbf{s}\|^2 + \mathrm{LSE}(\beta X^\top \mathbf{s}),
$$

where $\mathrm{LSE}(\mathbf{z}) = \log\sum_i e^{z_i}$.

### Why replace $\sum (\mathbf{s}\cdot\boldsymbol{\xi}^\mu)^2$ with LSE?

- The quadratic form $\sum (\mathbf{s}\cdot\boldsymbol{\xi}^\mu)^2$ creates relatively **shallow** and overlapping basins.
- The exponential inside the log-sum-exp creates **extremely sharp** peaks: when $\beta$ is large, the energy is dominated by the single best-matching pattern.
- Result: the attraction basins become almost disjoint even when $P \gg N$, giving **exponential storage capacity**.

<div align="center">

<img src="./Figures/energy_landscape_hopfield.png" alt="energy_landscape" width="80%">

> **Figure caption**  
> Energy landscapes of classical Hopfield (left: shallow basins) and modern generalized Hopfield (right: deep, narrow wells).  
> The classical model features broad, shallow attractors leading to slow and sometimes ambiguous convergence,  
> while the modern model creates sharp, deep wells that enable rapid and reliable convergence even with many stored patterns.

</div>

### Why do we need the quadratic regularizer $\frac12 \|\mathbf{s}\|^2$?

Without the $\frac12 \|\mathbf{s}\|^2$ term the energy would be unbounded from below: the system could make $\|\mathbf{s}\|$ arbitrarily large  
in the direction of any memory to drive $E\to -\infty$.  

The quadratic term acts as a *soft L2 penalty* (a spring pulling $\mathbf{s}$ toward the origin).  
It creates a balance between:

- the attraction toward stored patterns (LSE term), and
- the cost of large state norms.

This balance guarantees stable fixed points with finite norm and makes the dynamics equivalent to scaled self-attention.

## 3. Derivation of the Update Rule

Gradient descent on the modern energy yields

$$
\frac{d\mathbf{s}}{dt} = -\nabla E(\mathbf{s}) = -\mathbf{s} + X \cdot \mathrm{softmax}(\beta X^\top \mathbf{s}).
$$

In discrete form (most common in practice) we obtain the fixed-point iteration:

$$
\mathbf{s}^{(t+1)} = X \cdot \mathrm{softmax}(\beta X^\top \mathbf{s}^{(t)}).
$$

## 4. State-Dependent Weight Matrix $J(\mathbf{s})$

Define the attention weights

$$
\mathbf{p}(\mathbf{s}) = \mathrm{softmax}(\beta X^\top \mathbf{s}).
$$

The state-dependent weight matrix is

$$
J(\mathbf{s}) := X \mathrm{diag}(\mathbf{p}(\mathbf{s})) X^\top = \sum_{\mu=1}^P p_\mu(\mathbf{s}) \, \boldsymbol{\xi}^\mu (\boldsymbol{\xi}^\mu)^\top.
$$

$J(\mathbf{s})$ is the direct generalization of the classical Hebbian matrix: instead of equal weights $1/P$ for every pattern,  
each pattern $\boldsymbol{\xi}^\mu$ is weighted by how well the *current state* $\mathbf{s}$ matches it.  
Hence the matrix is **state-dependent**.

## 5. Modern Hopfield = Scaled Self-Attention

Let the current state $\mathbf{s}$ act as query, and the stored patterns $X$ act as both keys and values.  
The scaled dot-product attention formula is

$$
\mathrm{Attention}(\mathbf{q},K,V) = \mathrm{softmax}\Bigl(\frac{\mathbf{q} K^\top}{\sqrt{d_k}}\Bigr) V.
$$

Setting $\mathbf{q} = \mathbf{s}$, $K = X^\top$, $V = X$, and scaling factor $\sqrt{d_k} = 1/\beta$ gives exactly

$$
\mathbf{s}_\text{new} = X \cdot \mathrm{softmax}(\beta X^\top \mathbf{s}),
$$

which is identical to the modern Hopfield update.  
Thus a single modern Hopfield layer **is** one self-attention head (with patterns stored as key/value memory).

## 6. Conclusion

By replacing the quadratic similarity term $\sum (\mathbf{s}\cdot\boldsymbol{\xi}^\mu)^2$ with a log-sum-exp  
and adding the norm regularizer $\frac12\|\mathbf{s}\|^2$,  
the modern Hopfield network achieves exponential capacity and becomes mathematically equivalent to transformer self-attention.


## Appendix: Storage Capacity – Classical vs. Modern Hopfield Networks

### Classical Hopfield Network – Linear Capacity (≈ 0.14N)

**Result (Amit, Gutfreund & Sompolinsky, 1985–1987)**  
For random binary patterns $\xi^\mu \in \{-1,+1\}^N$ with $P = \alpha N$ stored patterns,  
the classical Hopfield network with Hebbian weights $W_{ij} = \frac{1}{N} \sum_\mu \xi_i^\mu \xi_j^\mu$ has a critical storage ratio

$$
\alpha_c \approx 0.138 \quad (\text{or roughly } 0.14N \text{ patterns}).
$$

Beyond this value, the network suffers **catastrophic forgetting**: spurious states proliferate, and the original patterns become unstable fixed points.

#### Proof Sketch (Signal-to-Noise + Replica Analysis)

1. **Local field decomposition**  
   The local field at neuron $i$ for pattern $\mu$ is

   $$
   h_i^\mu = \sum_{j \neq i} W_{ij} \xi_j^\mu = \xi_i^\mu + \mathrm{crosstalk term}.
   $$

   The signal term is $\xi_i^\mu$ (strength 1), while the crosstalk is

   $$
   \mathrm{crosstalk} = \frac{1}{N} \sum_{\nu \neq \mu} \left( \sum_j \xi_j^\nu \xi_j^\mu \right) \xi_i^\nu.
   $$

2. **Gaussian approximation**  
   For random orthogonal-ish patterns, the crosstalk term is approximately Gaussian with variance

   $$
   \sigma^2 \approx \frac{P-1}{N} \approx \alpha.
   $$

3. **Stability condition**  
   For pattern $\mu$ to be stable, the signal must overcome noise:

   $$
   1 > \kappa \sigma \quad \Rightarrow \quad \alpha < \frac{1}{\kappa^2}.
   $$

   Mean-field / replica-symmetric analysis gives the critical value $\kappa_c \approx 2.88$ (from the TAP equations or Gardner volume calculation), yielding

   $$
   \alpha_c \approx \frac{1}{2.88^2} \approx 0.138.
   $$

Beyond $\alpha_c$, many patterns become unstable, and the network falls into a spin-glass-like phase with many spurious attractors.

### Modern / Dense Hopfield Network – Exponential Capacity

**Result (Krotov & Hopfield 2016, Demircigil et al. 2017, Ramsauer et al. 2020)**  
In the modern Hopfield network with energy

$$
E(\mathbf{s}) = \frac{1}{2} \|\mathbf{s}\|^2 - \frac{1}{\beta} \log \sum_{\mu=1}^P \exp(\beta \, \boldsymbol{\xi}^\mu \cdot \mathbf{s}),
$$

the number of stable fixed points (stored patterns) can scale **exponentially** with the number of neurons $N$:

$$
P \sim \exp(c N) \quad \text{for some } c > 0.
$$

In practice, $P$ up to several thousands can be stored reliably in networks with $N \sim 10^3$–$10^4$ dimensions.

#### Proof Sketch (Exponential Number of Attractors)

1. **Fixed-point condition**  
   A pattern $\boldsymbol{\xi}^\mu$ is a fixed point if

   $$
   \boldsymbol{\xi}^\mu = X \cdot \mathrm{softmax}(\beta X^\top \boldsymbol{\xi}^\mu).
   $$

   At high $\beta$, $\mathrm{softmax}(\beta X^\top \boldsymbol{\xi}^\mu)$ becomes very close to a one-hot vector on pattern $\mu$:

   $$
   p_\nu \approx \delta_{\nu\mu} \quad \Rightarrow \quad \boldsymbol{\xi}^\mu \approx \boldsymbol{\xi}^\mu,
   $$

   so all stored patterns remain exact (or very close) fixed points even when $P \gg N$.

2. **Basin size and crosstalk suppression**  
   The key difference is the **sharpness** of the log-sum-exp term.  
   For a corrupted state $\mathbf{s} = \boldsymbol{\xi}^\mu + \boldsymbol{\delta}$ (small noise), the dot products are

   $$
   \boldsymbol{\xi}^\nu \cdot \mathbf{s} \approx
   \begin{cases}
   N + \boldsymbol{\xi}^\mu \cdot \boldsymbol{\delta} & \nu = \mu \\
   \mathcal{O}(\sqrt{N}) + \text{crosstalk} & \nu \neq \mu.
   \end{cases}
   $$

   At large $\beta$, the exponential suppresses all non-matching terms exponentially:

   $$
   \exp(\beta \boldsymbol{\xi}^\nu \cdot \mathbf{s}) \ll \exp(\beta N) \quad \text{for } \nu \neq \mu.
   $$

   → the softmax is dominated by the correct pattern → **basin of attraction remains large** despite many memories.

3. **Exponential number of attractors**  
   The number of stable states corresponds roughly to the number of directions in which the energy landscape has deep, narrow wells.  
   Because the log-sum-exp creates **winner-take-all** behavior, the basins remain separated even when patterns are linearly dependent or $P \gg N$.  
   Theoretical analyses (using random matrix theory or statistical mechanics) show that the number of attractors grows as

   $$
   P \lesssim \exp\left( c \frac{N}{\log N} \right) \quad \text{to} \quad \exp(c N)
   $$

   depending on the precise model variant and pattern correlation structure.  
   In practice, $P \sim 10^3–10^4$ is achievable with $N \sim 10^3–10^4$ in dense associative memory models.

### Summary Comparison

| Model                  | Energy term for memories              | Capacity          | Basin shape             | Convergence speed |
|------------------------|----------------------------------------|-------------------|--------------------------|-------------------|
| Classical Hopfield     | $-\sum_\mu (\mathbf{s} \cdot \xi^\mu)^2$ | $\sim 0.14N$     | shallow, overlapping    | slow, multi-step  |
| Modern Hopfield        | $-\frac{1}{\beta} \log \sum_\mu \exp(\beta \mathbf{s} \cdot \xi^\mu)$ | $\exp(cN)$       | deep, narrow, disjoint  | very fast (often 1–5 steps) |

The exponential capacity of the modern model is the direct consequence of replacing polynomial (quadratic) similarity with an exponential (log-sum-exp) similarity measure.
