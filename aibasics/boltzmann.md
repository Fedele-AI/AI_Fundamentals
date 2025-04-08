<!-- Written by Alex Jenkins and Dr. Francesco Fedele for CEE4803/LMC4813 - (c) Georgia Tech, Spring 2025 -->

<div align="center">

# Boltzmann Machine

<img src="./Figures/RBM.png" alt="Boltzmann Machine" width="40%">

</div>

## Introduction
A **Boltzmann Machine** is a type of stochastic (randomized) neural network used for learning complex patterns and solving optimization problems. It consists of neurons that are fully connected to each other, meaning every neuron influences every other neuron. Unlike traditional neural networks, Boltzmann Machines use **probabilistic activation functions**, meaning their neurons randomly switch on or off based on a probability determined by their inputs. This allows them to escape local minima and explore different possible solutions, making them useful for searching through large solution spaces.

Mathematically, a Boltzmann Machine works by minimizing an **energy function**, similar to how a Hopfield Network operates. The system continuously updates neuron states to reduce its overall energy, eventually settling into a stable pattern that represents the most likely solution to a given problem. Training a Boltzmann Machine involves adjusting the weights so that the network correctly captures the underlying patterns in the data, often using a learning algorithm called **Contrastive Divergence**.

To explain it simply, imagine a group of people in a room adjusting the thermostat. Each person can influence the temperature (just like neurons influence each other), and the room keeps adjusting until everyone is comfortable (the system reaches a low-energy state). 

Boltzmann Machines are particularly useful in deep learning as building blocks for **Restricted Boltzmann Machines (RBMs)** and **Deep Belief Networks (DBNs)**, which are used in tasks like feature learning and data representation.

## Mathematical Formulation
The Boltzmann Machine, introduced by Geoffrey Hinton and Terry Sejnowski, is an energy-based probabilistic model that consists of a network of stochastic binary units. It is a type of recurrent neural network where neurons are symmetrically connected. This model laid the foundation for many modern deep learning architectures by introducing concepts of energy minimization and probabilistic inference.

<div align="center">

<img src="./Figures/boltzman.png" alt="Architecture of a Restricted Boltzmann Machine (RBM)" width="50%"> 

*Visible units (bottom) are connected to hidden units (top) via weights (dashed lines), with no intra-layer connections.*

</div>

## Energy-Based Formulation
A Boltzmann Machine defines a probability distribution over binary states $\mathbf{v}$ (visible units) and $\mathbf{h}$ (hidden units) using an energy function:

$$ E(\mathbf{v}, \mathbf{h}) = - \sum_{i,j} W_{ij} v_i h_j - \sum_i b_i v_i - \sum_j c_j h_j, $$

where:
- $$W_{ij}$$ are the weights between visible unit $$v_i$$ and hidden unit $$h_j$$,
- $$b_i$$ and $$c_j$$ are the biases for visible and hidden units, respectively.

The probability of a configuration $$(\mathbf{v}, \mathbf{h})$$ is given by the Boltzmann distribution:

$$ P(\mathbf{v}, \mathbf{h}) = \frac{e^{-E(\mathbf{v}, \mathbf{h}) / T}}{Z}, $$

where $$Z$$ is the partition function:

$$ Z = \sum_{\mathbf{v}, \mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h}) / T}, $$

and $$T$$ is the temperature parameter. The temperature controls the stochasticity of the system: higher temperatures allow for more exploration, while lower temperatures lead to more deterministic behavior.

## Visible and Hidden Layers
The network consists of:
- **Visible Layer:** These units correspond to observed data, such as pixels in an image.
- **Hidden Layer:** These units capture higher-order correlations in the data.

A key aspect of learning is that hidden units encode useful features that are not directly observed in the data.

## Training with Contrastive Divergence (CD-1) and Metropolis Algorithm
Hinton proposed a training method called **Contrastive Divergence (CD-1)** to approximate gradient descent on the log-likelihood. CD-1 is a variant of the **Metropolis-Hastings algorithm**, a Markov Chain Monte Carlo (MCMC) method used to sample from complex probability distributions.

The objective of training a Boltzmann Machine is to maximize the log-likelihood of the observed data $$\mathbf{v}$$, which corresponds to minimizing the Kullback-Leibler (KL) divergence between the data distribution $$P_{\text{data}}(\mathbf{v})$$ and the model distribution $$P_{\text{model}}(\mathbf{v})$$. The log-likelihood for a given visible state $$\mathbf{v}$$ is:

$$ \log P(\mathbf{v}) = \log \left( \sum_{\mathbf{h}} P(\mathbf{v}, \mathbf{h}) \right) = \log \left( \sum_{\mathbf{h}} \frac{e^{-E(\mathbf{v}, \mathbf{h})}}{Z} \right), $$

where $$Z = \sum_{\mathbf{v}, \mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}$$ (assuming $$T = 1$$ for simplicity).

The KL divergence quantifies the difference between the true data distribution and the model’s approximation:

$$ D_{KL}(P_{\text{data}} || P_{\text{model}}) = \sum_{\mathbf{v}} P_{\text{data}}(\mathbf{v}) \log \left( \frac{P_{\text{data}}(\mathbf{v})}{P_{\text{model}}(\mathbf{v})} \right), $$

Minimizing this divergence adjusts the model parameters $$W_{ij}$$, $$b_i$$, $$c_j$$ to make $$P_{\text{model}}(\mathbf{v})$$ closer to $$P_{\text{data}}(\mathbf{v})$$.

The gradient of the log-likelihood with respect to the weights $$W_{ij}$$ is:

$$ \frac{\partial \log P(\mathbf{v})}{\partial W_{ij}} = \langle v_i h_j \rangle_{data} - \langle v_i h_j \rangle_{model}, $$

where:
- $$\langle v_i h_j \rangle_{data}$$: Expectation over the training data distribution, clamping $$\mathbf{v}$$ to observed data.
- $$\langle v_i h_j \rangle_{model}$$: Expectation over the model's equilibrium distribution, requiring sampling over all configurations.

The update rule for weights is:

$$ \Delta W_{ij} = \eta (\langle v_i h_j \rangle_{data} - \langle v_i h_j \rangle_{model}), $$

where $$\eta$$ is the learning rate.

Computing $$\langle v_i h_j \rangle_{model}$$ exactly is intractable due to the partition function $$Z$$. CD-1 approximates this term using a single step of Gibbs sampling.

### Two Phases of CD-1 Training
Training with CD-1 involves alternating between two phases: the **prediction phase** and the **dreaming phase**.

#### Prediction Phase: Hidden State Given Visible State
In the **prediction phase**, the visible units $$\mathbf{v}$$ are clamped to the training data, and the hidden units $$\mathbf{h}$$ are sampled based on:

$$P(h_j = 1 | \mathbf{v}) = \sigma \left( \sum_i W_{ij} v_i + c_j \right),$$

where $$\sigma(x) = \frac{1}{1 + e^{-x}}$$ is the sigmoid function.

- **Process**: 
  1. Clamp $$\mathbf{v}$$ to a training example.
  2. Compute $$P(h_j = 1 | \mathbf{v})$$ for each hidden unit.
  3. Sample $$h_j \in \{0, 1\}$$ stochastically.
  4. Compute $$\langle v_i h_j \rangle_{data} = v_i h_j$$.

#### Dreaming Phase: Visible State Given Hidden State
In the **dreaming phase**, hidden units $$\mathbf{h}$$ from the prediction phase are used to sample reconstructed visible units $$\mathbf{v}'$$:

$$ P(v_i' = 1 | \mathbf{h}) = \sigma \left( \sum_j W_{ij} h_j + b_i \right). $$

- **Process**: 
  1. Use $$\mathbf{h}$$ from the prediction phase.
  2. Compute $$P(v_i' = 1 | \mathbf{h})$$ for each visible unit.
  3. Sample $$v_i' \in \{0, 1\}$$.
  4. Compute $$\langle v_i' h_j \rangle$$ as an approximation of $$\langle v_i h_j \rangle_{model}$$.

### CD-1 Formulas and Approximation
CD-1 approximates the gradient for weights and biases:
- **Positive Phase**: 
  $$\langle v_i h_j \rangle_{data} = v_i \cdot \sigma \left( \sum_i W_{ij} v_i + c_j \right),$$
  (or sampled $$h_j$$).
- **Negative Phase**: 
  $$\langle v_i' h_j \rangle_{model} \approx v_i' \cdot h_j,$$
  with $$v_i' = \sigma \left( \sum_j W_{ij} h_j + b_i \right)$$.

- **Weight Update**: 
  $$\Delta W_{ij} = \eta \left( \langle v_i h_j \rangle_{data} - \langle v_i' h_j \rangle_{model} \right).$$

- **Bias Updates**: 
  The gradients for biases are derived similarly:
  $$\frac{\partial \log P(\mathbf{v})}{\partial b_i} = v_i - \langle v_i \rangle_{model},$$
  $$\frac{\partial \log P(\mathbf{v})}{\partial c_j} = \langle h_j \rangle_{data} - \langle h_j \rangle_{model}.$$
  Using CD-1’s one-step approximation:
  $$\Delta b_i = \eta (v_i - v_i'),$$
  $$\Delta c_j = \eta (h_j - h_j'),$$
  where:
  - $$v_i$$: Clamped visible unit from the data.
  - $$v_i'$$: Reconstructed visible unit from the dreaming phase.
  - $$h_j$$: Sampled hidden unit from the prediction phase.
  - $$h_j'$$: Re-sampled hidden unit from $$P(h_j | \mathbf{v}')$$ (often $$h_j$$ is reused in CD-1 for simplicity).

- **Explanation**: 
  - For $$b_i$$, the update increases the bias if the data’s $$v_i = 1$$ more often than the model predicts $$v_i' $$, aligning the visible units with the training data.
  - For $$c_j$$, the update adjusts the hidden unit bias based on the difference between data-driven and model-driven activations, refining feature detection.
  - In CD-1, $$h_j'$$ is typically not re-sampled after one step, so $$\Delta c_j = \eta (h_j - h_j) = 0$$ unless additional sampling is performed, but the formula is included for completeness.

CD-1 uses one Gibbs step to approximate the model distribution efficiently.

### Annealing Strategy
An **annealing strategy** involves gradually decreasing the temperature $$T$$ during sampling or training. At high $$T$$, the Boltzmann distribution $$P(\mathbf{v}, \mathbf{h}) = \frac{e^{-E(\mathbf{v}, \mathbf{h}) / T}}{Z}$$ is flatter, encouraging exploration of diverse states. As $$T$$ decreases, the distribution sharpens, focusing on low-energy (high-probability) states:
- **Formula with Temperature**: 
  $$P(\mathbf{v}, \mathbf{h}; T) = \frac{e^{-E(\mathbf{v}, \mathbf{h}) / T}}{Z(T)}, \quad Z(T) = \sum_{\mathbf{v}, \mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h}) / T}.$$
- **Process**: Start with a high $$T$$ (e.g., 10), reduce it (e.g., $$T \leftarrow T \cdot 0.9$$ each iteration or epoch, converging to $$T = 1$$ or lower.
- **Purpose**: Improves convergence by escaping local minima early and refining solutions later, commonly used in simulated annealing with MCMC methods like Metropolis-Hastings.

### Metropolis-Hastings Sampling for Image Generation
The Metropolis-Hastings algorithm, while related to Gibbs sampling, provides a general framework for generating samples from $$P(\mathbf{v}, \mathbf{h})$$. However, for Boltzmann Machines, classical **Gibbs sampling** is typically employed due to the conditional independence of units given the others. Here, we describe image generation using Gibbs sampling:

- **Gibbs Sampling Process**: Alternates between sampling visible and hidden units based on their conditional distributions:
  - $$P(h_j = 1 | \mathbf{v}) = \sigma \left( \sum_i W_{ij} v_i + c_j \right)$$,
  - $$P(v_i = 1 | \mathbf{h}) = \sigma \left( \sum_j W_{ij} h_j + b_i \right)$$.

- **Steps for Generating New Images**:
  1. **Initialization**: Start with a random visible vector $$\mathbf{v}^{(0)}$$ (e.g., binary noise for pixel values).
  2. **Sample Hidden Units**: For each hidden unit $$h_j$$, compute $$P(h_j = 1 | \mathbf{v}^{(0)})$$ and sample $$h_j^{(1)}$$ in $$[0, 1]$$.
  3. **Sample Visible Units**: For each visible unit $$v_i$$, compute $$P(v_i = 1 | \mathbf{h}^{(1)})$$ and sample $$v_i^{(1)}$$ in $$[0, 1]$$, updating $$\mathbf{v}^{(1)}$$.
  4. **Iterate**: Repeat steps 2–3 for $$k$$ iterations (e.g., $$k = 1000$$) to converge toward the equilibrium distribution $$P(\mathbf{v}, \mathbf{h})$$.
  5. **Output**: After sufficient iterations, $$\mathbf{v}^{(k)}$$ represents a generated image conforming to the learned distribution.

- **Explanation**: 
  - Each step conditions on the current state of the other layer, leveraging the bipartite structure of a Boltzmann Machine (if restricted) or full connectivity (if unrestricted).
  - The process explores the joint distribution, gradually refining $$\mathbf{v}$$ to resemble training data patterns (e.g., Van Gogh-like images).
  - Convergence is asymptotic; practical implementations use a finite $$k$$, often with annealing (decreasing $$T$$) to enhance sample quality.

Gibbs sampling is simpler than Metropolis-Hastings for Boltzmann Machines because it directly exploits the conditional distributions without needing a proposal distribution, making it efficient for image generation.

## Video Notes

### Video 1: Boltzmann Machine made simple
[![Watch the video](https://img.youtube.com/vi/CgA-O-iKmY8/0.jpg)](https://youtu.be/CgA-O-iKmY8)

## References and Further Reading

**Ackley, D. H.**, **Hinton, G. E.**, and **Sejnowski, T. J.** "A Learning Algorithm for Boltzmann Machines." *Cognitive Science*, vol. 9, no. 1, 1985, pp. 147–169. [DOI:10.1207/s15516709cog0901_7](https://doi.org/10.1207/s15516709cog0901_7).

---

<div align="center">

[⬅️ Previous](hopfieldnetwork.md) | [🏠 Home](/README.md) |  [Next ➡️](normalizingflow.md)

</div>