# Neurons and ISING model -- Part 2: The Ising Model

## Introduction

The Ising model is a cornerstone of statistical physics and plays a crucial role in machine learning. It serves as the basis for understanding complex networks such as Hopfield networks and Restricted Boltzmann Machines (RBMs). For example, a Hopfield network can be seen as a zero-temperature Ising model that settles into a local energy minimum, while an RBM represents a finite-temperature Ising model with a unique pattern of spin connections. Because of these connections to statistical mechanics, both RBMs and Hopfield networks offer a level of interpretability and efficiency that is often hard to achieve with other machine learning techniques.

The Ising model is also widely used in computational neuroscience to model neuronal networks, particularly in understanding collective neural activity and emergent behaviors in the brain. In this context, neurons are represented as spins, where each neuron can be in an active (firing) or inactive (silent) state, analogous to the up and down spins in a magnetic system.

Key applications include:

**Neural Correlations**: The Ising model helps describe correlations between neurons in networks, revealing statistical dependencies and underlying functional connectivity.

**Criticality in the Brain**: Studies suggest that neuronal networks operate near a critical point, and the Ising model helps explain self-organized criticality, optimizing information processing.

**Memory and Associative Learning**: The Hopfield network, a type of recurrent neural network used in associative memory models, is closely related to the Ising model, where stored memory patterns correspond to energy minima in the system.

**Brain Activity and Phase Transitions**: The Ising model is used to describe phase transitions between different brain states, such as wakefulness and sleep, or normal and epileptic activity.

## Background Information

### The Ising Model

The Ising model explains magnetism by considering the orientation of magnetic moments (or "spins") on a lattice. The energy (Hamiltonian) of an Ising configuration is given by:

$$H=-\sum_{i,j}J_{ij}s_is_j-\sum_ih_is_i$$

where:
- $s_i$ represents the spin at site $i$ (typically $s_i=\pm1$).
- $J_{ij}$ are the entries of a symmetric interaction matrix, defining the energy interaction between spins $i$ and $j$. In the simplest version, spins interact only with their nearest neighbors, and $J_{ij}=J$ (often set to 1).
- $h_i$ represents the external magnetic field at site $i$. In our simulation, we assume $h_i=0$.

### Statistical Mechanics Concepts

1. **Configurations:** A configuration $s$ specifies the orientation of every spin on the lattice. It completely describes the state of the system.
2. **Energy of a Configuration:** Each configuration $s$ has an associated energy $E(s)$ defined by the Hamiltonian above.
3. **Probability of a Configuration:** For a system in thermal equilibrium with a heat reservoir at temperature $T$, the probability of the system being in a configuration $s$ is given by the Boltzmann distribution:
$P(s)=\frac{1}{Z}\exp\left(-\beta E(s)\right)$
where $\beta=\frac{1}{kT}$ (we will use units with $k=1$) and $Z$ is the partition function that normalizes the probabilities.

### The Metropolis-Hastings Algorithm

The Metropolis algorithm is a Monte Carlo method that simulates the evolution of a spin system toward thermal equilibrium. The algorithm follows these steps:

1. **Initialization:** Generate a random initial configuration $\{s_i\}$ for the lattice, with an initial energy $E_0$.
2. **Spin Flip:** Select a random spin $s_k$ and propose flipping it (i.e., $s_k\to-s_k$). Calculate the new energy $E_t$ for the trial configuration.
3. **Energy Difference and Transition Probability:** Compute the energy difference $\Delta E=E_t-E_0$. Calculate the transition probability $p=\exp(-\beta\Delta E)$:
   - If $\Delta E<0$, accept the spin flip (it lowers the energy).
   - If $\Delta E>0$, accept the spin flip with probability $p$. Compare $p$ to a uniformly generated random number $r$; if $r<p$, the flip is accepted; otherwise, the original configuration is retained.
4. **Update and Iterate:** Update the system’s overall energy, magnetization, etc. Repeat steps 2--4 many times until the system reaches thermal equilibrium.

For further details on the algorithm, see the [Wikipedia article on the Metropolis--Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).

### 2D Ising Model Simulation

In this assignment, we focus on a 2D Ising model on an $N\times N$ lattice (with $N=100$). Here, the interaction matrix is constant ($J_{ij}=J$ with $J=1$), and there is no external magnetic field ($h_i=0$). The system undergoes a phase transition at a critical temperature $T_c$; below $T_c$, the lattice exhibits spontaneous magnetization, while above $T_c$ the system is disordered.

An interactive exploration of the Ising model is available via this [online JAVA simulator](https://physics.weber.edu/schroeder/software/demos/isingmodel.html).

## Tasks

Using the provided Python code (in the Jupyter Notebook `Ising_model.ipynb`) that simulates the 2D Ising model, complete the following:

**Task 1. Explain the Core Implementation of the Metropolis Algorithm:**
- Describe how the code initializes the lattice configuration.
- Explain the process of selecting a random spin and computing the energy change when flipping that spin.
- Detail how the algorithm decides whether to accept or reject the proposed spin flip based on the energy difference and the corresponding transition probability.
- Discuss how the code iterates through multiple Monte Carlo steps until equilibrium is reached.

**Task 2. Determine the Critical Temperature $T_c$:**
- Using your simulation, determine the critical temperature $T_c$ at which the system undergoes a phase transition to spontaneous magnetization.
- Run simulations at temperatures below and above $T_c$ and compare the spin configurations.
- Describe your observations. What differences do you notice in the spin patterns and overall magnetization of the lattice when the system is below versus above $T_c$?

**Task 3. Comparison with the Online JAVA Simulator:**
- Compare your simulation results and observations with those obtained from the online JAVA simulator provided above.
- Discuss any similarities or differences in the behavior of the system, particularly regarding the phase transition and magnetization.

## Deliverables

- A written explanation (or report) addressing Tasks 1, 2, and 3.
- Any figures, plots, or screenshots from your simulation that illustrate the spin configurations and the phase transition behavior.
- A discussion comparing your findings with those observed in the online JAVA simulator.

## Submission Instructions

- Save your completed assignment as a Word document (`.doc` or `.docx`) or a PDF.
- Include all required explanations, comparisons, and illustrations as specified above.
- Submit the file according to your instructor's guidelines.
