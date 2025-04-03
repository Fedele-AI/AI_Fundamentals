# Homework: Gaussian-Bernoulli Restricted Boltzmann Machine

## Part 1: Theory of the Gaussian-Bernoulli RBM

A Restricted Boltzmann Machine (RBM) can learn the probability structure of a given dataset and generate samples that share the same statistical properties as the original data.

### Sampling from Learned Distributions

- Draw 10 samples from a unimodal distribution of faces with large variance.
- Draw 10 samples from a unimodal distribution of faces with small variance.
- Draw 10 samples from a bimodal distribution of faces.
- Draw 10 samples from a unimodal distribution of still lifes.
- Draw 10 samples from a bimodal distribution of still lifes.
- Compare and interpret the differences in the generated samples for each case. How do variance and modality affect the diversity and realism of the generated faces?

### Architecture of the RBM

1. Sketch the architecture of a Gaussian-Bernoulli RBM. Explain the roles of the visible and hidden layers.
2. Why is it called a Gaussian-Bernoulli RBM? What do the terms "Gaussian" and "Bernoulli" refer to?
3. The energy function of an RBM is given by:

   $E(v,h) = -\sum_{i,j} w_{ij} v_i h_j - \sum_i b_i v_i - \sum_j c_j h_j$

   where $w_{ij}$ are the weights and $b_i, c_j$ are the biases. The probability distribution of states in an RBM follows the Boltzmann distribution:

   $P(v,h) = \frac{e^{-\beta E(v,h)}}{Z}$

   where the inverse temperature is defined as $\beta = \frac{1}{k_B T}$, $T$ is the temperature, and $Z$ is the partition function.

   - What is the role of temperature $T$ in this distribution?
   - Why don't we want the temperature to be too close to zero? (hint: check notes on Hopfield)
   - What kind of states are most likely in the low-temperature regime? (hint: states that maximize probability, check notes on Hopfield)
   - Why don't we want the temperature to be too high? (hint: check note on Ising model)
   - What kind of states are most likely in the high-temperature regime? (hint: states that maximize probability, check note on Ising model)

4. Explain the two main steps of the Metropolis algorithm for sampling from an RBM (which are also fundamental for training):
   - Positive phase
   - Negative (dreaming) phase

   These steps are based on the following conditional probabilities derived from the Boltzmann distribution:

   $P(h_j = 1 | v) = \sigma\left( \sum_i w_{ij} v_i + c_j \right)$

   where $\sigma(x)$ is the sigmoid function.

   $P(v_i | h) = \mathcal{N}\left(\sum_j w_{ij} h_j + b_i, \sigma^2 \right)$

   where $\mathcal{N}(\mu,\sigma^2)$ is a Gaussian distribution with mean $\mu$ and variance $\sigma^2$.

5. Explain in simple terms the $CD\text{-}1$ contrastive divergence algorithm for training an RBM.
6. Once an RBM is trained, how can it be used to generate new data?

## Part 2: Practical Use of an RBM to Generate New Data

Use the provided `Gaussian_Bernoulli_RBM_CEE4803_Spring2025.ipynb` code to analyze the following datasets:

1. Olivetti Faces
2. CIFAR-10
3. Art images (Still Life dataset: `StillLife64x64.npy`; other datasets: `VanGogh64x64.npy`, `Modigliani_paintings.npy`)

### Tasks

- Train the RBM on each dataset and generate new images. The number of hidden neurons is set to $N_{\text{hidden}} = 64$ or $32$, depending on the case.
- Plot the columns of the weight matrix as images. What features are stored in the weight matrix?
- Repeat the training for the Olivetti Faces and Still Life datasets with a very low number of hidden variables ($N_{\text{hidden}} = 8$). How does this affect the quality of the generated images and the learned features?
- Repeat the training for the Olivetti Faces and Still Life datasets with a very low number of epochs, say $50$. How does this affect the quality of the generated images and the learned features?
- Plot the probability density distributions for all datasets. Are they unimodal or bimodal? Broad or sharp? Explain why.

## Part 3: Learning Abstract and Monet Paintings with an RBM

Using the provided Jupyter Notebook `VanGogh_RBM_CEE4803_Spring2025.ipynb`, train an RBM on the following datasets:

1. Abstract paintings (Select your own set of paintings using cosine similarity. Data set `AbstractPaintings_Dataset.npy`)
2. Monet paintings (Select your own set of paintings using cosine similarity. Data set `Monet64x64.npy`)

### Tasks

- Train the RBM and generate new images.
- Interpret the features learned by the model from these datasets.
- Repeat the training with a very low number of hidden variables ($N_{\text{hidden}} = 8$). How does this affect the quality of the generated images and the learned features?
- Plot the probability density distributions for all datasets. Are they unimodal or bimodal? Broad or sharp? Explain why.
- Repeat the training for one of the datasets with a very low number of epochs, say $50$. How does this affect the quality of the generated images and the learned features?

---

<div align="center">
   
[🏠 Home](/README.md)

</div>
