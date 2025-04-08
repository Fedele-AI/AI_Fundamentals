

### 🔄 RBM versus VAE: Pros and Cons

You can use both Restricted Boltzmann Machines (RBMs) and Variational Autoencoders (VAEs) to generate art images — but the kind of images you’ll get, and the process you’ll follow, are quite different. Below is a breakdown to help you understand what you can expect:

---

#### 🧠 RBM (Restricted Boltzmann Machine)
**→ Good for:** low-resolution, abstract, grainy, noisy, dreamlike images

- RBMs are shallow, probabilistic models.
- They struggle with large, high-resolution, or high-dimensional color images.
- Typically work best on **28×28 grayscale images** (like MNIST).
- To use RBMs for art, you'd likely:
  - Reduce your image size (e.g. grayscale **32×32**)
  - Train on small datasets (~40–100 images)
  - Generate abstract patterns or textures
- **Output:** fragmented, stochastic images — more suggestive than literal
- Great for glitch-art, pixel art, textures, or concept exploration
- **Think of it as:** the *"unconscious dreaming"* of your dataset

🖼️ **Example result:** fuzzy blobs, hazy shapes, visual "hallucinations" resembling textures or loose compositions

---

#### 🎨 VAE (Variational Autoencoder)
**→ Good for:** more coherent, soft, blurry but structured images

- VAEs learn a **latent space** — so you can interpolate and explore creativity
- Work reasonably well on **64×64 or 128×128 images** (with a good enough architecture)
- You can:
  - Train a vanilla VAE (encoder-decoder) on art images
  - Sample from latent space (e.g. $z \sim \mathcal{N}(0, 1)$) to generate new images
  - Or interpolate between artworks
- **Output:** soft, smooth, sometimes blurry reconstructions
- **Feels like:** a visual fog where your dataset's artistic style *"lives"*

🖼️ **Example result:** impressionistic renditions of paintings, with recognizable patterns or color palettes, but not high realism


## 🔍 Summary Table: Generative Models

| Model | Input Image Size | Output Style | Strength | Weakness |
|-------|------------------|--------------|----------|----------|
| RBM   | ≤ 32×32           | Grainy, stochastic, noisy | Captures chaotic textures, good for conceptual patterns | Low fidelity, unstable training, requires large datasets |
| VAE   | ≤ 128×128         | Smooth, soft, blurry      | Structured and continuous latent space, good interpolation | Blurry reconstructions, less sharpness |
| AE + RBM | ≥ 128×128 (compressed) | More coherent than RBM alone | Combines structured latent space with generative sampling | Depends heavily on quality of encoder; training two models |
| GAN   | ≥ 256×256         | Sharp, photorealistic     | High visual quality, crisp images | Mode collapse, training instability |
| Diffusion | Any size (scalable) | Very high fidelity, progressive | State-of-the-art generation, strong diversity | Slow sampling, computationally intensive |

### ✨ Artistic Opportunities

- **RBM** = chance, abstraction, entropy  
  → Use for: **texture overlays**, **base layers**, or **glitch aesthetics**

- **VAE** = latent interpolation, generative blends  
  → Use for: **morphing artworks**, **style exploration**, or **creating unseen variations**


---

### Combining Autoencoders with Restricted Boltzmann Machines for Efficient Sampling

Training a **Restricted Boltzmann Machine (RBM)** directly on raw image data poses significant challenges due to the high dimensionality of the pixel space. For example, a 128×128 grayscale image has over 16,000 pixels, each of which would require a connection to every hidden unit. Learning a generative model in such a space requires a prohibitively large dataset to produce reliable estimates of the model parameters.

To address this, we adopt a hybrid approach that combines an **Autoencoder (AE)** with an RBM. The key idea is to use the autoencoder as a *nonlinear feature extractor*, compressing the input images into a **low-dimensional latent space**. This compact representation captures the essential features of the image in a tractable form that is easier to model probabilistically.

#### Workflow:

1. **Training the AE**:
   - The AE is trained on the original image dataset to minimize reconstruction loss (e.g., MSE).
   - Once trained, the encoder maps each high-dimensional image to a latent vector in a lower-dimensional space.

2. **Learning the latent distribution with an RBM**:
   - An RBM is trained on the latent vectors produced by the AE encoder.
   - Because the latent space is much smaller than the original image space, the number of parameters in the RBM is greatly reduced, requiring far fewer samples for training.

3. **Sampling and Decoding**:
   - New latent vectors are sampled from the trained RBM.
   - These latent vectors are passed through the AE decoder to generate new synthetic images.

#### Advantages:

- **Dimensionality Reduction**:
  Training directly in the pixel space is inefficient due to the curse of dimensionality. By compressing the data, the AE allows the RBM to operate in a space where meaningful structure can be learned from a smaller dataset.

- **Improved Estimation**:
  Estimating the parameters of an RBM is statistically equivalent to estimating the **mean and covariance** of the input distribution. In high-dimensional spaces, a small sample size leads to **wide confidence intervals**, meaning the estimates of the RBM weights are unreliable and unstable. By reducing the dimensionality of the input via the AE, we increase the **effective sample size per dimension**, leading to more confident and stable parameter estimates.

- **Generalization**:
  The AE focuses the representation on the most salient features of the data. As a result, the RBM learns a distribution over meaningful variations, improving the quality of generated samples.

- **Modularity**:
  This architecture allows decoupling of the feature learning and generative modeling stages, making it easier to adapt or improve each component independently.

#### Analogy with Statistical Estimation:

Consider estimating the **mean** of a high-dimensional Gaussian vector. If the number of samples is small relative to the number of dimensions, the estimate will have high variance, and the confidence interval will be wide. The same applies to estimating RBM weights — they represent expected correlations across dimensions. When the dimensionality is reduced by an AE, each parameter is estimated with more confidence, effectively shrinking the confidence intervals and improving reliability.

# Mathematics of Autoencoders (AE), Restricted Boltzmann Machines (RBM), and Statistical Estimators

## ✨ Autoencoder (AE) Equations

An autoencoder is a type of neural network used to learn low-dimensional representations (encodings) of data.

**Encoder**:  
Maps input $x \in \mathbb{R}^n$ to latent code $z \in \mathbb{R}^m$ using a deterministic function:

$$
z = f_\theta(x) = \sigma(W_e x + b_e)
$$

**Decoder**:  
Reconstructs input from latent code:

$$
\hat{x} = g_\phi(z) = \sigma(W_d z + b_d)
$$

**Loss Function**:  
Typically, mean squared error (MSE):

$$
\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2
$$

---

## 🔁 Restricted Boltzmann Machine (RBM) Equations

An RBM is a generative stochastic neural network that models the probability distribution of binary or continuous inputs.

**Energy Function** (for binary visible $v$ and hidden $h$ units):

$$
E(v, h) = -v^\top W h - b^\top v - c^\top h
$$

**Probability of a configuration**:

$$
P(v, h) = \frac{1}{Z} e^{-E(v, h)}
$$

where $Z = \sum_{v, h} e^{-E(v, h)}$ is the partition function.

**Marginal probability of visible vector $v$**:

$$
P(v) = \frac{1}{Z} \sum_h e^{-E(v, h)}
$$

**Training Objective**:  
Maximize the likelihood $\log P(v)$ using Contrastive Divergence.

---

## 📊 Statistical Estimators and Confidence Intervals

Estimating the weights of a **Restricted Boltzmann Machine (RBM)** is a **statistical inference** task. Each weight $w_{ij}$ connecting visible unit $i$ and hidden unit $j$ is treated as a parameter learned from data. Since RBMs are trained on small, often noisy datasets (especially in artistic applications), **confidence intervals (CIs)** and variance estimates become critical to understanding model reliability.

---

### 🧮 Sample Mean and Variance

Let $x_1, x_2, \dots, x_N$ be $N$ independent observations of some random variable $X$ (e.g., pixel intensities or activation states).

**Sample Mean**:

$$
\hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

This is our estimator for the expected value $\mathbb{E}[X]$.

**Sample Variance**:

$$
\hat{\sigma}^2 = \frac{1}{N - 1} \sum_{i=1}^{N} (x_i - \hat{\mu})^2
$$

This measures the variability of the sample — crucial when estimating the uncertainty in a model's weight.

---

### 📏 Confidence Interval (CI)

Assuming the sample mean $\hat{\mu}$ follows a normal distribution (by the Central Limit Theorem), a **95% confidence interval** for the true mean is:

$$
\hat{\mu} \pm z_{\alpha/2} \cdot \frac{\hat{\sigma}}{\sqrt{N}}
$$

Where:

- $z_{\alpha/2}$ is the critical value from the standard normal distribution (e.g., $1.96$ for 95% CI),
- $\hat{\sigma}$ is the sample standard deviation,
- $N$ is the number of samples.

---

## 🔗 Connecting RBM Weight Estimators to Confidence Intervals

Each RBM weight $w_{ij}$ connects visible unit $i$ and hidden unit $j$, and is updated using stochastic gradients over a limited dataset.

If the dataset is **small**, the estimated gradients will have **high variance**, leading to poor confidence in the learned weights.

The uncertainty in the weight estimates can be understood in terms of their **confidence intervals** — larger training sets produce narrower intervals and more reliable weights.

RBMs trained on raw high-dimensional images (e.g., $64 \times 64 = 4096$ pixels) require **enormous datasets** for statistically reliable training. This is due to the **curse of dimensionality**, where the number of parameters scales quadratically with the number of visible units.

---

### 🔗 Connection to RBM Weights

Training an RBM involves estimating probabilities such as:

$$
p(v_i = 1 \mid h_j) = \sigma(w_{ij} h_j + b_i)
$$

where $\sigma(\cdot)$ is the sigmoid function and $b_i$ is the bias for visible unit $i$.

Each weight $w_{ij}$ is updated using **stochastic gradient descent** or **Contrastive Divergence (CD-k)**. The update relies on expectations of the form:

$$
\Delta w_{ij} \propto \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}
$$

These expectations are empirically estimated — effectively sample means over mini-batches. Hence, small datasets or poor sampling lead to high-variance estimates for $w_{ij}$, making training unstable.

---

### ⚠️ Why This Matters

If the latent representation (e.g., image pixels) is too high-dimensional, we need exponentially more samples to achieve reliable estimates with narrow confidence intervals.

Using an **Autoencoder (AE)** to reduce dimensionality helps by:

- Lowering the number of parameters (weights) the RBM must learn
- Improving the reliability of each $\hat{w}_{ij}$ estimate
- Allowing smaller datasets to still yield useful generative models

This is akin to **reducing the dimensionality of your estimator space**, leading to **tighter confidence intervals** and more **stable, interpretable outputs**.

---

## 🚀 Why Combine AE + RBM?

Using an autoencoder, we can **compress** high-dimensional images into low-dimensional **latent codes**:

1. Train AE to map image $x$ to $z$.
2. Train RBM to model $P(z)$ in latent space.
3. Sample latent vectors $z'$ from RBM.
4. Decode $z'$ using the AE decoder to generate new images.

**Benefits**:
- Latent space is **low-dimensional** → fewer weights → more robust statistical estimation.
- Reduces **variance** in RBM weight updates.
- Efficient **unsupervised generative modeling** of complex data.

---

## Examples 

<div align="center">

<img src="./aibasics/Figures/RBMGenerated_STUDENT_PHOTOS128x128.png" alt="RBM generated samples" width="50%">

*FIGURE 1:RBM generated samples of faces: image size 128x128x3, 64 hidden neurons*  

</div>


<div align="center">

<img src="./aibasics/Figures/AE-RBMgeneratedsamples_STUDENT_PHOTOS128x128.png" alt="AE-RBM generated samples" width="50%">

*FIGURE 2:AE-RBM generated samples of faces: image size 128x128x3, latent space dimension 128, 64 hidden neurons*  

</div>

---

<div align="center">

<img src="./aibasics/Figures/RBM_piet.png" alt="RBM samples of Piet Mondrian artwork" width="50%">

*FIGURE 3:RBM samples of Piet Mondrian artwork: image size 128x128x3, 64 hidden neurons*  

</div>


<div align="center">

<img src="./aibasics/Figures/AE-RBMgeneratedsamples_Piet_Mondrian_geometric128x128.png" alt="AE-RBM samples of Piet Mondrian artwork" width="50%">

*FIGURE 4:AE-RBM samples of Piet Mondrian artwork: image size 128x128x3, latent space dimension 128, 64 hidden neurons*  

</div>



## 📚 References

- Hinton, G.E., & Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. *Science*.
- Fischer, A., & Igel, C. (2012). An introduction to restricted Boltzmann machines. *Progress in Pattern Recognition*.
- MacKay, D.J.C. (2003). *Information Theory, Inference, and Learning Algorithms*.


