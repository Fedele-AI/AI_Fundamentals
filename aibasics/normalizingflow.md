<!-- Written by Alex Jenkins and Dr. Francesco Fedele for CEE4803/LMC4813 - (c) Georgia Tech, Spring 2025 -->

<div align="center">

# Normalizing Flow

</div>

## Introduction
A Normalizing Flow is a powerful machine-learning technique used to model complex probability distributions by transforming simple distributions (like a Gaussian distribution) through a series of **invertible transformations**. These flows are especially useful for **density estimation**, **generative modeling**, and **sampling** from complex datasets, such as images, audio, or text.  

At its core, a normalizing flow allows us to map **complex data** (like a picture of a cat) to a **simple space** (like a Gaussian distribution) and vice versa—enabling both **generation** and **likelihood estimation** of data points.

---

### **How Normalizing Flows Work**  

1. **Basic Idea**:  
   - You start with a **simple distribution** (e.g., a Gaussian) and apply a series of **invertible, smooth transformations** to produce a **complex output distribution**.  
   - You can also **reverse** these transformations to map the data back to the simple distribution.  

2. **Mathematical Formulation**:  
If $x$ is the observed data and $z$ is a latent variable (from a simple distribution), we define a flow:

$$z\ = f(x) \quad \text{and} \quad x = f^{-1}(z)\$$

Using the **change of variables formula**, we can compute the probability density:

$$p(x) = p(z) \left| \det \left( \frac{\partial f}{\partial x} \right) \right|^{-1}\$$

Where:
- $p(x)$ is the probability density of the data.  
- $p(z)$ is the density of the simple distribution.  
- $f$ is the transformation function.  

3. **Flow Layers**:  
A normalizing flow is composed of multiple transformation layers, where each layer applies a small, simple invertible function. The composition of many layers enables the model to learn highly complex patterns.

### **Why Normalizing Flows Are Useful**  

1. **Exact Likelihood Computation**: You can precisely compute how likely a given data point is by tracking how the distribution is transformed.  
2. **Efficient Sampling**: Once trained, you can generate new samples by drawing from the simple distribution and applying the learned transformations.  
3. **Flexibility**: Normalizing flows can model very **complex data** (e.g., high-resolution images) while maintaining **mathematical rigor**.  

### **Types of Normalizing Flows**  

1. **Affine Coupling Layers**: Divide the data and transform one part while keeping the other fixed (used in RealNVP).  
2. **Autoregressive Flows**: Transform data sequentially, where each dimension depends on the previous ones (e.g., MAF, IAF).  
3. **Neural ODE Flows**: Use continuous-time dynamics for flows (e.g., FFJORD).  

### **Real-World Applications**  

- **Generative Models**: Creating new images, audio, or text (e.g., in **NICE**, **RealNVP**, and **Glow** architectures).  
- **Anomaly Detection**: Identifying rare, unusual patterns by measuring how likely they are under the learned distribution.  
- **Bayesian Inference**: Sampling complex posterior distributions in scientific applications.  

Imagine you have a simple balloon shape (a Gaussian distribution) and you want to twist and stretch it to look like a **cat-shaped balloon** (a complex dataset). **Normalizing flows** are a series of mathematical steps that let you twist and stretch that balloon in a precise, invertible way.

By learning these transformations, we can:
1. **Generate new cat balloons** (create new data).  
2. **Understand how common or rare a cat balloon is** (estimate probabilities).  

It's like having a magic recipe to turn simple shapes into detailed objects and back again—helping AI better understand and create the world!
