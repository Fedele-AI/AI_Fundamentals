# Homework: Training Deep Perceptrons for Boolean Gates

## PART 2: Modeling the XOR Gate with a Deep Perceptron

The XOR gate cannot be modeled by a single-layer perceptron because it is not linearly separable in the input space. In this assignment, you will:

1. Prove that XOR can be expressed using a combination of OR, AND, and NAND gates.
2. Analyze the latent (hidden) space transformation.
3. Implement a multi-layer perceptron (MLP) to model the XOR function.
4. Visualize the input and latent spaces and explain why training is possible.

### Boolean Logic Representation

Prove that the XOR gate can be represented as follows:

$z_1 = \text{OR}(x_1, x_2)$

$z_2 = \text{NAND}(x_1, x_2)$

$y = \text{AND}(z_1, z_2)$

**Question 4**: Draw the Boolean Gate architecture and verify that this combination correctly models the XOR gate by computing its output for all four possible inputs $(x_1, x_2)$.

### Visualization of Input and Latent Spaces

**Question 5**: Plot the four input points $(x_1, x_2)$ in the input space and the corresponding transformed points $(z_1, z_2)$ in the latent space. Explain why linear separation is not possible in the input space but becomes possible in the latent space.

### Multi-Layer Perceptron (MLP) Model

A two-layer (deep) perceptron can be used to model XOR as follows:

$z_1 = \sigma(w_1 x_1 + w_2 x_2 + b_1)$

$z_2 = \sigma(w_3 x_1 + w_4 x_2 + b_2)$

$y = \sigma(w_5 z_1 + w_6 z_2 + b_3)$

where $\sigma$ is an activation function such as the sigmoid or ReLU function.

**Question 6**: Draw the architecture of the deep perceptron.

**Question 7**: Train the model to learn the XOR function. Find the MLP parameters to model a XOR gate using the provided Jupyter Notebook.

**Question 8**: Plot the data points in both the input and latent spaces. Explain why the MLP successfully learns the XOR function.


---

<div align="center">
   
[🏠 Home](/README.md)

</div>
