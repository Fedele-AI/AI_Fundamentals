<!-- Written by Alex Jenkins and Dr. Francesco Fedele for CEE4803/LMC4813 - (c) Georgia Tech, Spring 2025 -->

<div align="center">

# Linear Perceptron

</div>

## Introduction
A **linear perceptron** is one of the simplest types of artificial neural networks, designed for binary classification problems — deciding whether an input belongs to one of two categories. It consists of multiple input neurons, each associated with a weight, and a single output neuron. The perceptron computes a weighted sum of the inputs and applies an activation function, typically a step function, which outputs either 0 or 1 depending on whether the sum exceeds a threshold.

To understand it simply, imagine a seesaw with different weights placed on it. If the total weight on one side is heavy enough, the seesaw tips in that direction; otherwise, it stays down. Similarly, a perceptron "tips" toward one decision or another based on the sum of weighted inputs. However, a basic linear perceptron can only classify data that is linearly separable—meaning it can be divided by a straight line. If the data is more complex, like an XOR pattern, a single perceptron won’t work, which is why multi-layer perceptrons (MLPs) with additional layers and non-linear activation functions were developed.


## Mathematical Formulation

The perceptron is a fundamental model in machine learning, originally introduced by McCulloch and Pitts in 1943. It serves as the foundation for modern artificial neural networks and is used for binary classification tasks.

A perceptron computes a weighted sum of its inputs and applies an activation function to determine the output. Mathematically, it is represented as:

$$ y = \sigma( \mathbf{w} \cdot \mathbf{x} + b ) $$

where:
- $\mathbf{w} = (w_1, w_2, ..., w_n)$ is the weight vector.
- $\mathbf{x} = (x_1, x_2, ..., x_n)$ is the input vector.
- $b$ is the bias term.
- $\sigma$ is the activation function, typically a step function for a perceptron:

$$ \sigma(z) = \begin{cases} 1, & \text{if } z \geq 0, \\ 0, & \text{otherwise} \end{cases} $$

### Dot Product Explanation
The dot product, also known as the inner product, between two vectors $\mathbf{w}$ and $\mathbf{x}$ is defined as:

$$ \mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^{n} w_i x_i $$

This operation measures the similarity between the weight vector and the input vector and determines how strongly each input influences the perceptron’s decision.

### Binary Classification Using the Perceptron
Binary classification is the process of categorizing inputs into one of two possible classes. The perceptron assigns an input to a class based on the sign of the weighted sum of inputs plus bias:

$$ \mathbf{w} \cdot \mathbf{x} + b \geq 0 \Rightarrow y = 1, \quad \text{otherwise } y = 0. $$

This creates a linear decision boundary that separates the two classes.

### Decision Boundary, Weights, and Bias
The decision boundary is a hyperplane that separates different classes in the input space. It is defined by setting the perceptron output equation to zero:

$$ \mathbf{w} \cdot \mathbf{x} + b = 0 $$

The weights $\mathbf{w}$ determine the orientation of the decision boundary, while the bias $b$ shifts it.

### Supervised Learning for an OR Gate
Supervised learning involves training a model using labeled examples. For the OR gate, we provide input-output pairs:

| $x_1$ | $x_2$ | OR Output $y$ |
|-------|-------|----------------|
| 0     | 0     | 0              |
| 0     | 1     | 1              |
| 1     | 0     | 1              |
| 1     | 1     | 1              |

The perceptron is trained to adjust $\mathbf{w}$ and $b$ such that it correctly classifies these inputs.

### Least Squares for Training the Perceptron
To train the perceptron, we minimize the least squares error:

$$ E = \sum_{i=1}^{N} (y_i - (\mathbf{w} \cdot \mathbf{x}_i + b))^2 $$

where $y_i$ is the actual label and $\mathbf{x}_i$ is the input.

The weights are updated using gradient descent:

$$ \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \eta (y_i - \hat{y_i}) \mathbf{x}_i $$

where $\eta$ is the learning rate, and $\hat{y_i}$ is the predicted output.

Similarly, the bias is updated as:

$$ b^{(t+1)} = b^{(t)} + \eta (y_i - \hat{y_i}) $$

By iterating these updates, the perceptron learns to correctly classify the OR gate inputs.

### Limitations of the Linear Perceptron
One key limitation of the perceptron is that it can only model linearly separable functions. This means that it fails to correctly classify the XOR gate, which is not linearly separable. We will need a Multi-Layer Perceptron (MLP). 


## Video Notes

### Video 1: Linear Perceptron: The Model 
[![Watch the video](https://img.youtube.com/vi/qWjrk0jrrKw/0.jpg)](https://youtu.be/qWjrk0jrrKw)

### Video 2: Linear Perceptron: The Learning Algorithm 
[![Watch the video](https://img.youtube.com/vi/koLnKPWWglE/0.jpg)](https://youtu.be/koLnKPWWglE)

### Video 3: Why The Linear Perceptron Cannot Model The XOR Gate 
[![Watch the video](https://img.youtube.com/vi/KNU9PIiSCMU/0.jpg)](https://youtu.be/KNU9PIiSCMU)

### Video 4:  Python Implementation of a Linear Perceptron 
[![Watch the video](https://img.youtube.com/vi/TJo4r1cm1y8/0.jpg)](https://youtu.be/TJo4r1cm1y8)
