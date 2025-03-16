<!-- Written by Alex Jenkins and Dr. Francesco Fedele for CEE4803/LMC4813 - (c) Georgia Tech, Spring 2025 -->

<div align="center">

# Hopfield Network

</div>

## Introduction
A **Hopfield network** is a type of artificial neural network used for associative memory, meaning it can recall stored patterns when given partial or noisy input. It consists of a set of neurons that are fully connected to each other, with each connection having a weight that determines how strongly one neuron influences another. The network updates its neurons based on a specific mathematical rule to minimize an "energy" function, similar to how a ball rolls downhill to find a stable resting place. When you present an input, the network iteratively adjusts itself until it settles into a stable state, ideally matching one of the stored patterns. 

This process is like a magic notebook that remembers drawings: if you give it a messy or incomplete sketch, it fills in the missing details to recreate the original image as best as it can. The key idea behind Hopfield networks is that they store memories as attractors in an energy landscape, making them useful for tasks like pattern recognition and error correction.

The Hopfield network, introduced by John Hopfield in 1982, is a recurrent neural network designed to model memory as an emergent collective behavior of neuronal networks. It provides a framework for understanding how the brain stores and recalls memories as stable states, or attractors, of a dynamical system.

Hopfield proposed that memories are retained as stable entities, or gestalts, which can be correctly recalled even when only a portion of the original information is presented. The network reconstructs missing parts of a memory, demonstrating the principle that the whole is greater than the sum of its parts.

The Hopfield network is an idealized model for associative memory. Associative memory refers to the ability of a system to recall information based on partial or incomplete inputs. Unlike traditional memory models, where information is retrieved via exact matching, associative memory enables the recovery of entire patterns from partial cues. This property is central to human cognition, where an individual can often remember the full context of a situation from a brief or fragmented reminder.

In the Hopfield model, memories are represented as binary patterns stored within the network's connections. These patterns are learned and retrieved through the network's dynamics. The retrieval process starts with an initial, possibly noisy or incomplete, state. The network iteratively adjusts the activations of its neurons, gradually converging to one of the stored memory states. This process is known as **content-addressable memory**, meaning that the memory is retrieved based on its content, not the specific input address.

The connection between Hopfield networks and associative memory lies in the way the network stores and retrieves memories. In a Hopfield network, each memory is represented by a set of neuron states that encode the binary pattern. The network is trained using Hebbian learning, where the connection weights between neurons are adjusted to reinforce the patterns stored in the network.

Once the patterns are stored, the Hopfield network exhibits the property of **auto-association**. That is, when presented with a partial or corrupted version of a stored memory, the network can retrieve the complete memory by iteratively updating its neuron states until they stabilize. The network does not require the exact input to recall the memory; it is capable of filling in the missing or corrupted parts, making it a model for how the brain may process memories.

Thus, the Hopfield network provides a mechanism for associative memory by allowing retrieval based on partial or noisy inputs. This property is particularly important in biological systems, where memories are often retrieved under imperfect or incomplete conditions, highlighting the robustness of the Hopfield network as a model of human memory.

## Mathematical Formulation
The Hopfield network is one layer of $N$ binary neurons (perceptrons) whose outputs are listed in an $(N \times 1)$ column vector $\mathbf{x} = (x_1, x_2, ..., x_N)$, where $x_i \in \{-1, 1\}$. The weights of the perceptrons are the columns of the symmetric weight matrix $\mathbf{W} = (W_{ij})$ of size $N \times N$. Since neurons do not connect to themselves, we enforce $W_{ii} = 0$ for all $i$, ensuring that the diagonal elements of $\mathbf{W}$ remain zero. The network is defined by the energy function:

$$ E = - \frac{1}{2} \sum_{i,j} W_{ij} x_i x_j = - \frac{1}{2} \mathbf{x}^T \mathbf{W} \mathbf{x}, $$

where $T$ denotes matrix transpose. The sum of each index runs implicitly from $1$ to $N$.

The dynamical evolution of the network is governed by the update rule:

$$ x_i(t+1) = \text{sgn}\left(\sum_j W_{ij} x_j(t)\right), $$

or in matrix form:

$$ \mathbf{x}(t+1) = \text{sgn}(\mathbf{W} \mathbf{x}(t)). $$

Here, $\text{sgn}(y) = 1 \cdot (y \geq 0) - 1 \cdot (y < 0)$ is the activation function. This equation describes how the state of the entire system at time $t+1$ is determined by the weighted sum of the previous states. The function $\text{sgn}(\cdot)$ ensures that each neuron state remains binary ($\pm 1$). This iterative process allows the network to evolve towards a stable configuration, ideally corresponding to a stored pattern. The attractor dynamics of the system ensure that even if the initial state is a noisy or incomplete version of a stored pattern, the network converges to the correct memory.

A fixed point in this context is a state $\mathbf{x}^*$ that remains unchanged under the update rule:

$$ \mathbf{x}^* = \text{sgn}(\mathbf{W} \mathbf{x}^*). $$

This means that once the network reaches such a configuration, it will not change in subsequent updates. Fixed points correspond to stored patterns or stable states of the system, acting as attractors for nearby initial conditions. The weight matrix $W_{ij}$ encodes the stored patterns $\{\boldsymbol{\xi}^n\}$, where each $\boldsymbol{\xi}^n = (\xi_1^n, \xi_2^n, ..., \xi_N^n)^T$ represents a stored binary pattern, $n = 1, \dots, P$, with $P$ the number of patterns.

### Pattern Storage and Retrieval
We want to impose that each stored pattern $\boldsymbol{\xi}^n$, $n = 1, \dots, P$, is a fixed point of the dynamics:

$$ \boldsymbol{\xi}^n = \text{sgn}(\mathbf{W} \boldsymbol{\xi}^n), \quad \forall n. $$

This condition is satisfied when the product $\xi_i^n \cdot \left( \sum_j W_{ij} \xi_j^n \right) = (\mathbf{\xi}^n)^T \cdot (\mathbf{W} \boldsymbol{\xi}^n) > 0$, meaning that the data $\xi_i^n$ is classified correctly. If the product is negative, the data is misclassified, and the energy:

$$ E = - \frac{1}{2} \mathbf{x}^T \mathbf{W} \mathbf{x} = - \frac{1}{2} \sum_{i,j} W_{ij} x_i x_j $$

increases. Therefore, we want the weights that minimize this energy.

The gradient of the energy with respect to the weights is:

$$ \frac{\partial E}{\partial W_{ij}} = - \frac{1}{2} x_i x_j, $$

or in matrix form:

$$ \frac{\partial E}{\partial \mathbf{W}} = - \frac{1}{2} \mathbf{x} \mathbf{x}^T, $$

where $T$ denotes transpose. Note that $\mathbf{x}$ is a column vector $(N \times 1)$, so $\mathbf{x} \mathbf{x}^T$ is an $(N \times N)$ matrix. Thus, the weight update rule is:

$$ \mathbf{W}(t+1) = \mathbf{W}(t) + \gamma \mathbf{x} \mathbf{x}^T, $$

where $\gamma/2$ is redefined as the learning rate $\gamma$. This update rule implies that when $x_i x_j > 0$, meaning that neurons $i$ and $j$ fire together (both being either $+1$ or $-1$), the rule strengthens their connection by increasing the weight $W_{ij}$. This mechanism reflects the principle that "neurons that fire together wire together."

### Hebbian Learning Rule: Neurons That Fire Together Wire Together
A simple way to define the weights is to require that each pattern $\boldsymbol{\xi}^n$ satisfies the retrieval condition above. This leads to the Hebbian learning rule:

$$ \mathbf{W} = \frac{1}{P} \sum_{n=1}^{P} \boldsymbol{\xi}^n (\boldsymbol{\xi}^n)^T, $$

or equivalently:

$$ W_{ij} = \frac{1}{P} \sum_{n=1}^{P} \xi_i^n \xi_j^n. $$

This rule follows from the principle that "neurons that fire together wire together." However, it is not optimal since it does not minimize interference between stored patterns.

The Hebbian learning rule is considered unsupervised because it adjusts the synaptic weights based on the correlation between the activity of two neurons without requiring external guidance or labels. It operates under the principle that "neurons that fire together, wire together," meaning that if two neurons are activated simultaneously, the connection between them is strengthened. This type of learning is driven by the internal dynamics of the system, not by a target output or error signal. In contrast, least squares learning is supervised because it involves adjusting the weights to minimize the difference between the network's output and a known target output, which requires a labeled dataset and an external error signal to guide the learning process.

### Least Squares Training Approach
An alternative method is to minimize the squared error function:

$$ E = \frac{1}{2P} \sum_{n=1}^{P} \left( \boldsymbol{\hat{\xi}}^n - \boldsymbol{\xi}^n \right)^T \left( \boldsymbol{\hat{\xi}}^n - \boldsymbol{\xi}^n \right), $$

where the predicted output $\boldsymbol{\hat{\xi}}^n = \text{sgn}(\mathbf{W} \boldsymbol{\xi}^n)$ for the input $\boldsymbol{\xi}^n$. Using gradient descent, the weight update rule is derived as:

$$ \mathbf{W}(t+1) = \mathbf{W}(t) - \gamma \sum_{n=1}^{P} \left( \boldsymbol{\hat{\xi}}^n - \boldsymbol{\xi}^n \right) (\boldsymbol{\xi}^n)^T, $$

where $\gamma$ is the learning rate. This rule ensures that the weight updates are driven by the difference between the desired and actual output, improving recall accuracy compared to the Hebbian rule. Least squares learning is supervised because it involves adjusting the weights to minimize the difference between the network's output and a known target output, which requires a labeled dataset and an external error signal to guide the learning process.

### Storage Capacity and Correlation Effects
The maximum number of patterns that can be stored without significant retrieval errors is approximately $0.15 N$. When patterns are uncorrelated, storage is more efficient. However, if patterns are highly correlated, interference increases, reducing the effective storage capacity and causing spurious attractors.

### Conclusion
The Hopfield network provides a simple yet powerful model of associative memory. While the Hebbian rule is intuitive, more sophisticated learning rules improve storage capacity and robustness.

## Video Explanations

### Video 1. Hopfield Network hallucinations
[![Watch the video](https://img.youtube.com/vi/xUunIk9EJVo/0.jpg)](https://www.youtube.com/watch?v=xUunIk9EJVo)
[![Watch the video](https://img.youtube.com/vi/td2RiuvfyCs/0.jpg)](https://www.youtube.com/watch?v=td2RiuvfyCs)


## References
- [Hopfield1982] J. J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities," *Proceedings of the National Academy of Sciences*, vol. 79, no. 8, pp. 2554-2558, 1982.
