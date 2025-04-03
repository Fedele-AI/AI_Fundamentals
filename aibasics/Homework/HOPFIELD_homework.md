# Homework: Hopfield Network and Associative Memory 

## Introduction

In this homework, you will explore the Hopfield network, a type of recurrent neural network, and investigate its ability to store and recall patterns. The Hopfield network is a model of associative memory, where memories can be retrieved from partial or noisy inputs. You will train a Hopfield network using both Hebbian and least squares learning rules to store various types of patterns, such as images of faces, digits, and Van Gogh paintings. You will also explore the connection between the Hopfield network's weight matrix and the stored patterns and analyze the network's behavior under different temperatures using the Metropolis algorithm.

## Tasks

### Theory Questions

1. **How does the Hopfield network store patterns?**
    - Explain the mechanism by which the Hopfield network encodes and stores patterns in the network's weight matrix.
    - Discuss why the Hopfield network is considered a model of associative memory.
    - When and how can a pattern be correctly recalled from a noisy or incomplete input?

### Practical Exercise: Running the Hopfield Network

You will run the provided Python code to implement and train a Hopfield network for storing different types of patterns. The provided code allows you to apply both Hebbian learning and least squares learning to the Hopfield network.

For each of the following datasets:

1. **Olivetti Faces**: Train the network to store several Olivetti face images.
2. **MNIST Digits**: Train the network to store images of digits from the MNIST dataset.
3. **Van Gogh Painting**: Train the network to store images of a Van Gogh painting.

For each dataset, answer the following:

- Determine the maximum number of patterns the network can memorize using both Hebbian learning and least squares learning. Note that each learning approach leads to a different number of patterns the network can store correctly.
- Provide the initial patterns used to train the network.
- Show a plot of the two trained weight matrices using the two learning approaches and compare their structure.
- Test if the images are correctly recalled by the network. Provide plots of the original, initial noisy, and recalled patterns. Provide a case when the patterns are recalled correctly and when they are not.

### Discussion of Weight Matrix Columns

Discuss the columns of the trained weight matrix as they correspond to the stored patterns. What do these columns represent, and how do they relate to the patterns stored in the network? How do they differ when the network correctly stores the patterns (correct recall) and when it does not?

### Metropolis Algorithm (for Olivetti faces only)

For the Olivetti faces dataset, apply the Metropolis algorithm to the Hopfield network at two different temperatures:

- Low temperature: $\beta = \frac{1}{k_B T} = 5$
- High temperature: $\beta = \frac{1}{k_B T} = 0.005$

Where:

- $k_B$ is the Boltzmann constant.
- $T$ is the temperature.

For each temperature, generate a sample from the network using the Metropolis algorithm. Compare the results and discuss the following:

- At low temperature (high $\beta$), the Boltzmann distribution $p(\mathbf{x}) \sim \exp(-\beta E)$ favors the most likely states, which correspond to the local minima (fixed points) of the energy function. What does this mean for the behavior of the network?
- At high temperature (low $\beta$), the network explores a broader set of possible states, because all states have the same probability to occur, i.e., $p(\mathbf{x})$ is uniform. How does this affect the network's ability to recall patterns?

### Conclusion

Summarize your findings from the experiments. Discuss the limitations of the Hopfield network in terms of the number of patterns it can store and recall. How do the two learning approaches (Hebbian and least squares) compare in terms of performance? What insights did you gain from the Metropolis sampling at different temperatures?

## References

- Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*, 79(8), 2554-2558.

---

<div align="center">
   
[🏠 Home](/README.md)

</div>
