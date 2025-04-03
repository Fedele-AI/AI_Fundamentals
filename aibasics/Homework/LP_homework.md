# Homework: Training Linear for Boolean Gates

## Training Algorithm for a Linear Perceptron

A perceptron models a binary classification problem using the function:

$y = \text{sgn}(\textbf{w} \cdot \textbf{x} + b)$

where $\textbf{x}=(x_1,x_2)$ is the binary input vector, $\textbf{w}=(w_1,w_2)$ is the weight vector, $b$ is the bias, and $y$ is the predicted binary output.

Given $N$ data pair $(\textbf{x}_n, y_n)$ of input vectors $\textbf{x}_n$ and associated labels $y_n$, the perceptron can be trained using gradient descent on the energy function:

$E = - \sum_{n=1}^N y_n \cdot (\textbf{w} \cdot \textbf{x}_n + b)$

The energy $E$ measures when a data pair is misclassified. This occurs when $y_n y_n^{(pred)}<0$, where $y_n^{(pred)}=\textbf{w} \cdot \textbf{x}_n + b$ is the predicted output of the perceptron. Then the energy $E$ increases in value. So, we want to find the parameters $(\textbf{w},b)$ that minimize the energy, or loss function $E$.

The gradients of $E$ with respect to $\textbf{w}$ and $b$ are:

$\frac{dE}{d\textbf{w}} = - y_n \textbf{w}_n, \quad \frac{dE}{db} = - y_n$

The weight and bias updates are performed using:

$\textbf{w}(t+1) = \textbf{w}(t) - \gamma \frac{dE}{d\textbf{w}} = w(t) + \gamma y_n \textbf{w}_n$

$b(t+1) = b(t) - \gamma \cdot \frac{dE}{db} = b(t) + \gamma y_n$

where $\gamma$ is the learning rate and $t$ is the epoch, or iteration $t$. The parameter update is applied only when a data misclassification occurs:

$y_n (\textbf{w} \cdot \textbf{x}_n + b) < 0$

This ensures that all data points are classified correctly.

### PART 1: Training the Linear Perceptron for the OR, AND and NAND Gates

For example, the OR gate can be represented by the dataset in Table 1.

| $x_1$ | $x_2$ | $y$ (Output) |
|-------|-------|--------------|
| -1    | -1    | -1           |
| -1    |  1    |  1           |
|  1    | -1    |  1           |
|  1    |  1    |  1           |

*Table 1: Truth table for OR gate*

**Question 1**: Use the Jupyter notebook included in the HW and train a perceptron to model the OR, AND and NAND gates. For each Gate/Perceptron, list the data in a table, plot the data points in the input space and explain why the perceptron can do the binary classification (hint: observe that a single line (decision boundary) can separate the positive from the negative outputs.)

### Modify the Training Algorithm to use a Least Squares Approach

**Question 2**: Write a new Python code by modifying the given code to implement a least squares training using the loss function:

$E = \frac{1}{2} \sum_{n=1}^N (y_n - \hat{y}_n)^2$

where $\hat{y}_n = \text{sgn}(\textbf{w} \cdot \textbf{x}_n + b)$.

The parameter updates are given by:

$\textbf{w}(t+1) = \textbf{w}(t) - \gamma \frac{dE}{d\textbf{w}}$

$b(t+1) = b(t) - \gamma \cdot \frac{dE}{db}$

where $\gamma$ is the learning rate. The derivatives easily follow as:

$\frac{dE}{d\textbf{w}}=(y_n - \hat{y}_n)\textbf{x}_n, \quad \frac{dE}{d b}=(y_n - \hat{y}_n) y_n$

Then,

$\textbf{w}(t+1) = \textbf{w}(t) - \gamma (y_n - \hat{y}_n)\textbf{x}_n$

$b(t+1) = b(t) - \gamma (y_n - \hat{y}_n) y_n$

Note that the update is done for any data set. The code already includes the needed changes as comments. Train the linear perceptron for the OR gate and verify that the training algorithm converges and can learn the data.

### Train the Linear Perceptron for the XOR Gate

**Question 3**: Use the provided Jupyter notebook, train a perceptron for the XOR gate using the following truth table:

| $x_1$ | $x_2$ | $y$ (Output) |
|-------|-------|--------------|
| -1    | -1    | -1           |
| -1    |  1    |  1           |
|  1    | -1    |  1           |
|  1    |  1    | -1           |

*Table 2: Truth table for XOR gate*

Why does the perceptron fail to train for the XOR gate?

- Plot the data points in the input space.
- Observe that no single line (decision boundary) can separate the positive from the negative outputs.

---

<div align="center">
   
[🏠 Home](/README.md)

</div>
