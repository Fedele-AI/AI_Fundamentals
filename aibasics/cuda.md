<!-- Written by Alex Jenkins and Dr. Francesco Fedele for CEE4803/LMC4813 - (c) Georgia Tech, Spring 2025 -->

<div align="center">

# CUDA Fundamentals

</div>

> [!NOTE]
> CUDA is a complex topic to master, but it has been instrumental in driving the current AI revolution! Below is a high-level overview of CUDA, designed to help you get started with the topic. While CUDA programming is typically done in C/C++, which falls outside the scope of this series, understanding its core concepts is crucial for anyone working in AI and high-performance computing.

## Introduction

**CUDA** (Compute Unified Device Architecture) is a platform developed by [NVIDIA](https://www.nvidia.com) for running programs on GPUs, allowing us to speed up computations by harnessing the parallel processing power of the GPU. While coding directly in CUDA can be complex and requires knowledge of low-level GPU programming (usually with C or C++), we can use CUDA with Python through various high-level libraries like **CuPy**, **TensorFlow**, **PyTorch**, and **NumPy**, which abstract away much of the complexity.

When using CUDA with Python:
1. **High-Level Libraries**: Python libraries like **CuPy** are designed to take advantage of CUDA. These libraries allow you to write code almost the same way you'd write normal Python code, but under the hood, they optimize computations for the GPU. They let you use familiar functions (like array manipulations in NumPy) but accelerate them by running in parallel on the GPU instead of the CPU.
   
2. **Data Transfer**: One key aspect of using CUDA with Python is data transfer between the CPU and GPU. To perform calculations on the GPU, we need to send the data from the CPU’s memory to the GPU’s memory. Once the computation is done, the result is transferred back to the CPU’s memory. High-level libraries handle this process for you, so you can focus on your algorithm, not on managing memory manually.

3. **Kernels and Parallelization**: Even though you don't write kernels (functions that run on the GPU) directly in CUDA, high-level libraries like PyTorch or CuPy create and launch kernels automatically when you perform operations like matrix multiplication, element-wise operations, etc. They handle the parallelization behind the scenes, allowing the GPU to process multiple pieces of data simultaneously.

4. **Performance Gains**: With CUDA, Python can access the massive parallelism of GPUs, where thousands of simple tasks can be executed at once. This makes tasks like deep learning, numerical simulations, and large matrix operations run **significantly faster** than on CPUs.

To understand why CUDA is so much faster, it's important to understand the architecture of **CPUs vs GPUs**:
- **CPUs**: A CPU (Central Processing Unit) is designed for tasks that require **sequential** processing. It has a few powerful cores optimized for executing complex tasks, one step at a time. CPUs excel at tasks where quick decision-making and high single-threaded performance are necessary.
  
- **GPUs**: A GPU (Graphics Processing Unit) is designed for tasks that can be executed in **parallel**. It contains **thousands of small, efficient cores** that can execute the same operation on many pieces of data at once. This makes GPUs much more suitable for problems like matrix operations, image processing, and training machine learning models, where many calculations are similar and can be done in parallel.

When using CUDA in Python, you are essentially enabling your Python code to take advantage of the GPU’s architecture, performing computations across many cores simultaneously. Tasks like matrix multiplication, which involve applying the same operation across a large set of data, are handled much faster by the GPU’s parallel cores than by the CPU’s sequential ones.

Imagine you're trying to solve a giant jigsaw puzzle. If you have one person (the CPU), they have to go piece by piece, which takes time. But if you have hundreds or thousands of people (the GPU), each person can work on their own piece of the puzzle simultaneously, and you get the entire puzzle done much faster.

In the world of Python programming, CUDA helps us use the "many hands" of the GPU to solve big problems more efficiently. While it may not always be necessary to write the low-level CUDA code yourself, understanding how CUDA works (and using Python libraries that leverage it) allows you to take full advantage of the power of GPUs. This is why deep learning models and scientific computing tasks run much faster with GPUs compared to CPUs.

## Video Explanations


### Video 1. How CUDA works in 100 Seconds (by Fireship)
[![Watch the video](https://img.youtube.com/vi/pPStdjuYzSI/0.jpg)](https://www.youtube.com/watch?v=pPStdjuYzSI)

### Video 2. Mythbusters Demo GPU Versus CPU (by NVIDIA)
[![Watch the video](https://img.youtube.com/vi/-P28LKWTzrI/0.jpg)](https://youtu.be/-P28LKWTzrI)

### Video 3. Intro to CUDA (by NVIDIA)
[![Watch the video](https://img.youtube.com/vi/IzU4AVcMFys/0.jpg)](https://www.youtube.com/watch?v=IzU4AVcMFys)
