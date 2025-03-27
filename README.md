<!-- Written by Alex Jenkins and Dr. Francesco Fedele -->

<div align="center">

# AI Fundamentals

### Written by: [Dr. Francesco Fedele](https://scholar.google.com/citations?user=iaHIkTAAAAAJ) & [Kenneth (Alex) Jenkins](https://alexj.io)

<img src="./aibasics/Figures/AI_Fedele.png" alt="AI" width="400" height="400">

</div>

<div align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank">
    <img src="./aibasics/Figures/GPLV3_Logo.svg" alt="GPLv3 Logo" style="height: 50px; display: block;">
  </a>

  <a href="https://www.gnu.org/licenses/fdl-1.3.html" target="_blank">
    <img src="./aibasics/Figures/GFDL_Logo.svg" alt="GFDL Logo" style="height: 50px; display: block;">
  </a>
</div>

<div align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <a href="https://www.python.org" target="_blank">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python Logo">
  </a>

  <a href="https://pytorch.org" target="_blank">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white" alt="Pytorch Logo">
  </a>
</div>

</div>

> [!CAUTION]
> This project is an active work in progress. Please check back later for our first release TBA!

___

## Introduction
Artificial Intelligence (AI) is an exciting and rapidly evolving field of computer science that focuses on building systems capable of intelligent behavior. From recognizing speech and images to translating languages and making complex decisions, AI systems can learn from data, identify patterns, and perform tasks that typically require human intelligence.

Whether you're a student diving into AI for the first time or someone who's been around since the early days of computing, this course is designed to guide you through the core concepts and practical techniques in modern AI. We aim to make these topics accessible and engaging!

Throughout this series, you'll explore foundational models, mathematical formulas, and real world examples; from simple perceptrons to cutting-edge deep learning architectures like transformers. Each module is paired with hands-on coding exercises to help you reinforce what you learn and build your own intelligent systems. By the end of this course, you'll have a solid understanding of AI fundamentals and the tools to create and experiment with intelligent algorithms and applications.

---

## Table of Contents
This series covers the following topics, and you are encouraged to read the modules in order to build a strong foundation in the basics of AI.

| **Module**                          | **Homework**                | **Code**                  |
|-------------------------------------|-----------------------------|---------------------------|
| [0. Preface, About, & Ethics](aibasics/about.md) |   |  |
| [1. Ising Model](aibasics/isingmodel.md)  | [ HW1 ](aibasics/Homework/ISING_homework.md)   | [Ising Model](aibasics/Python_Codes/Ising_model.ipynb), [Ising Model With Intermediate Plots](aibasics/Python_Codes/Ising_model_with_intermediate_plots.ipynb)               |
| [2. Linear Perceptron](aibasics/linearperceptron.md) | [HW2](aibasics/Homework/LP_homework.md)  | [Linear Perceptron](aibasics/Python_Codes/Linear_Perceptron.ipynb)             |
| [3. Deep Perceptron](aibasics/deepperceptron.md) | [HW3](aibasics/Homework/DP_homework.md)      | [Deep Perceptron](aibasics/Python_Codes/Linear_Perceptron.ipynb)             |
| [4. Hopfield Network](aibasics/hopfieldnetwork.md) | [HW4](aibasics/Homework/HOPFIELD_homework.md) | [Hopfield Code](aibasics/Python_Codes/HOPFIELD_NETWORK_TRAINING.ipynb)    |
| [5. Boltzmann Machine](aibasics/boltzmann.md) | [HW5](aibasics/Homework/RBM_homework.md)  | [Gaussian-Bernoulli RBM](aibasics/Python_Codes/Gaussian_Bernoulli_RBM_CEE4803_Spring2025.ipynb), [VanGogh RBM](aibasics/Python_Codes/VanGogh_RBM_CEE4803_Spring2025.ipynb), [Converting Images in a Numpy Array](aibasics/Python_Codes/Convert_images_in_npy_array_CEE4803_Spring2025.ipynb)    |
| [6. Normalizing Flow](aibasics/normalizingflow.md) | [ ](aibasics/homework/)    | [Normalizing Flow](aibasics/Python_Codes/Normalizing_Flow_Matt_code.ipynb)            |
| [7. CNN Autoencoders](aibasics/autoencoders.md) | [ ](aibasics/homework/) | [Convolution Autoencoder](aibasics/Python_Codes/Art_convolution_autoencoder_CEE4803_Spring2025.ipynb)  |
| [8. Transformers (LLMs)](aibasics/transformer.md)| [ ](aibasics/homework/)   | [LLM Transformer](aibasics/Python_Codes/LLM_Transformer_CEE4803_Spring2025.ipynb)   |
| [9. Bidirectional CNN Encoder-Decoder](aibasics/encoder_transformer_decoder.md) | [ ](aibasics/homework/) | [CNN Transformer](aibasics/Python_Codes/CNN-Transformer_ART-CEE4803_Spring2025.ipynb) |
| [10. CUDA](aibasics/cuda.md)  |  Optional Unit  | [CUDA Examples in Python](aibasics/Python_Codes/CUDA_examples.ipynb)                            |

---

## About
This educational series has been meticulously crafted to serve a diverse audience of learners, from those taking their very first steps into artificial intelligence to those with prior exposure seeking to deepen their understanding. The curriculum follows a carefully designed progression that builds foundational knowledge while gradually introducing more complex concepts.

For beginners, we've taken special care to explain concepts clearly with intuitive examples and visualizations that make abstract ideas concrete. Meanwhile, more experienced learners will find sufficient depth and advanced material to expand their knowledge boundaries. If you already possess familiarity with certain fundamental topics, you're encouraged to navigate directly to modules that challenge your current expertise level.

This series represents our commitment to making high-quality AI education accessible to everyone, regardless of background or prior technical experience. We aim to democratize access to AI knowledge, foster critical thinking about AI's capabilities and limitations, and empower a new generation of innovators to apply these tools ethically and creatively. The interdisciplinary approach integrates perspectives from computer science, mathematics, engineering, and cognitive science to provide a comprehensive understanding of how artificial intelligence systems work and evolve.

We sincerely hope this learning journey proves valuable as you explore the fascinating world of artificial intelligence, whether your goals involve academic advancement, professional development, or personal enrichment! Your feedback is welcomed as we continuously strive to improve and expand these educational resources.

---

## License
This textbook contains code samples and documentation. Due to license incompatibilities, this project is dual-licensed under the GPLv3 and the GFDL 1.3. Please ensure compliance with both licenses when using, modifying, or distributing this material.

### Documentation License 📃
The documentation in this repository is licensed under the **GNU Free Documentation License 1.3 (GFDL 1.3)**. This means you are free to copy, modify, and distribute this document under the terms of the GFDL 1.3, provided that you retain this notice and provide attribution.

### Code License 👾
The code provided in this repository is licensed under the **GNU General Public License v3.0 (GPLv3)**. This means you are free to use, modify, and distribute the code, provided that any derivative works also comply with the GPLv3 terms.

#### TL;DR:
- 🤑 **This is free of charge**, if you paid money for this textbook or code - request a refund immediately.
- ✅ **You can** copy, modify, and distribute the content and code.
- 🚫 **You cannot** impose additional restrictions beyond the GFDL 1.3 for documentation and GPLv3 for code.
- 📜 **You must** give proper attribution, include the license notice in all copies, and release any derivative works of the code under the same license.
- 👨‍⚖️ **For full details**, see [our license file](LICENSE.md).

---

> [!IMPORTANT]
> If you're wanting to contribute to this repository, [you'll need to read and accept our contributor code of conduct](./CODE_OF_CONDUCT.md). Fedele_AI is committed to creating a welcoming and collaborative environment for everyone, and your participation is a crucial part of that! For issues, ideas, or questions - please see the `Discussions` tab above. Thanks, and [Go Jackets](https://gatech.edu)!
