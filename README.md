<!-- Written by Alex Jenkins and Dr. Francesco Fedele -->

<div align="center">

# AI Fundamentals

## Written by: [Dr. Francesco Fedele](https://scholar.google.com/citations?user=iaHIkTAAAAAJ) & [Kenneth (Alex) Jenkins](https://alexj.io)

<img src="./aibasics/Figures/AI_Fedele.png" alt="AI" width="400" height="400">

</div>

<div align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">

  ### Licensed Under:

  <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" style="text-decoration: none;"><img src="./aibasics/Figures/GPLV3_Logo.svg" alt="GPLv3 Logo" style="height: 50px; display: block;"></a><a href="https://www.gnu.org/licenses/fdl-1.3.html" target="_blank" style="text-decoration: none;"><img src="./aibasics/Figures/GFDL_Logo.svg" alt="GFDL Logo" style="height: 50px; display: block;"></a>

</div>


<div align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">

  ### Runs on:
  
  [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org)
  [![CUDA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)](https://nvidia.com)
  [![Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
  
</div>

<div align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">

  ### Built with:
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
  [![NumPY](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
  [![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
  [![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
  [![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
  
</div>

___


<div align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">

  [![stars - AI_Fundamentals](https://img.shields.io/github/stars/Fedele-AI/AI_Fundamentals?style=social)](https://github.com/Fedele-AI/AI_Fundamentals)
  ![](https://view-counter.tobyhagan.com/?user={Fedele-AI}/{AI_Fundamentals})
  [![forks - AI_Fundamentals](https://img.shields.io/github/forks/Fedele-AI/AI_Fundamentals?style=social)](https://github.com/Fedele-AI/AI_Fundamentals)
  
</div>


<div align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">  

  **Version: 1.1**

</div>

___

## Introduction
The term “Artificial Intelligence” is everywhere — but it’s often misunderstood. Despite the name, these systems don’t actually possess intelligence or understanding. What’s typically called “AI” today refers to models that generate outputs based on patterns in data. They don’t comprehend meaning, context, or truth; they operate by predicting what comes next in a sequence, based on what they've seen before.

This textbook is open to everyone — no background in machine learning is required. If you’re comfortable with basic Python and introductory calculus, you’ll be able to follow along and engage with every concept. Whether you're encountering AI for the first time or revisiting it with a more critical perspective, our goal is to make the inner workings of these systems clear, approachable, and honest. We focus on what these tools actually do — and just as importantly, what they can’t.

You’ll explore core models and techniques, from basic neural networks to complex architectures like transformers, paired with hands-on coding exercises. Along the way, we emphasize the limitations, risks, and philosophical questions that come with generative models. By the end, you’ll not only have the skills to build these systems — you’ll also have the language to talk about them honestly!

---

## Table of Contents
This series covers the following topics, and you are encouraged to read the modules in order to build a strong foundation in the basics of AI.

| **Module**                          | **Homework**                | **Code**                  |
|-------------------------------------|-----------------------------|---------------------------|
| [0. Preface, About, & Ethics](aibasics/about.md) |   |  |
| [1. Ising Model](aibasics/isingmodel.md)  | [ HW1 ](aibasics/Homework/ISING_homework.md)   | <ul><li>[Ising Model](aibasics/Python_Codes/Ising_model.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/Ising_model.ipynb)</li><li>[Ising Model With Intermediate Plots](aibasics/Python_Codes/Ising_model_with_intermediate_plots.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/Ising_model_with_intermediate_plots.ipynb)</li></ul> |
| [2. Linear Perceptron](aibasics/linearperceptron.md) | [HW2](aibasics/Homework/LP_homework.md)  | <ul><li>[Linear Perceptron](aibasics/Python_Codes/Linear_Perceptron.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/Linear_Perceptron.ipynb)</li></ul> |
| [3. Deep Perceptron](aibasics/deepperceptron.md) | [HW3](aibasics/Homework/DP_homework.md) | <ul><li>[Deep Perceptron](aibasics/Python_Codes/Linear_Perceptron.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/Linear_Perceptron.ipynb)</li></ul> |
| [4. Hopfield Network](aibasics/hopfieldnetwork.md) | [HW4](aibasics/Homework/HOPFIELD_homework.md) | <ul><li>[Hopfield Network](aibasics/Python_Codes/HOPFIELD_NETWORK_TRAINING.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/HOPFIELD_NETWORK_TRAINING.ipynb)</li></ul> |
| [5. Boltzmann Machine](aibasics/boltzmann.md) | [HW5](aibasics/Homework/RBM_homework.md) | <ul><li>[Gaussian-Bernoulli RBM](aibasics/Python_Codes/Gaussian_Bernoulli_RBM_CEE4803_Spring2025.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/Gaussian_Bernoulli_RBM_CEE4803_Spring2025.ipynb)</li><li>[VanGogh RBM](aibasics/Python_Codes/VanGogh_RBM_CEE4803_Spring2025.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/VanGogh_RBM_CEE4803_Spring2025.ipynb)</li><li>[Converting Images in a Numpy Array](aibasics/Python_Codes/Convert_images_in_npy_array_CEE4803_Spring2025.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/Convert_images_in_npy_array_CEE4803_Spring2025.ipynb)</li></ul> |
| [6. Normalizing Flow](aibasics/normalizingflow.md) |  | <ul><li>[Normalizing Flow Model](aibasics/Python_Codes/Normalizing_Flow_Matt_code.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/Normalizing_Flow_Matt_code.ipynb)</li></ul> |
| [7. CNN Autoencoders](aibasics/autoencoders.md) |  | <ul><li>[Convolution Autoencoder](aibasics/Python_Codes/Art_convolution_autoencoder_CEE4803_Spring2025.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/Art_convolution_autoencoder_CEE4803_Spring2025.ipynb)</li><li>[AE + RBM Hybrid](aibasics/Python_Codes/AE-RBM-CEE4803_Spring2025.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/AE-RBM-CEE4803_Spring2025.ipynb)</li><li>[Variational Autoencoder](aibasics/Python_Codes/VAE_STYLE_TRANSFER_CEE4803_Spring2025.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/VAE_STYLE_TRANSFER_CEE4803_Spring2025.ipynb)</li></ul> |
| [8. Combined Autoencoder with Boltzmann machines](aibasics/autoencoders2.md) |  | <ul><li>[AE + RBM](aibasics/Python_Codes/AE-RBM-CEE4803_Spring2025.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/AE-RBM-CEE4803_Spring2025.ipynb)</li></ul> |
| [9. Transformers (LLMs)](aibasics/transformer.md) |  | <ul><li>[LLM Transformer](aibasics/Python_Codes/LLM_Transformer_CEE4803_Spring2025.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/LLM_Transformer_CEE4803_Spring2025.ipynb)</li></ul> |
| [10. Bidirectional CNN Encoder-Decoder](aibasics/encoder_transformer_decoder.md) |  | <ul><li>[CNN Transformer](aibasics/Python_Codes/CNN-Transformer_ART-CEE4803_Spring2025.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/CNN-Transformer_ART-CEE4803_Spring2025.ipynb)</li></ul> |
| [11. CUDA](aibasics/cuda.md)  |  Optional Unit  | <ul><li>[CUDA Examples in Python](aibasics/Python_Codes/CUDA_examples.ipynb)</li><li>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fedele-AI/AI_Fundamentals/blob/main/aibasics/Python_Codes/CUDA_examples.ipynb)</li></ul> |

---

## About
This educational series has been meticulously crafted to serve a diverse audience of learners, from those taking their very first steps into artificial intelligence to those with prior exposure seeking to deepen their understanding. The curriculum follows a carefully designed progression that builds foundational knowledge while gradually introducing more complex concepts.

For beginners, we've taken special care to explain concepts clearly with intuitive examples and visualizations that make abstract ideas concrete. Meanwhile, more experienced learners will find sufficient depth and advanced material to expand their knowledge boundaries. If you already possess familiarity with certain fundamental topics, you're encouraged to navigate directly to modules that challenge your current expertise level.

This series represents our commitment to making high-quality AI education accessible to everyone, regardless of background or prior technical experience. We aim to democratize access to AI knowledge, foster critical thinking about AI's capabilities and limitations, and empower a new generation of innovators to apply these tools ethically and creatively. The interdisciplinary approach integrates perspectives from computer science, mathematics, engineering, and cognitive science to provide a comprehensive understanding of how artificial intelligence systems work and evolve.

We sincerely hope this learning journey proves valuable as you explore the fascinating world of artificial intelligence, whether your goals involve academic advancement, professional development, or personal enrichment! Your feedback is welcomed as we continuously strive to improve and expand these educational resources.

---

## License
This textbook features libre code samples and documentation. [We respect your freedom](https://www.gnu.org/philosophy/free-sw.en.html#four-freedoms). To accommodate license compatibility concerns, the project is dual-licensed under the `GPLv3` and the `GFDL 1.3`. Users are encouraged to honor the principles of free software by ensuring full compliance with both licenses when using, modifying, or sharing this material.

### Documentation License
The documentation in this repository is licensed under the **GNU Free Documentation License 1.3 (GFDL 1.3)**. This means you are free to copy, modify, and distribute this document under the terms of the GFDL 1.3, provided that you retain this notice and provide attribution.

### Code License
The code provided in this repository is licensed under the **GNU General Public License v3.0 (GPLv3)**. This means you are free to use, modify, and distribute the code, provided that any derivative works also comply with the GPLv3 terms.

#### TL;DR:
- 🤑 **This is free of charge**, if you paid money for this textbook or code - request a refund immediately.
- ✅ **You can** copy, modify, and distribute the content and code. Commercial use is allowed.
- 🚫 **You cannot** impose additional restrictions beyond the GFDL 1.3 for documentation and GPLv3 for code.
- 🏴‍☠️ **You hold harmless** the authors of these texts, and understand that there is no warranty.
- 📜 **You must** give proper attribution, include the license notice in all copies, and release any derivative works of the code under the same license.
- 👨‍⚖️ **For full details**, see [our license file](LICENSE.md).

---

> [!IMPORTANT]  
> We want your help! Whether through code, comments, ideas, or documentation - we're committed to making this textbook the best it can be.
>   
> If you'd like to contribute to this repository, please [read and accept our Contributor Code of Conduct](./CODE_OF_CONDUCT.md). Fedele_AI is dedicated to fostering a welcoming and collaborative environment for everyone, and your participation is essential to that mission.  
>   
> For issues, ideas, or questions, check out the [Discussions](https://github.com/Fedele-AI/AI_Fundamentals/discussions) tab above.  

This textbook was made with ❤️ in Atlanta, Georgia 🇺🇸 - [Go Jackets!](https://gatech.edu)

___

[![Star History Chart](https://api.star-history.com/svg?repos=Fedele-AI/AI_Fundamentals&type=Timeline)](https://www.star-history.com/#Fedele-AI/AI_Fundamentals&Timeline)
