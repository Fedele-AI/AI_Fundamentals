<!-- Written by Alex Jenkins and Dr. Francesco Fedele for CEE4803/LMC4813 - (c) Georgia Tech, Spring 2025 -->

<div align="center">

# Bidirectional CNN Encoder - CNN Decoder for Art Image Sequence Prediction

<img src="./Figures/DNN2.png" alt="Autoencoder" width="40%">

</div>


## Model Overview and Function

The **Bidirectional CNN Encoder-Transformer-CNN Decoder** is a neural network architecture designed to predict a sequence of $K$ art images from an input of $N$ sequences of images. This model integrates Convolutional Neural Networks (CNNs) in both the encoder and decoder, leveraging their spatial feature extraction capabilities, alongside a transformer for sequence generation. It is tailored for generative art tasks, such as creating a series of paintings from sequences of sketches or style references.

- **Input**: $N$ sequences, each represented as $S_n = [s_{n1}, s_{n2}, ..., s_{nT}]$, where $n = 1, 2, ..., N$, $T$ is the sequence length (e.g., number of images), and each $s_{nt}$ is an image tensor of shape $(H, W, C)$ (height, width, channels).
- **Output**: A sequence of $K$ art images, denoted $[o_1, o_2, ..., o_K]$, where each $o_k$ is an image tensor of shape $(H, W, C)$, reflecting artistic patterns derived from the input sequences.
- **Function**: The model maps $N$ image sequences to $K$ output images by:
  1. Encoding each $S_n$ into feature maps using a bidirectional CNN encoder.
  2. Aggregating the $N$ encoded sequences into a context $C$.
  3. Generating $K$ feature representations autoregressively with a transformer decoder.
  4. Decoding these representations into images using a CNN decoder.

## How CNNs Work

Convolutional Neural Networks (CNNs) are specialized for processing grid-like data, such as images, by extracting spatial features through convolution operations.

- **Convolution Operation**: 
  - A filter (e.g., $3 \times 3$ kernel) slides over the input image, computing a dot product between the filter weights and local image patches, producing a feature map:
    $F(i,j) = \sum_{m,n} I(i+m, j+n) \cdot K(m,n)$
    where $I$ is the input image, $K$ is the kernel, and $F$ is the feature map.
  - Multiple filters detect different features (e.g., edges, textures).

- **Layers**:
  - **Convolutional Layer**: Applies convolution with learnable filters, followed by an activation function (e.g., ReLU: $f(x) = \max(0, x)$) to introduce nonlinearity.
  - **Pooling Layer**: Reduces spatial dimensions (e.g., max pooling takes the maximum value in a region), decreasing computation and preventing overfitting.
  - **Fully Connected Layer**: Optional, flattens feature maps into vectors for classification or regression (not used here for spatial outputs).

- **Advantages**: CNNs exploit local spatial correlations and translation invariance, making them ideal for encoding raw art images into feature representations and decoding features back into images.

## Overall Model Architecture

The architecture comprises a **Bidirectional CNN Encoder**, a **Transformer Decoder**, and a **CNN Decoder**.

### 1. Bidirectional CNN Encoder

- **Input**: $N$ sequences of images, $S_n = [s_{n1}, s_{n2}, ..., s_{nT}]$, each $s_{nt} \in \mathbb{R}^{H \times W \times C}$.
- **Architecture**:
  - **CNN Backbone**: A shared CNN processes each image $s_{nt}$:
    - Conv Layer 1: $64$ filters, $3 \times 3$ kernel, stride 1, ReLU, output: $(H, W, 64)$.
    - Max Pooling: $2 \times 2$, stride 2, output: $(H/2, W/2, 64)$.
    - Conv Layer 2: $128$ filters, $3 \times 3$ kernel, stride 1, ReLU, output: $(H/2, W/2, 128)$.
    - Flatten: Reshape to $D = (H/2) \cdot (W/2) \cdot 128$ per image.
- **Bidirectional Processing**: 
  - Each sequence $S_n$ is processed as a time series of feature vectors.
  - A Bidirectional LSTM (BiLSTM) operates over the sequence:
    - Forward: $h_{nt} = \text{LSTM}\_\text{fwd}(f(s_{nt}), h_{n,t-1})$
    - Backward: $h_{nt} = \text{LSTM}\_\text{bwd}(f(s_{nt}), h_{n,t+1})$
    - Combined: $h_{nt} = [h_{nt}; h_{nt}] \in \mathbb{R}^{2H}$, where $H$ is the LSTM hidden size.
  - Output: $H_n \in \mathbb{R}^{T \times 2H}$ per sequence.
  - **Aggregation**: Attention or pooling over $H_n$ across $N$ sequences yields context $C \in \mathbb{R}^{T' \times 2H}$.
- **Purpose**: Extracts spatial features with CNNs and captures bidirectional sequence context.

### 2. Transformer Decoder

- **Input**: Context $C$ and a start token; generates $K$ steps autoregressively.
- **Architecture**:
  - **Embedding Layer**: Maps previous outputs (or start token) to $D''$-dimensional vectors.
  - **Relative Positional Encoding**: Adds biases based on $k - k'$ distances.
  - **Transformer Layers**: 
    - Masked self-attention: $Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V$, with causal mask $M$.
    - Cross-attention to $C$.
    - Feed-forward network with ReLU.
  - Output: $z_k \in \mathbb{R}^{D''}$ per step $k$.
- **Purpose**: Generates $K$ feature representations sequentially.

### 3. CNN Decoder

- **Input**: $z_k$ from the transformer.
- **Architecture**:
  - **Fully Connected Layer**: Reshapes $z_k$ to $(H/2, W/2, 128)$.
  - **Deconv Layer 1**: $64$ filters, $3 \times 3$ kernel, stride 1, ReLU, output: $(H/2, W/2, 64)$.
  - **Upsampling**: $2 \times 2$, output: $(H, W, 64)$.
  - **Deconv Layer 2**: $C$ filters, $3 \times 3$ kernel, stride 1, sigmoid, output: $(H, W, C)$.
- **Output**: $o_k \in \mathbb{R}^{H \times W \times C}$, a full art image.
- **Purpose**: Reconstructs images from transformer features.

## Why Bidirectional Masking and Relative Positioning?

### Bidirectional vs. Causal Masking

- **Bidirectional Masking**:
  - **In Encoder**: CNN features are fed to a BiLSTM, allowing $h_{nt}$ to attend to all $s_{n1}, ..., s_{nT}$. Unmasked attention in a transformer alternative would be:
    $Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
  - **Why**: Captures full context (past and future) of art sequences, e.g., understanding a style’s evolution across $S_n$.

- **Ca Masking**:
  - **In Decoder**: Causal mask ensures $o_k$ depends only on $o_1, ..., o_{k-1}$:
    $Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V$.
  - **Why Not in Encoder**: Limits context to past only, missing future artistic cues.

- **Rationale**: Bidirectional masking in the encoder leverages CNN-extracted features holistically, unlike forward masking, which suits the decoder’s generation.

### Relative vs. Absolute Positioning

- **Relative Positioning**:
  - **In Both**: Adds $R_{ij}$ based on $i - j$ to attention, focusing on image relationships.
  - **Why**: Emphasizes transitions (e.g., style shifts), adaptable to varying $T$ and $K$.

- **Absolute Positioning**:
  - **Alternative**: Fixed $PE(pos, i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$.
  - **Why Not**: Rigid indices ignore relational dynamics critical for art.

- **Rationale**: Relative positioning aligns with artistic sequence coherence over absolute’s static order.

## Summary

The CNN-based bidirectional encoder and decoder, paired with a transformer, use bidirectional masking and relative positioning to leverage spatial and relational context, outperforming forward masking and absolute positioning for art sequence generation.

## Video Notes

### Video 1. What's an Autoencoder?  
[![Watch the video](https://img.youtube.com/vi/4Dk7Kfeal5o/0.jpg)](https://www.youtube.com/watch?v=4Dk7Kfeal5o)

### Video 2. Latent Space Interpolation & Image style transfer 
[![Watch the video](https://img.youtube.com/vi/qVYRzunQiAQ/0.jpg)](https://www.youtube.com/watch?v=qVYRzunQiAQ)

## References and Further Reading

**Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.**  
"Attention Is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.  
[DOI:10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762).

**Zhu, Y., & Zhang, H.** "BiO-Net: Learning Recurrent Bi-directional Connections for Encoder-Decoder Architecture." *arXiv*, 2020. [DOI:10.48550/arXiv.2007.00243](https://doi.org/10.48550/arXiv.2007.00243).


---

<div align="center">

[⬅️ Previous](transformer.md) | [🏠 Home](/README.md) | [Next ➡️](cuda.md)

</div>
