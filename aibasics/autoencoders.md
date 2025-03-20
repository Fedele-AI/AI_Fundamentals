<!-- Written by Alex Jenkins and Dr. Francesco Fedele for CEE4803/LMC4813 - (c) Georgia Tech, Spring 2025 -->

<div align="center">

# Autoencoders

<img src="./Figures/Autoencoder.png" alt="Autoencoder" width="300" height="300">

</div>

An **autoencoder** is a type of neural network used for unsupervised learning that learns to compress (encode) data into a compact representation and then reconstruct (decode) it back to its original form. It consists of two main parts:
- **Encoder**: This part takes the input and maps it to a lower-dimensional **latent space**, which is a compressed representation of the data.
- **Decoder**: This part reconstructs the input from the latent space representation, aiming to recreate the original data as closely as possible.

The **latent space** is the compressed, lower-dimensional space where the essential features of the data are stored. For example, in an image autoencoder, the encoder might learn to represent an image as a smaller vector that captures key features like edges, shapes, and colors.

**Latent space interpolation** refers to the process of taking two points (representations) in the latent space and generating intermediate points between them. This is useful because the latent space is continuous, meaning small changes in the latent space can correspond to meaningful changes in the original data. By interpolating between two latent vectors, you can create new, plausible data points that blend the characteristics of both input points.

To explain it simply, imagine you're using an autoencoder to compress a picture of a dog into a smaller "summary" of the picture. The encoder learns how to represent that dog in a much simpler form. Then, you can take two "summaries" of two different dogs and mix them together (interpolate between their summaries) to create a new picture that might look like a combination of the two original dogs.

Autoencoders with latent space interpolation are often used for **generative tasks**, like creating new images, generating realistic data, or exploring the structure of the data in a more intuitive way.


## Video Notes

### Video 1. What's an Autoencoder?  
[![Watch the video](https://img.youtube.com/vi/4Dk7Kfeal5o/0.jpg)](https://www.youtube.com/watch?v=4Dk7Kfeal5o)

### Video 2. Latent Space Interpolation & Image style transfer 
[![Watch the video](https://img.youtube.com/vi/qVYRzunQiAQ/0.jpg)](https://www.youtube.com/watch?v=qVYRzunQiAQ)






