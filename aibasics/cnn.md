<!-- Written by Alex Jenkins and Dr. Francesco Fedele for CEE4803/LMC4813 - (c) Georgia Tech, Spring 2025 -->

<div align="center">

# Convolutional Neural Network (CNN)

</div>

## Introduction
A **Convolutional Neural Network (CNN)** is a specialized type of neural network designed primarily for **image processing**, although it can also be applied to other types of spatial data (e.g., audio, video). It uses a technique called **convolution** to automatically extract **features** (like edges, shapes, and patterns) from input images, reducing the need for manual feature engineering.  

CNNs are the foundation behind modern computer vision tasks like **image classification, object detection, facial recognition**, and **medical image analysis**. They are widely used in industry and research due to their ability to learn complex patterns and generalize well to new data.

### **How CNNs Work**  

A typical CNN consists of several key layers:

1. **Convolutional Layer**  
   - **Purpose**: This is the core building block where the model scans the input (e.g., an image) with small filters (or kernels) to detect local features.  
   - **How it works**: A **filter** slides (or convolves) over the input, multiplying pixel values by the filter's weights and summing them to produce a **feature map**.  
   - Example: If you’re analyzing an image of a cat, the first layers might detect edges, while deeper layers capture complex patterns like whiskers or ears.  

2. **Activation Function (ReLU)**  
   - **Purpose**: Introduce **non-linearity** to allow the network to model complex patterns.  
   - **How it works**: After convolution, the **ReLU (Rectified Linear Unit)** function replaces all negative values with zero, making the model more efficient.  

3. **Pooling Layer**  
   - **Purpose**: This layer **downsamples** the feature maps, reducing their size to make the network computationally efficient while preserving important information.  
   - **Types**:  
      - **Max Pooling**: Selects the **maximum value** from a region (most common).  
      - **Average Pooling**: Computes the **average value** in a region.  

4. **Fully Connected (FC) Layer**  
   - **Purpose**: After extracting features, the CNN flattens the feature maps and passes them to a traditional neural network (dense layers) to make predictions.  
   - Example: For image classification, the FC layer outputs probabilities for different categories (e.g., "cat," "dog," "car").  

5. **Output Layer**  
   - **Purpose**: Produces the final prediction. For classification tasks, this is usually done using **softmax**, which outputs probabilities for each class.  

### **Why CNNs Are So Effective**  

1. **Local Feature Detection**: By using small filters, CNNs can detect **local patterns** like edges or textures.  
2. **Parameter Sharing**: The same filter is used across the image, reducing the number of parameters and improving efficiency.  
3. **Translation Invariance**: Features detected by convolutional filters remain identifiable even if they move within the image (e.g., recognizing a cat anywhere in a picture).  

### **Real-World Applications of CNNs**  

- **Image Classification** (e.g., identifying cats vs. dogs)  
- **Object Detection** (e.g., finding cars in traffic)  
- **Medical Imaging** (e.g., detecting tumors in X-rays)  
- **Facial Recognition** (e.g., unlocking your phone)  

Imagine you're looking at a picture of a cat. Instead of analyzing the whole image at once, a CNN **scans small parts**—detecting edges, curves, and textures. It’s like a detective who examines close-up details first (edges of whiskers), then forms a complete picture (a cat) using all the clues.

A **convolutional neural network** is a super-fast, detail-oriented model that helps computers "see" and recognize images, just like your brain identifies familiar objects.