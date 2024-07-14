# Cats-Dogs-Classification
Ai/ Deeplearning Cats &amp; Dogs Classification Model
Here Is the Colab Link : https://colab.research.google.com/drive/1CKVBw5RWoXLHiLwB8eISMkrycpWdSjrJ?usp=sharing

### Cat vs. Dog Image Classifier Project

#### Project Overview
The goal of this project was to develop a convolutional neural network (CNN) to classify images of cats and dogs. This task involved using various deep learning techniques such as data augmentation, feature extraction, and fine-tuning to improve the model's performance. The final model achieved a test accuracy of 90.4% and a validation accuracy of 92.0%, demonstrating its effectiveness in distinguishing between cat and dog images.

#### Dataset
The dataset used for this project contained a large number of images of cats and dogs. The images were divided into training, validation, and test sets. Data augmentation techniques were applied to the training set to increase its diversity and prevent overfitting.

#### Data Augmentation
Data augmentation was employed to enhance the training dataset by applying random transformations to the images. These transformations included:

- Random rotations
- Flipping (horizontal and vertical)
- Zooming
- Shifting (horizontal and vertical)
- Shearing

This technique helps to make the model more robust by exposing it to a variety of image variations, thereby improving its generalization capability.

#### Model Architecture
The architecture of the final model consisted of a pre-trained VGG16 network followed by a custom fully connected network. Here is a summary of the model architecture:

1. **VGG16 Base Model**:
   - The VGG16 model was used as the base for feature extraction. The pre-trained weights on the ImageNet dataset were leveraged, and the convolutional layers of VGG16 were frozen to retain the learned features.

2. **Custom Fully Connected Layers**:
   - **Flatten Layer**: Flattened the output from the VGG16 model.
   - **Dense Layer**: A fully connected layer with 512 units and ReLU activation function.
   - **Dropout Layer**: Applied dropout with a rate of 0.5 to prevent overfitting.
   - **Output Layer**: A single unit with a sigmoid activation function for binary classification (cat vs. dog).

#### Model Compilation
The model was compiled using the Adam optimizer and binary cross-entropy loss function, which is suitable for binary classification tasks. The metrics used for evaluation included accuracy.

#### Training and Fine-Tuning
The model was trained using the training dataset with data augmentation applied. Fine-tuning was performed by unfreezing some of the top layers of the VGG16 base model and retraining them alongside the custom fully connected layers. This allowed the model to adapt the pre-trained features to the specific task of classifying cats and dogs.

#### Performance
The model achieved impressive results:
- **Test Accuracy**: 90.4%
- **Validation Loss**: 0.1822
- **Validation Accuracy**: 92.0%

The final architecture summary of the model is as follows:

```
 Layer (type)                Output Shape              Param #   
=================================================================
 vgg16 (Functional)          (None, 4, 4, 512)         14714688  
                                                                 
 flatten_1 (Flatten)         (None, 8192)              0         
                                                                 
 dense_2 (Dense)             (None, 512)               4194816   
                                                                 
 dropout_1 (Dropout)         (None, 512)               0         
                                                                 
 dense_3 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 18910017 (72.14 MB)
Trainable params: 11274753 (43.01 MB)
Non-trainable params: 7635264 (29.13 MB)
```

#### Conclusion
This project successfully demonstrated the application of various deep learning techniques to build an effective cat vs. dog image classifier. The use of a pre-trained VGG16 model for feature extraction, combined with custom fully connected layers and data augmentation, resulted in a robust model with high accuracy. The achieved performance metrics highlight the model's capability to accurately classify images of cats and dogs, making it a valuable tool for such binary classification tasks.
