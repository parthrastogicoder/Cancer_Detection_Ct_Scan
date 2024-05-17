# README

## Requirements

To run the code provided in this repository, you need the following dependencies:

- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- NumPy
- Scikit-learn

You can install these dependencies using pip:

```bash
pip install tensorflow keras matplotlib numpy scikit-learn pandas 
```

## Instructions

1. Clone this repository to your local machine:

```bash
git clone <repository_url>
```

2. Navigate to the project directory:

```bash
cd <project_directory>
```

3. Run the provided Python script.

## Dataset

The dataset used for this project consists of CT scan images of the chest. Each image is labeled with one of four categories: You can download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/data).

## Model Architectures

### Convolutional Neural Network (CNN)

The CNN model architecture consists of multiple convolutional layers followed by max-pooling layers. The final layers include fully connected layers with dropout regularization to prevent overfitting. The model is compiled using the Adam optimizer and categorical cross-entropy loss function.

### ResNet50

The ResNet50 model architecture is a pre-trained convolutional neural network that has been trained on the ImageNet dataset. The final layers of ResNet50 are removed and replaced with a custom dense layer followed by a sigmoid activation layer. The model is compiled using Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.0001 and categorical cross-entropy loss function.

### Previously Used Models

1. **Principal Component Analysis (PCA)**: A dimensionality reduction technique used to reduce the number of features in the dataset while preserving most of the information. PCA is commonly used for preprocessing before applying other machine learning algorithms.

2. **Stochastic Gradient Descent (SGD) with Perceptron Loss**: A linear classification algorithm that updates the model parameters in the direction of the negative gradient of the loss function with respect to the parameters.

3. **Logistic Regression**: A linear classification algorithm that models the probability that each input belongs to a particular category.

4. **K-Nearest Neighbors (KNN)**: A non-parametric classification algorithm that classifies new data points based on the majority class of their k-nearest neighbors in the feature space.

5. **Multi-Layer Perceptron (MLP)**: A feedforward neural network with one or more hidden layers between the input and output layers. Each neuron in the hidden layers uses a nonlinear activation function.

6. **Support Vector Machine (SVM)**: A supervised learning algorithm that constructs a hyperplane or set of hyperplanes in a high-dimensional feature space that can be used for classification or regression tasks.

7. **Gradient Boosting**: An ensemble learning method that builds a strong predictive model by combining the predictions of multiple weak models, typically decision trees.

## Evaluation Metrics

The performance of all models is evaluated using accuracy as the primary metric. Additionally, loss curves are plotted to visualize the training and validation performance over epochs.

\
