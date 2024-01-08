# DSML-machine-learning-and-Deep-Learning-task

## Overview
This repository contains the projects and tasks completed in the 4th and 5th modules of the Data Science and Machine Learning course, focusing on Machine Learning (ML) and Deep Learning (DL).

![pexels-tara-winstead-8386434](https://github.com/anurashikvk/DSML-machine-learning-and-Deep-Learning-task/assets/134492695/7d935c01-3268-48da-ba55-333271f6b751)

## Machine Learning Tasks

### Task 1: Insurance Claims Charges Prediction
#### Overview
Predict insurance charges based on various factors using machine learning models.

#### Project Structure
- **Data Collection:**
  - The dataset is obtained from [source link](insert_source_link) using the wget command.
  - Data is read into a Pandas DataFrame for further analysis.
- **Exploratory Data Analysis (EDA):**
  - Basic libraries such as NumPy, Pandas, Matplotlib, and Seaborn are imported.
  - Descriptive statistics, null checks, and visualizations are performed.
- **Data Preprocessing:**
  - Label encoding is applied to categorical variables.
- **Feature Selection:**
  - Independent (features) and dependent (target) variables are selected.
- **Modeling:**
  - Regression models include Linear Regression, Decision Tree, Random Forest, and Support Vector Machine (SVR).

### Task 2: Drug Classification with Various ML Models
#### Objectives
Explore the dataset using various types of data visualization and build ML models to predict drug type.

#### Machine Learning Models Used
- Linear Logistic Regression
- Linear Support Vector Machine (SVM)
- K Neighbours
- Naive Bayes (Categorical & Gaussian)
- Decision Tree
- Random Forest

## Deep Learning Tasks

![pexels-pavel-danilyuk-8438922](https://github.com/anurashikvk/DSML-machine-learning-and-Deep-Learning-task/assets/134492695/af739497-a37e-4155-bc39-75a708ccd6ec)

### Task 1: MLP for Binary Classification
- Use the Ionosphere binary classification dataset to demonstrate an MLP for binary classification.
- Predict whether a structure is in the atmosphere or not.
- Utilize 'relu' activation with 'he_normal' weight initialization, sigmoid activation function, and cross-entropy loss.

### Task 2: MLP for Multiclass Classification
- Use the Iris flowers multiclass classification dataset for an MLP in multiclass classification.
- Predict the species of iris flower.
- Use softmax activation, 'sparse_categorical_crossentropy' loss.

### Task 3: MLP for Regression
- Use the Boston housing regression dataset to demonstrate an MLP for regression.
- Predict house value based on house and neighborhood properties.
- Use linear activation and mean squared error (mse) loss.

### Task 4: Deep Learning CNN for Fashion-MNIST Clothing Classification
- Utilize a CNN for classifying Fashion-MNIST clothing images.
- Dataset: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- Import dataset using Keras: `from tf.keras.datasets import fashion_mnist`

## References
If you have any doubts, refer to this [URL](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/).
