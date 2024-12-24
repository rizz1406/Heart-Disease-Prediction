# Heart Disease Prediction


This is a machine learning project that predicts whether a patient is at risk of heart disease based on various medical features. The model uses a dataset to classify patients into two categories: having heart disease or not.
![image](https://github.com/user-attachments/assets/f848a274-141a-4598-ac01-a25b13ec155a)


## Table of Contents
1. [Project Description](#project-description)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Model Implementation](#model-implementation)
5. [How to Run the Code](#how-to-run-the-code)
6. [Results](#results)
7. [License](#license)

## Project Description

The goal of this project is to build a machine learning model to predict the likelihood of a patient having heart disease. The dataset used for this model contains medical features like age, sex, blood pressure, cholesterol levels, etc. We used a classification model (e.g., Logistic Regression) to predict whether the patient has heart disease.

The project includes:
- Data preprocessing: Handling missing values, encoding categorical variables, and normalizing data.
- Model training: Using a machine learning algorithm to train the model.
- Evaluation: Measuring the performance using metrics like accuracy, precision, recall, and F1-score.

## Technologies Used

- **Python 3.x**
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For building and evaluating the machine learning model.
- **Matplotlib/Seaborn**: For data visualization.

## Dataset

The dataset used in this project contains various health-related features of patients, which are used to predict whether they have heart disease.

- The dataset can be downloaded from [Kaggle](https://www.kaggle.com/).
- It consists of 14 columns:
  - `age`: Age of the patient.
  - `sex`: Gender of the patient (1 = male, 0 = female).
  - `cp`: Chest pain type.
  - `trestbps`: Resting blood pressure.
  - `chol`: Serum cholesterol.
  - `fbs`: Fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false).
  - `restecg`: Resting electrocardiographic results.
  - `thalach`: Maximum heart rate achieved.
  - `exang`: Exercise induced angina (1 = yes, 0 = no).
  - `oldpeak`: Depression induced by exercise relative to rest.
  - `slope`: Slope of the peak exercise ST segment.
  - `ca`: Number of major vessels colored by fluoroscopy.
  - `thal`: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect).
  - `target`: Target variable (1 = heart disease, 0 = no heart disease).

## Model Implementation

1. **Data Preprocessing**: 
   - Load the dataset using Pandas.
   - Handle missing values (if any).
   - Encode categorical variables (e.g., gender, chest pain types).
   - Normalize continuous features (e.g., age, cholesterol).

2. **Model Building**:
   - Train a machine learning model using the processed data (e.g., Logistic Regression, Random Forest).
   - Split the dataset into training and testing sets.
   - Train the model on the training set and evaluate its performance on the test set.

3. **Model Evaluation**:
   - Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.

## How to Run the Code

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/rizz1406/Heart-Disease-Prediction.git

