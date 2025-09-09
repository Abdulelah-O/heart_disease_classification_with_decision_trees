# Heart Disease Classification with Decision Trees

## Project Overview
This project implements a machine learning model to classify heart disease based on patient data. The solution uses a **Decision Tree Classifier**,a powerful algorithm for both classification and regression tasks. The model is trained and tested on a provided dataset, with a specific focus on evaluating performance, visualizing the decision tree, and addressing the issue of overfitting.

## Files

  - **ML_HD.py:** The main Python script that contains all the code for data processing, model training, evaluation, and visualization.
  - **heart.csv:** The dataset used for training and testing the model. It contains various features related to patient health and a target variable indicating the presence of heart disease.

## Implementation Details

The ML_HD.py script performs the following key steps:

  1. **Data Loading:** The ***heart.csv*** dataset is loaded into a Pandas DataFrame.
  2. **Data Splitting:** The dataset is split into features ***(X)*** and the target variable ***(Y).*** The data is then divided into training and testing sets to evaluate the model's performance on unseen data.
  3. **Model Training:** A ***DecisionTreeClassifier*** is initialized and trained on the training data. The ***entropy*** criterion is used to measure the quality of a split.
  4. **Prediction and Evaluation:** The trained model is used to make predictions on the test set, and its performance is evaluated using the **accuracy_score.**
  5. **Visualization:** The code generates a visual representation of the trained decision tree, which helps in understanding the model's decision-making process.
  6. **Overfitting:** The problem of overfitting is addressed, as specified in the assignment document, to ensure the model generalizes well to new data.

## How to Run

  - Python
  - The following Python libraries: **pandas**, **scikit-learn**, and **matplotlib**.

## Execution

  1. Ensure the **ML_HD.py** script and the **heart.csv** file are in the same directory.
  2. Open a terminal or command prompt.
  3. Navigate to the project directory.
  4. Run the script using the Python interpreter:
       ```
       python ML_HD.py
       ```

The script will print the model's accuracy on the test set and display a visualization of the decision tree.

Here's an example for the output

<img width="331" height="48" alt="image" src="https://github.com/user-attachments/assets/c09d6daa-2a4d-4058-8341-ab61daeb7ac3" />

<img width="1911" height="1111" alt="image" src="https://github.com/user-attachments/assets/41a9f17e-3179-4ae2-979d-c363d9037657" />


Contributors:

- Abdulaziz Alhaizan
- Abdulelah Bin Obaid
- Fawaz Mufti

