# Import the needed libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree # Import DecisionTreeClassifier and plot_tree
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Read the dataset
data = pd.read_csv(r"heart.csv")


# Split the dataset into features (X) and target variable (Y)
X = data[["age",  "sex", "cp", "trestbps","chol", "fbs", "restecg",
          "thalach", "exang", "oldpeak", "slope",  "ca", "thal"]]
Y = data.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1)

# Initialize the Decision Tree Classifier with entropy as the criterion
model = DecisionTreeClassifier(criterion="entropy", random_state=1)

# Train the Decision Tree Classifier
model.fit(x_train, y_train)

# Function to make predictions
def prediction(x_test, dtc_obj):
    y_pred = dtc_obj.predict(x_test)
    return y_pred

# Function to calculate accuracy
def calc_accuracy(y_test , y_pred):
    print("Accuracy: " , accuracy_score(y_test , y_pred) * 100)

# Print results: make predictions and calculate accuracy
print("Results:")
y_pred_entropy = prediction(x_test, model)
calc_accuracy(y_test, y_pred_entropy)

# Function to print the decision tree
def print_tree():
    feature_names = ["age",  "sex", "cp", "trestbps","chol", "fbs", "restecg",
          "thalach", "exang", "oldpeak", "slope",  "ca", "thal"]

    class_names = ["HD-YES" , "HD-NO"]

    plt.figure("Heart Disease Decision Tree", dpi=80)
    plot_tree(model, fontsize=10, filled=True, feature_names=feature_names, class_names=class_names, proportion=True)
    plt.show() # Display the decision tree plot

# Call the print_tree function to visualize the decision tree
print_tree()
