# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Load dataset
data = pd.read_csv("student_marks.csv")
print(data)

# Step 3: Split data into input and output
X = data[['Hours']]   # Input (Study hours)
y = data['Marks']     # Output (Marks)

# Step 4: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create and train the ML model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict marks
y_pred = model.predict(X_test)

# Step 7: Show prediction
print("\nPredicted Marks:", y_pred)
print("Actual Marks:", y_test.values)

# Step 8: Plot graph
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction using ML")
plt.legend()
plt.show()

# Step 9: Predict for new input
hours = 6
predicted_marks = model.predict([[hours]])
print(f"\nIf a student studies {hours} hours, predicted marks = {predicted_marks[0]:.2f}")
