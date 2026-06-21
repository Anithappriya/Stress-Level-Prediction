import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv("C:/Users/ELCOT/Documents/12 (1).csv")

# Step 2: Rename columns for better readability
data.columns = ['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 
                'blood_oxygen', 'eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']

# Step 3: Data overview
print("Rows and Columns of the dataset:", data.shape)
print(data.info())
print(data.describe(include="all"))
print("Null values per column:\n", data.isnull().sum())

# Step 4: Handle missing values
data.fillna(data.mean(), inplace=True)

# Step 5: Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('stress_level', axis=1))
data_scaled = pd.DataFrame(scaled_features, columns=data.columns[:-1])

# Step 6: Splitting the data
X = data_scaled
y = data['stress_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Initialize and train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Step 8: Make predictions and evaluate
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
error_rate_dt = 1 - accuracy_dt
report_dt = classification_report(y_test, y_pred_dt, zero_division=1)

print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")
print(f"Error rate: {error_rate_dt:.2f}")
print("Decision Tree Classification Report:\n", report_dt)

# Step 9: Initialize and train Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 10: Make predictions and evaluate
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
error_rate_rf = 1 - accuracy_rf
report_rf = classification_report(y_test, y_pred_rf, zero_division=1)

print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print(f"Error rate: {error_rate_rf:.2f}")
print("Random Forest Classification Report:\n", report_rf)

# Step 11: Initialize and train Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Step 12: Make predictions and evaluate
y_pred_nb = nb_classifier.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
error_rate_nb = 1 - accuracy_nb
report_nb = classification_report(y_test, y_pred_nb, zero_division=1)

print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}")
print(f"Error rate: {error_rate_nb:.2f}")
print("Naive Bayes Classification Report:\n", report_nb)

# Step 13: Plotting accuracies
models = ['Decision Tree', 'Random Forest', 'Naive Bayes']
accuracies = [accuracy_dt, accuracy_rf, accuracy_nb]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.title('Accuracy Comparison of Machine Learning Models')
plt.xlabel('Models')
plt.ylabel('Accuracy')

# Display accuracy values on top of the bars
for i in range(len(models)):
    plt.text(i, accuracies[i], f'{accuracies[i]*100:.2f}%', ha='center', va='bottom')

plt.ylim(0, 1)  # Set y-axis limits
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Step 14: Predicting new data
new_data = pd.DataFrame([[90.0, 23.0, 92.0, 15.0, 90.0, 95.0, 2.0, 70.0]], columns=X.columns)
scaled_new_data = scaler.transform(new_data)  # Scale the new data
predicted_stress_level = rf_classifier.predict(scaled_new_data)

# Step 15: Map predicted stress level to labels
stress_level_labels = {0: "Low/Normal", 1: "Medium low", 2: "Medium", 3: "Medium High", 4: "High"}
predicted_stress_label = stress_level_labels[predicted_stress_level]

# Provide feedback based on the predicted stress level
if predicted_stress_level == 0:
    print("Low stress level. Keep up the good work!")
elif predicted_stress_level == 1:
    print("Medium low stress level. Try to incorporate some relaxation techniques.")
elif predicted_stress_level == 2:
    print("Medium. Consider taking a break and engaging in stress-reducing activities.")
elif predicted_stress_level == 3:
    print("Medium high stress level. It's important to prioritize self-care and seek support.")
else:
    print("High stress level. Seek professional help and prioritize your well-being.")

# Print the predicted stress label
print("Predicted Stress Level:", predicted_stress_level, "(", predicted_stress_label, ")")
