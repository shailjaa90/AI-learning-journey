import numpy as np
from numpy.polynomial import Polynomial

# Dummy dataset: Study hours vs Exam scores
study_hours = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
exam_scores = np.array([55, 60, 65, 70, 75, 80, 85, 90, 95])

# Simple linear regression using NumPy
# Calculate mean values
mean_study_hours = np.mean(study_hours)
mean_exam_scores = np.mean(exam_scores)

# Calculate slope (b1) and intercept (b0)
numerator = np.sum((study_hours - mean_study_hours) * (exam_scores - mean_exam_scores))
denominator = np.sum((study_hours - mean_study_hours) ** 2)
slope = numerator / denominator
intercept = mean_exam_scores - slope * mean_study_hours

# Display model parameters
print("=" * 50)
print("Linear Regression Model: Exam Score Prediction")
print("=" * 50)
print(f"Dataset:")
print(f"  Study Hours: {study_hours}")
print(f"  Exam Scores: {exam_scores}")
print()
print(f"Model Parameters:")
print(f"  Slope (b1): {slope:.4f}")
print(f"  Intercept (b0): {intercept:.4f}")
print()
print(f"Linear Regression Equation: Score = {intercept:.4f} + {slope:.4f} * Study_Hours")
print()

# Make predictions
print("Predictions for new study hours:")
print("-" * 50)
test_hours = np.array([1, 5.5, 11])
predictions = intercept + slope * test_hours

for hours, score in zip(test_hours, predictions):
    print(f"  Study Hours: {hours:>4} → Predicted Score: {score:>6.2f}")

print("=" * 50)
