import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Data generation function
def generate_data(n_samples=1000, r_minus=2, r_plus=4, sigma=1):
    theta = np.random.uniform(-np.pi, np.pi, n_samples)
    noise = np.random.normal(0, sigma, (n_samples, 2))

    x_minus = r_minus * np.column_stack((np.cos(theta), np.sin(theta))) + noise
    x_plus = r_plus * np.column_stack((np.cos(theta), np.sin(theta))) + noise

    X = np.vstack((x_minus, x_plus))
    y = np.hstack((-np.ones(n_samples), np.ones(n_samples)))

    return X, y

# Generate training and test data
X_train, y_train = generate_data(n_samples=1000)
X_test, y_test = generate_data(n_samples=10000)

# Define SVM model and parameter grid
svm = SVC(kernel='rbf')
param_grid = {
    'C': [0.1, 1, 10, 100],  # Box constraint parameter
    'gamma': [0.01, 0.1, 1, 10]  # Gaussian kernel width parameter
}

# Use GridSearchCV with 10-fold cross-validation to select the best hyperparameters
grid_search = GridSearchCV(svm, param_grid, cv=10, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train, y_train)

# Output the best hyperparameters
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Extract cross-validation results and convert to DataFrame for visualization
results = pd.DataFrame(grid_search.cv_results_)
results_pivot = results.pivot(index="param_C", columns="param_gamma", values="mean_test_score")

# Visualize cross-validation results
plt.figure(figsize=(8, 6))
sns.heatmap(results_pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'})
plt.title("10-Fold Cross-Validation Accuracy for Different Hyperparameter Combinations")
plt.xlabel("Gamma")
plt.ylabel("C")
plt.show()

# Train the final SVM model with the best hyperparameters
best_svm = grid_search.best_estimator_
best_svm.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_error = 1 - test_accuracy
print("Test accuracy:", test_accuracy)
print("Test error probability:", test_error)

# Visualize the classification boundary
plt.figure(figsize=(8, 8))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=10, alpha=0.5)
plt.title("Test Data with SVM Classification Boundary")

# Create a grid to display the classification boundary
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the classification boundary
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
