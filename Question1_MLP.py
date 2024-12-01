import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# Define quadratic polynomial activation function
class QuadraticActivation(nn.Module):
    def forward(self, x):
        return 0.5 * x ** 2 + x  # a*x^2 + b*x, assuming a=0.5, b=1


# Data generation function
def generate_data(n_samples=1000, r_minus=2, r_plus=4, sigma=1):
    theta = np.random.uniform(-np.pi, np.pi, n_samples)
    noise = np.random.normal(0, sigma, (n_samples, 2))

    x_minus = r_minus * np.column_stack((np.cos(theta), np.sin(theta))) + noise
    x_plus = r_plus * np.column_stack((np.cos(theta), np.sin(theta))) + noise

    X = np.vstack((x_minus, x_plus)).astype(np.float32)
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples))).astype(np.float32)  # Labels 0 and 1

    return X, y


# Generate data
X, y = generate_data(n_samples=1000)

# Convert data to Tensor
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)


# Define MLP model
class MLP(nn.Module):
    def __init__(self, hidden_neurons):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, hidden_neurons)  # Hidden layer size specified by parameter
        self.quadratic = QuadraticActivation()  # Use quadratic polynomial activation function
        self.output = nn.Linear(hidden_neurons, 1)  # Output layer for binary classification task
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function for output layer

    def forward(self, x):
        x = self.hidden(x)
        x = self.quadratic(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


# Use 10-fold cross-validation to select the optimal number of hidden neurons
kf = KFold(n_splits=10, shuffle=True, random_state=42)
hidden_neurons_list = [10, 20, 50, 100]  # Test different hidden layer neuron counts
best_accuracy = 0
best_neurons = None

for neurons in hidden_neurons_list:
    cv_accuracies = []
    for train_index, val_index in kf.split(X):
        # Split training and validation sets
        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]

        # Initialize model, loss function, and optimizer
        model = MLP(hidden_neurons=neurons)
        criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        epochs = 200
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train).squeeze()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate model on validation set
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val).squeeze().round()
            accuracy = accuracy_score(y_val.numpy(), y_val_pred.numpy())
            cv_accuracies.append(accuracy)

    # Calculate average cross-validation accuracy for this number of neurons
    avg_accuracy = np.mean(cv_accuracies)
    print(f"Neurons: {neurons}, CV Accuracy: {avg_accuracy:.4f}")

    # Update the best neuron count
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_neurons = neurons

print(f"Best Neurons: {best_neurons}, Best CV Accuracy: {best_accuracy:.4f}")

# Retrain model with the best number of neurons and evaluate
model = MLP(hidden_neurons=best_neurons)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor).squeeze()
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# Visualize classification boundary
X_test, y_test = generate_data(n_samples=10000)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze().round().numpy()
    test_accuracy = accuracy_score(y_test, y_pred)
    test_error = 1 - test_accuracy
    print("Test accuracy:", test_accuracy)
    print("Test error probability:", test_error)

# Visualize classification boundary
plt.figure(figsize=(8, 8))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=10, alpha=0.5)
plt.title("Test Data with MLP Classification Boundary (Quadratic Activation)")

# Create a grid to display the classification boundary
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
Z = model(grid_tensor).detach().numpy().reshape(xx.shape)

# Plot the classification boundary as a black solid line
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
