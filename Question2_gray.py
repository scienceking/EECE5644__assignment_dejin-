# to run this code, we much ensure the picture location is right, or it would encounter errors.

import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Load the image and extract features
def extract_features(image):
    rows, cols, _ = image.shape
    features = []
    for row in range(rows):
        for col in range(cols):
            # Extract 5-dimensional features for each pixel: row index, column index, red, green, and blue channel values
            r, g, b = image[row, col]
            features.append([row, col, r, g, b])
    return np.array(features, dtype=np.float32)

# 2. Normalize features
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

# 3. Fit GMM directly using 10 components
def fit_gmm(features, n_components=10):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(features)
    return gmm

# 4. Assign labels to each pixel
def assign_labels(gmm, features):
    labels = gmm.predict(features)
    return labels

# 5. Display the original image and segmentation results
def display_segmentation(original_image, labels, n_components):
    rows, cols, _ = original_image.shape
    label_image = labels.reshape(rows, cols)

    # Map labels to grayscale values
    unique_labels = np.unique(labels)
    label_colors = np.linspace(0, 255, len(unique_labels)).astype(np.uint8)
    label_image_colored = np.zeros((rows, cols), dtype=np.uint8)

    for i, label in enumerate(unique_labels):
        label_image_colored[label_image == label] = label_colors[i]

    # Display the original image and the segmentation result
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(label_image_colored, cmap='gray')
    plt.title("GMM Segmentation")

    plt.show()

# Main workflow
image_path = r"C:\Users\cugwd\OneDrive - Northeastern University\Desktop\course\EECE_Assignment4\119082.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Extract, normalize features, and directly fit GMM
features = extract_features(image)
normalized_features = normalize_features(features)
gmm = fit_gmm(normalized_features, n_components=10)

# Assign labels to each pixel and display the results
labels = assign_labels(gmm, normalized_features)
display_segmentation(image, labels, n_components=10)
