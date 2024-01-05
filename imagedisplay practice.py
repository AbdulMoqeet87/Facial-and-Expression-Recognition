import cv2
import numpy as np
import matplotlib.pyplot as plt
def project_and_back_project(test_image, pca_basis, mean_vector):
    centered_test_image = test_image - mean_vector
    projected_data = np.dot(centered_test_image, pca_basis)
    back_projected_data = np.dot(projected_data, pca_basis.T) + mean_vector
    return back_projected_data

# Load the images
image_path1 = r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.1.png'
image_path2 = r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.2.png'
image_path3 = r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.3.png'
image_path4 = r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.4.png'
image_path5 = r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.5.png'

image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread(image_path3, cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(image_path4, cv2.IMREAD_GRAYSCALE)
image5 = cv2.imread(image_path5, cv2.IMREAD_GRAYSCALE)

# Flatten the images
flattened_image1 = image1.flatten()
flattened_image2 = image2.flatten()
flattened_image3 = image3.flatten()
flattened_image4 = image4.flatten()
flattened_image5 = image5.flatten()

# Combine the flattened images into a data matrix
data_matrix = np.vstack((flattened_image1, flattened_image2, flattened_image3, flattened_image4, flattened_image5))
# Transpose the data matrix if you want each column to represent a feature
data_matrix = data_matrix.T

# Step 1: Center the data by subtracting the mean
mean_vector = np.mean(data_matrix, axis=0)
centered_data = data_matrix - mean_vector

# Step 2: Compute the covariance matrix
covariance_matrix = np.cov(centered_data, rowvar=False)

# Step 3: Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Step 4: Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Choose the number of principal components to retain (based on desired energy retention)
cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
num_components = np.argmax(cumulative_energy >= 0.12) + 1

# Step 6: Select the top 'num_components' eigenvectors
selected_eigenvectors = eigenvectors[:, :num_components]

# Step 7: Project the original data onto the PCA space
projected_data = np.dot(centered_data, selected_eigenvectors)

# Step 8: Reconstruct the data from the projected space
reconstructed_data = np.dot(projected_data, selected_eigenvectors.T) + mean_vector

# Reshape the reconstructed data to the original image shape
reconstructed_images = reconstructed_data.T.reshape((5,) + image1.shape)

# Display the original and reconstructed images
# Display the original and reconstructed images
plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title("Original Image 1")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_images[0], cmap='gray')  # Corrected index
plt.title("Reconstructed Image 1")

# Display the PCA basis matrix
print(selected_eigenvectors.shape)
plt.figure(figsize=(12, 6))
for i in range(num_components):
    plt.subplot(1, num_components, i + 1)
    # Reshape each column of the selected eigenvectors to the original image shape
    eigenface = selected_eigenvectors.T[:, i].reshape(image1.shape[0], image1.shape[1])
    plt.imshow(eigenface, cmap='gray')
    plt.title(f"PC {i + 1}")

plt.show()
