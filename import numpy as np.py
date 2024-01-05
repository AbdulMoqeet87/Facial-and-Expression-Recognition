import cv2
import numpy as np
import matplotlib.pyplot as plt

def project_and_back_project(test_image, pca_basis, mean_vector):
    centered_test_image = test_image - mean_vector
    projected_data = np.dot(centered_test_image, pca_basis)
    back_projected_data = np.dot(projected_data, pca_basis.T) + mean_vector
    return back_projected_data

# Load images for 15 subjects with 10 images each
num_subjects = 5
num_images_per_subject = 5

image_paths = [
    [f'E:\\3rd semester\\Linear Algebra\\Python_Assignment\\yalefaces\\subject{i:02d}\\subject{i}.{j}.png' for j in range(1, num_images_per_subject + 1)]
    for i in range(1, num_subjects + 1)
]

images = [
    [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in subject_paths]
    for subject_paths in image_paths
]

# Resize images if needed
images_resized = [
    [cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR) for img in subject_images]
    for subject_images in images
]

# Display the original images
plt.figure(figsize=(15, 15))

for i, subject_images in enumerate(images_resized):
    for j, img in enumerate(subject_images):
        plt.subplot(num_subjects, num_images_per_subject, i * num_images_per_subject + j + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Subject {i + 1}, Image {j + 1}")
        plt.axis('off')

plt.show()

# Flatten the images and stack them into a data matrix
data_matrix = np.vstack([img.flatten() for subject_images in images_resized for img in subject_images])

print("data set", data_matrix.shape)

# Center the data by subtracting the mean
mean_vector = np.mean(data_matrix, axis=0)
centered_data = data_matrix - mean_vector
print("centered data", centered_data.shape)

# Compute the covariance matrix for centered data
covariance_matrix = np.cov(centered_data, rowvar=False)
print("covariance matrix ", covariance_matrix.shape)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Create the unit eigen matrix
unit_eigen_matrix = eigenvectors

cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
num_components = np.argmax(cumulative_energy >= 0.99) + 1
selected_eigenvectors = eigenvectors[:, :num_components]
print("selected eigenvec", selected_eigenvectors.shape)

# Project the test image onto the selected eigenvectors
test_image_path = r'E:\3rd semester\Linear Algebra\Python_Assignment\testimages\testimage5.png'
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
test_image_resized = cv2.resize(test_image, (50, 50), interpolation=cv2.INTER_LINEAR)
test_flattened = test_image_resized.flatten()
centered_test = test_flattened - mean_vector

# Project the centered test image onto the selected eigenvectors
projected_test = np.dot(centered_test, selected_eigenvectors)

# Back-project the projected test image to the original space
back_projected_test = np.dot(projected_test, selected_eigenvectors.T) + mean_vector

# Reshape the back-projected test image to its original shape
back_projected_test = back_projected_test.reshape(test_image_resized.shape)

# Display the original, centered, and back-projected test images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt
