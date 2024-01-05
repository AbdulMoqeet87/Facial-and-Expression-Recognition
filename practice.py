import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function for project and back-project
def project_and_back_project(test_image, pca_basis, mean_vector):
    centered_test_image = test_image - mean_vector
    projected_data = np.dot(centered_test_image, pca_basis)
    back_projected_data = np.dot(projected_data, pca_basis.T) + mean_vector
    return back_projected_data

# Load five images
image_paths1 = [
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.1.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.2.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.3.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.4.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.5.png',
]
image_paths2 = [
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject02\subject2.1.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject02\subject2.2.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject02\subject2.3.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject02\subject2.4.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject02\subject2.5.png',
]

images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths1]
images2= [cv2.imread(path2, cv2.IMREAD_GRAYSCALE) for path2 in image_paths2]
# Resize images if needed
images_resized = [cv2.resize(img, (100, 100), interpolation=cv2.INTER_LINEAR) for img in images]
images_resized2 = [cv2.resize(img2, (100, 100), interpolation=cv2.INTER_LINEAR) for img2 in images2]

# Display the original images
plt.figure(figsize=(15, 5))
for i, img in enumerate(images_resized):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Subject 1 - Image {i + 1}")

# Display the original images for Subject 2
for i, img in enumerate(images_resized2):
    plt.subplot(2, 5, i + 6)
    plt.imshow(img, cmap='gray')
    plt.title(f"Subject 2 - Image {i + 1}")

plt.show()

# Flatten the images and stack them into a data matrix
data_matrix = np.vstack([img.flatten() for img in images_resized]).T
data_matrix2 = np.vstack([img.flatten() for img in images_resized2]).T
print("datamatrix:",data_matrix.shape)

# Center the data by subtracting the mean
mean_vector = np.mean(data_matrix, axis=0)
centered_data = data_matrix - mean_vector

# Center the data for the second set of images
mean_vector2 = np.mean(data_matrix2, axis=0)
centered_data2 = data_matrix2 - mean_vector2

# Compute the covariance matrix for centered data
covariance_matrix = np.cov(centered_data, rowvar=False)
covariance_matrix2 = np.cov(centered_data2, rowvar=False)
print("covar",covariance_matrix.shape)
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
eigenvalues2, eigenvectors2 = np.linalg.eigh(covariance_matrix2)

# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
sorted_indices2 = np.argsort(eigenvalues2)[::-1]
eigenvalues2 = eigenvalues2[sorted_indices2]
eigenvectors2 = eigenvectors2[:, sorted_indices2]

# Create the unit eigen matrix
unit_eigen_matrix = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
unit_eigen_matrix2 = eigenvectors2 / np.linalg.norm(eigenvectors2, axis=0)

# Keep only the principal component with the highest variance for each subject
unit_eigen_matrix = unit_eigen_matrix[:, 0][:, np.newaxis]
unit_eigen_matrix2 = unit_eigen_matrix2[:, 0][:, np.newaxis]

print("unit eigen",unit_eigen_matrix.shape)
# Project the original data matrix with the updated unit eigen matrix
resultant_matrix = data_matrix @ unit_eigen_matrix
resultant_matrix2 = data_matrix2 @ unit_eigen_matrix2

# Calculate percentage variance for each subject
percentage_variance = (eigenvalues / np.sum(eigenvalues)) * 100
print("Percentage Variance for Subject 1:")
print(f"PC 1: {percentage_variance[0]:.2f}%")

# Display the final image based on the first principal component for Subject 1
plt.figure(figsize=(5, 5))
final_image = resultant_matrix[:, 0].reshape(images_resized[0].shape)
plt.imshow(final_image, cmap='gray')
plt.title(f"Subject 1 - PC 1")
plt.show()

# Calculate percentage variance for Subject 2
percentage_variance2 = (eigenvalues2 / np.sum(eigenvalues2)) * 100
print("Percentage Variance for Subject 2:")
print(f"PC 1: {percentage_variance2[0]:.2f}%")

# Display the final image based on the first principal component for Subject 2
plt.figure(figsize=(5, 5))
final_image2 = resultant_matrix2[:, 0].reshape(images_resized2[0].shape)
print("image2 final",final_image2.shape)
plt.imshow(final_image2, cmap='gray')
plt.title(f"Subject 2 - PC 1")
plt.show()
print("image2 final",final_image2.shape)
# Load the test image
test_image_path = r'E:\3rd semester\Linear Algebra\Python_Assignment\testimages\testimage6.png'  # Replace with the path to your test image
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
test_image_resized = cv2.resize(test_image, (100, 100), interpolation=cv2.INTER_LINEAR)
test_flattened = test_image_resized.flatten()

# Center the test image
centered_test = test_flattened - mean_vector[:, np.newaxis]

# Project onto the 1st principal component for both subjects
projected_test_subject1 = np.dot(unit_eigen_matrix[:, 0], centered_test)
projected_test_subject2 = np.dot(unit_eigen_matrix2[:, 0], centered_test)

# Calculate reconstruction loss for each subject
loss_subject1 = np.linalg.norm(centered_test - projected_test_subject1 * unit_eigen_matrix[:, 0][:, np.newaxis])
loss_subject2 = np.linalg.norm(centered_test - projected_test_subject2 * unit_eigen_matrix2[:, 0][:, np.newaxis])


print("Loss for Subject 1:", loss_subject1)
print("Loss for Subject 2:", loss_subject2)

# Identify the subject with the lowest loss
predicted_subject = 1 if loss_subject1 < loss_subject2 else 2

print(f"The test image is predicted to resemble Subject {predicted_subject}")

# Display the original test image
plt.figure(figsize=(5, 5))
plt.imshow(test_image_resized, cmap='gray')
plt.title("Original Test Image")
plt.show()
print("datamatrix", data_matrix.shape)
print("datamatrix", resultant_matrix.shape)



