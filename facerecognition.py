import cv2
import numpy as np
import matplotlib.pyplot as plt
def project_and_back_project(test_image, pca_basis, mean_vector):
    centered_test_image = test_image - mean_vector
    projected_data = np.dot(centered_test_image, pca_basis)
    back_projected_data = np.dot(projected_data, pca_basis.T) + mean_vector
    return back_projected_data

# Load five images
image_paths1 = [
    r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject1\subject1.1.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject1\subject1.2.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject1\subject1.3.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject1\subject1.4.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject1\subject1.5.png',
]
image_paths2 = [
    r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject2\subject2.1.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject2\subject2.2.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject2\subject2.3.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject2\subject2.4.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject2\subject2.5.png',
]

images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths1]
images2= [cv2.imread(path2, cv2.IMREAD_GRAYSCALE) for path2 in image_paths2]
# Resize images if needed
images_resized = [cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR) for img in images]
images_resized2 = [cv2.resize(img2, (50, 50), interpolation=cv2.INTER_LINEAR) for img2 in images2]

# Display the original images

# for i, img in enumerate(images_resized):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(img, cmap='gray')
#     plt.title(f"Image {i + 1}")
#     plt.axis('off')
# # Display the original images

# for i, img in enumerate(images_resized2):
#     plt.subplot(1, 10, i + 6)
#     plt.imshow(img, cmap='gray')
#     plt.title(f"Image {i + 1}")
#     plt.axis('off')


# plt.show()

# Flatten the images and stack them into a data matrix
data_matrix = np.vstack([img.flatten() for img in images_resized])
data_matrix2 = np.vstack([img.flatten() for img in images_resized2])

print("data set",data_matrix.shape)
print("data set 2",data_matrix2.shape)
# Center the data by subtracting the mean
mean_vector = np.mean(data_matrix, axis=0)
print("mean vec",mean_vector.shape)
centered_data = data_matrix - mean_vector
print("centered data",centered_data.shape)
# Center the data for the second set of images
mean_vector2 = np.mean(data_matrix2, axis=0)
centered_data2 = data_matrix2 - mean_vector2

# Compute the covariance matrix for centered data
covariance_matrix = np.cov(centered_data, rowvar=False)
covariance_matrix2 = np.cov(centered_data2, rowvar=False)
print("covariance matrix ",covariance_matrix.shape)
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
unit_eigen_matrix = eigenvectors
unit_eigen_matrix2 = eigenvectors2

cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
cumulative_energy2 = np.cumsum(eigenvalues2) / np.sum(eigenvalues2)

num_components2 = np.argmax(cumulative_energy2 >= 0.99) + 1
num_components = np.argmax(cumulative_energy >= 0.99) + 1
selected_eigenvectors = eigenvectors[:, :num_components]
selected_eigenvectors2 = eigenvectors2[:, :num_components2]

print("selected eigenvec",selected_eigenvectors.shape)

print("eigenvector",eigenvectors.shape)
# Multiply the original data matrix with the unit eigen matrix

test_image_path = r'E:\3rd semester\Linear Algebra\Python_Assignment\testimages\testimage5.png'  # Replace with the path to your test image
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

test_image_resized = cv2.resize(test_image, (50, 50), interpolation=cv2.INTER_LINEAR)
test_flattened = test_image_resized.flatten()
centered_test = test_flattened - mean_vector
centered_test2 = test_flattened - mean_vector2
print("mean vec",mean_vector.shape,centered_test.shape)
print("testimg",test_image_resized.shape)
# Project the centered test image onto the selected eigenvectors
projected_test = np.dot(centered_test, selected_eigenvectors)
projected_test2 = np.dot(centered_test2, selected_eigenvectors2)

# Back-project the projected test image to the original space
back_projected_test = np.dot(projected_test, selected_eigenvectors.T) + mean_vector
back_projected_test2 = np.dot(projected_test2, selected_eigenvectors2.T) + mean_vector2

# Reshape the back-projected test image to its original shape
back_projected_test = back_projected_test.reshape(test_image_resized.shape)
back_projected_test2 = back_projected_test2.reshape(test_image_resized.shape)

# Display the original, projected, and back-projected test images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(test_image_resized, cmap='gray')
plt.title("Original Test Image")

plt.subplot(1, 3, 2)
plt.imshow(test_image_resized - mean_vector.reshape(test_image_resized.shape), cmap='gray')
plt.title("Centered Test Image")

plt.subplot(1, 3, 3)
plt.imshow(back_projected_test, cmap='gray')
plt.title("Back-Projected Test Image")

mse= np.mean((test_image_resized-back_projected_test)**2)
mse2= np.mean((test_image_resized-back_projected_test2)**2)
print("mse",mse)
print("mse2",mse2)
if(mse>mse2): print("predicted to be subject 2" )
else: print("predicted to be subject1")
plt.show()
# Center the test image

# # Load the test image
# test_image_path = r'E:\3rd semester\Linear Algebra\Python_Assignment\testimages\testimage5.png'
# test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
# test_image_resized = cv2.resize(test_image, (100, 100), interpolation=cv2.INTER_LINEAR)
# flattened_test_image = test_image_resized.flatten()

# # Center the test image
# centered_test_image = flattened_test_image - mean_vector

# # Project and back-project the test image
# reconstructed_test_image = project_and_back_project(centered_test_image, pca_basis, mean_vector)

# # Reshape the reconstructed image to its original shape
# reconstructed_test_image = reconstructed_test_image.reshape(test_image_resized.shape)

# # Display the original and reconstructed test images
# plt.subplot(1, 2, 1)
# plt.imshow(test_image_resized, cmap='gray')
# plt.title("Original Test Image")

# plt.subplot(1, 2, 2)
# plt.imshow(reconstructed_test_image, cmap='gray')
# plt.title("Reconstructed Test Image")

# plt.show()








# percentage_variance2 = (eigenvalues2 / np.sum(eigenvalues2)) * 100
# print("Percentage Variance for Each Principal Component:")
# for i, variance2 in enumerate(percentage_variance2):
#     print(f"PC {i + 1}: {variance2:.2f}%")

# # Find the index of the principal component with the maximum variance
# max_variance_index = np.argmax(percentage_variance)
# max_variance_index2 = np.argmax(percentage_variance2)

# # Display the final image based on the principal component with the maximum variance
# final_image = resultant_matrix[:, max_variance_index].reshape(images_resized[0].shape)
# final_image2 = resultant_matrix2[:, max_variance_index2].reshape(images_resized2[0].shape)

# plt.imshow(final_image, cmap='gray')
# plt.title(f"Final Image (PC {max_variance_index + 1}, {percentage_variance[max_variance_index]:.2f}% variance)")
# plt.show()

# plt.imshow(final_image2, cmap='gray')
# plt.title(f"Final Image (PC {max_variance_index2 + 1}, {percentage_variance2[max_variance_index2]:.2f}% variance)")
# plt.show()
