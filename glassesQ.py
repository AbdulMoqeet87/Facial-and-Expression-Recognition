import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

glasses_images = []
main_folder = r"E:\3rd semester\Linear Algebra\Python_Assignment\glasses"

for subject in range(6, 16):
    img_path = os.path.join(main_folder, f"{subject}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        glasses_images.append(img)
        print(f"Image {subject}.png loaded successfully.")
    else:
        print(f"Error loading image {subject}.png at path: {img_path}")

without_glasses_images = []
main_folder2 = r"E:\3rd semester\Linear Algebra\Python_Assignment\withoutGlasses"

for subject in range(6, 16):
    img_path = os.path.join(main_folder2, f"testimage{subject}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        without_glasses_images.append(img)
        print(f"Image testimage{subject}.png loaded successfully.")
    else:
        print(f"Error loading image testimage{subject}.png at path: {img_path}")


# Flatten and resize images
images_resized = [cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) for img in glasses_images]
images_resized2 = [cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) for img in without_glasses_images]

data_matrix = np.vstack([img.flatten() for img in images_resized])
data_matrix2 = np.vstack([img.flatten() for img in images_resized2])


print("data set",data_matrix.shape)

# Center the data by subtracting the mean
mean_vector = np.mean(data_matrix, axis=0)
mean_vector2 = np.mean(data_matrix2, axis=0)

print("mean vec",mean_vector.shape)
centered_data = data_matrix - mean_vector
print("centered data",centered_data.shape)
centered_data2 = data_matrix2 - mean_vector2

# Center the data for the second set of images

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

num_components = np.argmax(cumulative_energy >= 0.99) + 1
selected_eigenvectors = eigenvectors[:, :num_components]

num_components2 = np.argmax(cumulative_energy2 >= 0.99) + 1
selected_eigenvectors2 = eigenvectors2[:, :num_components2]

print("selected eigenvec",selected_eigenvectors.shape)

#all data matrices and mean vectors
test_image_path = r'E:\3rd semester\Linear Algebra\Python_Assignment\glasses\5.png'  # Replace with the path to your test image
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

test_image_resized = cv2.resize(test_image, (64, 64), interpolation=cv2.INTER_LINEAR)
test_flattened = test_image_resized.flatten()
centered_test = test_flattened - mean_vector
centered_test2 = test_flattened - mean_vector2

# Project the centered test image onto the selected eigenvectors
projected_test = np.dot(centered_test, selected_eigenvectors)
projected_test2 = np.dot(centered_test2, selected_eigenvectors2)

# Back-project the projected test image to the original space
back_projected_test = np.dot(projected_test, selected_eigenvectors.T) + mean_vector
back_projected_test2 = np.dot(projected_test2, selected_eigenvectors2.T) + mean_vector2

# Reshape the back-projected test image to its original shape
back_projected_test = back_projected_test.reshape(test_image_resized.shape)
back_projected_test2 = back_projected_test2.reshape(test_image_resized.shape)

'E:\3rd semester\Linear Algebra\Python_Assignment\testimages\testglasses'

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
mse = np.linalg.norm(test_image_resized - back_projected_test)
mse2 = np.linalg.norm(test_image_resized - back_projected_test2)
# mse= np.mean((test_image_resized-back_projected_test)**2)
# mse2= np.mean((test_image_resized-back_projected_test2)**2)
print("mse",mse)
print("mse2",mse2)
if(mse>mse2): print("prediction: not wearing glasses" )
else: print("predicted: wearing glasses")
plt.show()


plt.show()
