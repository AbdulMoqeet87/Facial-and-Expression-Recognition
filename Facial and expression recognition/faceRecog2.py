import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
all_images = []
main_folder = r"E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces"

for subject in range(1, 16):
    subject_folder = os.path.join(main_folder, f"subject{subject}")
    subject_images = []

    for img_num in range(1, 11):
        img_path = os.path.join(subject_folder, f"subject{subject}.{img_num}.png")

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                subject_images.append(img)
            else:
                print(f"Error loading image at path: {img_path}. Image is None.")
        except Exception as e:
            print(f"Error loading image at path: {img_path}")
            print(e)
        
    # Append the list of images for the current subject to all_images
    all_images.append(subject_images)


# Flatten and resize images
images_resized = [
    [cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR) for img in subject_images]
    for subject_images in all_images
]
#all data matrices and mean vectors
data_matrices = [np.vstack([img.flatten() for img in subject_images]) for subject_images in images_resized]
all_mean_vectors = [np.mean(data_matrix, axis=0) for data_matrix in data_matrices]
centered_data_matrices = [data_matrix - mean_vector for data_matrix, mean_vector in zip(data_matrices, all_mean_vectors)]


# Center the data by subtracting the mean
covariance_matrices = [np.cov(centered_data_matrix, rowvar=False) for centered_data_matrix in centered_data_matrices]
# Compute the covariance matrix for centered data
# Compute eigenvalues and eigenvectors
eigenvalue_lists = []
eigenvector_lists = []

for covariance_matrix in covariance_matrices:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalue_lists.append(eigenvalues)
    eigenvector_lists.append(eigenvectors)



# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices_lists = [np.argsort(eigenvalues)[::-1] for eigenvalues in eigenvalue_lists]
for i in range(len(eigenvalue_lists)):
    eigenvalue_lists[i] = eigenvalue_lists[i][sorted_indices_lists[i]]
    eigenvector_lists[i] = eigenvector_lists[i][:, sorted_indices_lists[i]]


# Create the unit eigen matrix
cumulative_energy_list = [np.cumsum(eigenvalues) / np.sum(eigenvalues) for eigenvalues in eigenvalue_lists]

num_components_list = [np.argmax(cumulative_energy >= 0.99) + 1 for cumulative_energy in cumulative_energy_list]


selected_eigenvectors_list = [eigenvectors[:, :num_components] for eigenvectors, num_components in zip(eigenvector_lists, num_components_list)]

# Multiply the original data matrix with the unit eigen matrix

test_image_path = r'E:\3rd semester\Linear Algebra\Python_Assignment\testimages\testimage9.png'  # Replace with the path to your test image
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
test_image_resized = cv2.resize(test_image, (50, 50), interpolation=cv2.INTER_LINEAR)
test_flattened = test_image_resized.flatten()
centered_test_list = [test_flattened - mean_vector for mean_vector in all_mean_vectors]

# Project the centered test image onto the selected eigenvectors
projected_test_list = [np.dot(centered_test, selected_eigenvectors) for centered_test, selected_eigenvectors in zip(centered_test_list, selected_eigenvectors_list)]



# Back-project the projected test image to the original space
back_projected_test_list = [np.dot(projected_test, selected_eigenvectors.T) + mean_vector for projected_test, selected_eigenvectors,mean_vector in zip(projected_test_list,selected_eigenvectors_list,all_mean_vectors)]


# Reshape the back-projected test image to its original shape\
for i in range(len(back_projected_test_list)):
    back_projected_test_list[i] = back_projected_test_list[i].reshape(test_image_resized.shape)
    

# Display the original, projected, and back-projected test images



plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(test_image_resized, cmap='gray')
plt.title("Original Test Image")


mse_list = [np.mean((test_image_resized - back_projected_test) ** 2) for  back_projected_test in back_projected_test_list]
min_mse_index = np.argmin(mse_list)
min_mse_value = mse_list[min_mse_index]
print("min loss",min_mse_value)
print(f'predicted to be subject{min_mse_index+1}')
plt.subplot(1, 3, 3)
plt.imshow(back_projected_test_list[min_mse_index], cmap='gray')
plt.title("Back-Projected Test Image")



plt.show()
