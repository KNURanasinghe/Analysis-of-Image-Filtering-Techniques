import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# Function to calculate PSNR, SSIM, and MSE
def calculate_metrics(original, filtered):
    """Calculate PSNR, SSIM, and MSE between two images."""
    psnr_value = psnr(original, filtered)

    # Set win_size to 3 for SSIM calculation
    win_size = 3
    ssim_value = ssim(original, filtered, multichannel=True, win_size=win_size)

    mse_value = np.mean((original.astype("float") - filtered.astype("float")) ** 2)
    return psnr_value, ssim_value, mse_value


# Load the original image
original_image = cv2.imread('source/noicyImg.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Ensure the image is large enough
if original_image.shape[0] < 7 or original_image.shape[1] < 7:
    raise ValueError("Original image is too small. Please use a larger image.")


# Define individual filter functions
def apply_median(image):
    return cv2.medianBlur(image, 5)


def apply_gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def apply_bilateral(image):
    return cv2.bilateralFilter(image, 9, 75, 75)


def apply_non_local_means(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


# Define combination functions
def apply_non_local_then_median(image):
    """Apply Non-Local Means filter first, then Median filter."""
    non_local_filtered = apply_non_local_means(image)
    combined_filtered = apply_median(non_local_filtered)
    return combined_filtered


def apply_bilateral_then_gaussian(image):
    """Apply Bilateral filter first, then Gaussian filter."""
    bilateral_filtered = apply_bilateral(image)
    combined_filtered = apply_gaussian(bilateral_filtered)
    return combined_filtered


def apply_bilateral_then_non_local(image):
    """Apply Bilateral filter first, then Non-Local Means filter."""
    bilateral_filtered = apply_bilateral(image)
    combined_filtered = apply_non_local_means(bilateral_filtered)
    return combined_filtered


# Apply filters
combined_bilateral_then_gaussian = apply_bilateral_then_gaussian(original_image)
gaussian_filtered = apply_gaussian(original_image)
median_filtered = apply_median(original_image)
bilateral_filtered = apply_bilateral(original_image)
non_local_filtered = apply_non_local_means(original_image)


# Display each filtered image in a window
cv2.imshow('Original Image', original_image)
cv2.imshow('Gaussian Filtered', gaussian_filtered)
cv2.imshow('Median Filtered', median_filtered)
cv2.imshow('Bilateral Filtered', bilateral_filtered)
cv2.imshow('Non-Local Means Filtered', non_local_filtered)
cv2.imshow('Bilateral then Gaussian', combined_bilateral_then_gaussian)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Initialize a DataFrame to store the metric values
metrics_df = pd.DataFrame(index=['PSNR', 'SSIM', 'MSE'],
                          columns=['Bilateral + Gaussian', 'Gaussian', 'Median', 'Bilateral', 'Non-Local'])

# Calculate metrics for each combination and store in the DataFrame
psnr_value, ssim_value, mse_value = calculate_metrics(original_image, combined_bilateral_then_gaussian)
metrics_df['Bilateral + Gaussian'] = [psnr_value, ssim_value, mse_value]

psnr_value, ssim_value, mse_value = calculate_metrics(original_image, gaussian_filtered)
metrics_df['Gaussian'] = [psnr_value, ssim_value, mse_value]

psnr_value, ssim_value, mse_value = calculate_metrics(original_image, median_filtered)
metrics_df['Median'] = [psnr_value, ssim_value, mse_value]

psnr_value, ssim_value, mse_value = calculate_metrics(original_image, bilateral_filtered)
metrics_df['Bilateral'] = [psnr_value, ssim_value, mse_value]

psnr_value, ssim_value, mse_value = calculate_metrics(original_image, non_local_filtered)
metrics_df['Non-Local'] = [psnr_value, ssim_value, mse_value]

# Display the metrics DataFrame
print("\nMetrics for Filters:")
print(metrics_df)
