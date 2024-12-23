import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Function to calculate PSNR, SSIM, and MSE
def calculate_metrics(original, filtered):
    """Calculate PSNR, SSIM, and MSE between two images."""
    psnr_value = cv2.PSNR(original, filtered)

    # Set win_size to 3 for SSIM calculation
    win_size = 3
    ssim_value = ssim(original, filtered, multichannel=True, win_size=win_size)

    mse_value = np.mean((original.astype("float") - filtered.astype("float")) ** 2)
    return psnr_value, ssim_value, mse_value

# Load your original image
original_image = cv2.imread('source/noicyImg.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Print original image dimensions
print(f"Original image dimensions: {original_image.shape}")

# Ensure the image is large enough
if original_image.shape[0] < 7 or original_image.shape[1] < 7:
    raise ValueError("Original image is too small. Please use a larger image.")

# Define filter functions
def apply_gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_median(image):
    return cv2.medianBlur(image, 5)

def apply_bilateral(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def apply_non_local_means(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Create a dictionary to hold the filtered images
filters = {
    'gaussian': apply_gaussian(original_image),
    'median': apply_median(original_image),
    'bilateral': apply_bilateral(original_image),
    'non-LM': apply_non_local_means(original_image),

}

# Print filtered images dimensions
for key, filtered_image in filters.items():
    print(f"Filtered image {key} dimensions: {filtered_image.shape}")

# Initialize DataFrames for metrics
metrics_psnr_df = pd.DataFrame(index=filters.keys(), columns=filters.keys())
metrics_ssim_df = pd.DataFrame(index=filters.keys(), columns=filters.keys())
metrics_mse_df = pd.DataFrame(index=filters.keys(), columns=filters.keys())

# Calculate metrics for all combinations of filters
for filter1 in filters:
    for filter2 in filters:
        combined_filtered_image = cv2.addWeighted(filters[filter1], 0.5, filters[filter2], 0.5, 0)
        psnr_value, ssim_value, mse_value = calculate_metrics(original_image, combined_filtered_image)

        # Fill the DataFrames with calculated metrics
        metrics_psnr_df.at[filter1, filter2] = psnr_value
        metrics_ssim_df.at[filter1, filter2] = ssim_value
        metrics_mse_df.at[filter1, filter2] = mse_value

# Display the metrics DataFrames
print("\nPSNR Values for All Combinations:")
print(metrics_psnr_df)

print("\nSSIM Values for All Combinations:")
print(metrics_ssim_df)

print("\nMSE Values for All Combinations:")
print(metrics_mse_df)

# Create a summary table with overall metrics
summary_metrics_df = pd.DataFrame(index=['PSNR', 'SSIM', 'MSE'], columns=filters.keys())

# Fill in the summary table
for filter1 in filters:
    summary_metrics_df.at['PSNR', filter1] = metrics_psnr_df[filter1].mean()
    summary_metrics_df.at['SSIM', filter1] = metrics_ssim_df[filter1].mean()
    summary_metrics_df.at['MSE', filter1] = metrics_mse_df[filter1].mean()

# Display the summary metrics DataFrame
print("\nSummary of Metrics:")
print(summary_metrics_df)




