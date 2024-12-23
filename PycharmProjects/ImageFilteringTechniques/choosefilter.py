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


# Load the original image
original_image = cv2.imread('source/noicyImg.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Ensure the image is large enough
if original_image.shape[0] < 7 or original_image.shape[1] < 7:
    raise ValueError("Original image is too small. Please use a larger image.")


# Define filter functions
def apply_gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def apply_bilateral(image):
    return cv2.bilateralFilter(image, 9, 75, 75)


# Process the original image with filter combinations
combined_bilateral_gaussian = apply_gaussian(apply_bilateral(original_image))
combined_gaussian_bilateral = apply_bilateral(apply_gaussian(original_image))

# Calculate metrics for both combinations
metrics_df = pd.DataFrame(index=['PSNR', 'SSIM', 'MSE'], columns=['Bilateral + Gaussian', 'Gaussian + Bilateral'])

# Calculate metrics for Bilateral + Gaussian
psnr_value, ssim_value, mse_value = calculate_metrics(original_image, combined_bilateral_gaussian)
metrics_df['Bilateral + Gaussian'] = [psnr_value, ssim_value, mse_value]

# Calculate metrics for Gaussian + Bilateral
psnr_value, ssim_value, mse_value = calculate_metrics(original_image, combined_gaussian_bilateral)
metrics_df['Gaussian + Bilateral'] = [psnr_value, ssim_value, mse_value]

# Display the metrics DataFrame
print("\nMetrics for Bilateral + Gaussian and Gaussian + Bilateral Combinations:")
print(metrics_df)

# Display filtered images
cv2.imshow("Bilateral + Gaussian", cv2.cvtColor(combined_bilateral_gaussian, cv2.COLOR_RGB2BGR))
cv2.imshow("Gaussian + Bilateral", cv2.cvtColor(combined_gaussian_bilateral, cv2.COLOR_RGB2BGR))

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
