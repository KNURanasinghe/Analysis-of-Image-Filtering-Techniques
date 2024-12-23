import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Function to calculate PSNR
def calculate_psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')  # No difference between images
    pixel_max = 255.0
    psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    return psnr

# Function to calculate MSE
def calculate_mse(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    return mse

# Load the original and noisy images
original_image = cv2.imread('../source/noicyImg.jpg', cv2.IMREAD_GRAYSCALE)
noisy_image = cv2.GaussianBlur(original_image, (5, 5), 0)  # Simulating noise for the comparison

# Apply different noise removal techniques
gaussian_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 0)
median_filtered = cv2.medianBlur(noisy_image, 5)
bilateral_filtered = cv2.bilateralFilter(noisy_image, 9, 75, 75)

# Store results for each technique
techniques = ['Original Image', 'Gaussian Blur', 'Median Filter', 'Bilateral Filter']
psnr_values = []
ssim_values = []
mse_values = []

# Calculate PSNR, SSIM, and MSE for the original noisy image
psnr_values.append(calculate_psnr(original_image, noisy_image))
ssim_value, _ = ssim(original_image, noisy_image, full=True)
ssim_values.append(ssim_value)
mse_values.append(calculate_mse(original_image, noisy_image))

# Calculate PSNR, SSIM, and MSE for each filtering technique
for denoised_image in [gaussian_filtered, median_filtered, bilateral_filtered]:
    psnr_values.append(calculate_psnr(original_image, denoised_image))
    ssim_value, _ = ssim(original_image, denoised_image, full=True)
    ssim_values.append(ssim_value)
    mse_values.append(calculate_mse(original_image, denoised_image))

# Plot PSNR, SSIM, and MSE for comparison
x = np.arange(len(techniques))

fig, ax1 = plt.subplots()

# Bar plot for PSNR and MSE
bar_width = 0.35
ax1.bar(x - bar_width/2, psnr_values, bar_width, label='PSNR', color='b')
ax1.bar(x + bar_width/2, mse_values, bar_width, label='MSE', color='r')

# Create another axis for SSIM
ax2 = ax1.twinx()
ax2.plot(x, ssim_values, label='SSIM', color='g', marker='o', linewidth=2)

# Set labels and titles
ax1.set_xlabel('Filtering Technique')
ax1.set_ylabel('PSNR / MSE')
ax2.set_ylabel('SSIM')
ax1.set_title('Comparison of Image Quality Metrics (Including Original Image)')
ax1.set_xticks(x)
ax1.set_xticklabels(techniques)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show plot
plt.show()
