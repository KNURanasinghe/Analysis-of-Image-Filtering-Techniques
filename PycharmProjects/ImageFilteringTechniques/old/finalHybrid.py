import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.restoration import estimate_sigma
from scipy.fftpack import fft2, fftshift
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse, structural_similarity as ssim

# Noise Detection
def detect_noise(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Luminance Noise
    luminance_var = np.var(gray_image)

    # Chromatic Noise
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    chromatic_var = np.var(red_channel) + np.var(green_channel) + np.var(blue_channel)

    # Banding Noise
    f_transform = fftshift(fft2(gray_image))
    f_transform_magnitude = np.abs(f_transform)
    banding_detected = np.max(f_transform_magnitude) > 1e6

    # Shot Noise
    sigma_r = estimate_sigma(red_channel, average_sigmas=True)
    sigma_g = estimate_sigma(green_channel, average_sigmas=True)
    sigma_b = estimate_sigma(blue_channel, average_sigmas=True)
    shot_noise_detected = (sigma_r + sigma_g + sigma_b) / 3 > 0.5

    return luminance_var, chromatic_var, banding_detected, shot_noise_detected

# Noise Removal Based on Detected Noise
def remove_noise(image, luminance_var, chromatic_var, banding_detected, shot_noise_detected):
    filtered_image = image

    # Luminance Noise
    if luminance_var > 1000:
        filtered_image1 = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        filtered_image = cv2.fastNlMeansDenoisingColored(filtered_image1, None, h=10, templateWindowSize=7,
                                                         searchWindowSize=21)

    # Chromatic Noise
    if chromatic_var > 10000:
        filtered_image1 = cv2.fastNlMeansDenoisingColored(filtered_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        filtered_image = cv2.bilateralFilter(filtered_image1, d=9, sigmaColor=75, sigmaSpace=75)

    # Banding Noise
    if banding_detected:
        print("Banding noise detected - custom frequency filtering required")

    # Shot Noise
    if shot_noise_detected:
        filtered_image = cv2.fastNlMeansDenoisingColored(filtered_image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    return filtered_image

# Calculate PSNR, MSE, SSIM
def calculate_metrics(original, filtered):
    psnr_value = psnr(original, filtered)
    mse_value = mse(original, filtered)
    ssim_value = ssim(original, filtered, win_size=3, channel_axis=-1)
    return psnr_value, mse_value, ssim_value

# Function to plot bar charts for PSNR, MSE, SSIM
def plot_metrics(psnr_value, mse_value, ssim_value):
    metrics = ['PSNR', 'MSE', 'SSIM']
    original_values = [psnr_value, mse_value, ssim_value]
    filtered_values = [psnr_value, mse_value, ssim_value]

    # Bar chart for original and filtered image metrics
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Original Image Metrics
    ax[0].bar(metrics, original_values, color=['blue', 'orange', 'green'])
    ax[0].set_title('Original Image Metrics')
    ax[0].set_ylim([0, max(original_values) + 10])

    # Plot Filtered Image Metrics
    ax[1].bar(metrics, filtered_values, color=['blue', 'orange', 'green'])
    ax[1].set_title('Filtered Image Metrics')
    ax[1].set_ylim([0, max(filtered_values) + 10])

    plt.show()

# Main function
if __name__ == "__main__":
    image_path = '/source/OIP.jpeg'

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Failed to load the image. Please check the file path: {image_path}")

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Detect Noise Types
    luminance_var, chromatic_var, banding_detected, shot_noise_detected = detect_noise(original_image)

    print(f"Luminance Noise Variance: {luminance_var:.2f}")
    print(f"Chromatic Noise Variance: {chromatic_var:.2f}")
    print(f"Banding Detected: {banding_detected}")
    print(f"Shot Noise Detected: {shot_noise_detected}")

    # Apply noise removal based on detection
    filtered_image = remove_noise(original_image, luminance_var, chromatic_var, banding_detected, shot_noise_detected)

    # Calculate metrics
    psnr_value, mse_value, ssim_value = calculate_metrics(original_image, filtered_image)

    print(f"PSNR: {psnr_value:.2f}")
    print(f"MSE: {mse_value:.2f}")
    print(f"SSIM: {ssim_value:.2f}")

    # Plot metrics bar chart
    plot_metrics(psnr_value, mse_value, ssim_value)
