import cv2
import numpy as np
from skimage import img_as_ubyte, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
import time
import matplotlib.pyplot as plt


# Function to apply Bilateral + Non-Local Means Denoising (Hybrid 1)
def apply_hybrid_filter_1(img, iterations=1):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    for i in range(iterations):
        # Apply Bilateral Filter
        img = cv2.bilateralFilter(img, 9, 75, 75)
        # Apply Non-Local Means Denoising
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    return img


# Function to apply Gaussian + Median Filtering (Hybrid 2)
def apply_hybrid_filter_2(img, iterations=1):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    for i in range(iterations):
        # Apply Gaussian Blur
        img = cv2.GaussianBlur(img, (9, 9), 0)
        # Apply Median Filter
        img = cv2.medianBlur(img, 5)

        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    return img


# Evaluation Metrics
def evaluate_metrics(original, filtered):
    original = img_as_ubyte(original)
    filtered = img_as_ubyte(filtered)

    # PSNR
    psnr_value = psnr(original, filtered)
    # SSIM
    ssim_value = ssim(original, filtered, win_size=7, channel_axis=-1)
    # MSE
    mse_value = mse(original, filtered)

    return psnr_value, ssim_value, mse_value


# Performance Measurement
def measure_performance(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time


# Load Image
img_path = '../source/noicyImg.jpg'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("Image file not found!")

img = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Apply Filters and Measure Performance
iterations_1 = 5  # Iterations for Hybrid 1
iterations_2 = 3  # Iterations for Hybrid 2

filtered_img_1, filter_time_1 = measure_performance(apply_hybrid_filter_1, img, iterations_1)
filtered_img_2, filter_time_2 = measure_performance(apply_hybrid_filter_2, img, iterations_2)

# Calculate Metrics for Hybrid 1
psnr_val_1, ssim_val_1, mse_val_1 = evaluate_metrics(img, filtered_img_1)

# Calculate Metrics for Hybrid 2
psnr_val_2, ssim_val_2, mse_val_2 = evaluate_metrics(img, filtered_img_2)

# Display Results for Hybrid 1
print(f"Hybrid 1 (Bilateral + Non-Local Means) - Iterations: {iterations_1}")
print(f"PSNR: {psnr_val_1}, SSIM: {ssim_val_1}, MSE: {mse_val_1}")
print(f"Filtering Time: {filter_time_1} seconds")

# Display Results for Hybrid 2
print(f"Hybrid 2 (Gaussian + Median) - Iterations: {iterations_2}")
print(f"PSNR: {psnr_val_2}, SSIM: {ssim_val_2}, MSE: {mse_val_2}")
print(f"Filtering Time: {filter_time_2} seconds")

# Show Original, and Filtered Images for Comparison
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(filtered_img_1)
ax[1].set_title(f'Hybrid 1 (Bilateral+NLM), Iter: {iterations_1}')
ax[2].imshow(filtered_img_2)
ax[2].set_title(f'Hybrid 2 (Gaussian+Median), Iter: {iterations_2}')
for a in ax:
    a.axis('off')
plt.show()
