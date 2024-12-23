import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt




# Load the original image
original = cv2.imread('../source/noicyImg.jpg')
#############
# # Step 1: Median Blur for noise reduction
# median_blurred = cv2.medianBlur(original, 5)
#
# # Step 2: Bilateral Filter for edge-preserving smoothing
# bilateral = cv2.bilateralFilter(median_blurred, 9, 75, 75)
#
# # Step 3: Unsharp Masking for sharpening
# gaussian_blur = cv2.GaussianBlur(bilateral, (5, 5), 0)
# filtered = cv2.addWeighted(bilateral, 1.5, gaussian_blur, -0.5, 0)
##############
denoised = cv2.fastNlMeansDenoisingColored(original, None, 10, 10, 7, 21)

# Step 2: Bilateral Filter for further noise reduction while preserving edges
filtered = cv2.bilateralFilter(denoised, 9, 75, 75)

denoised1 = cv2.fastNlMeansDenoisingColored(original, None, 10, 10, 7, 21)
filtered1 = cv2.bilateralFilter(denoised, 9, 75, 75)
# Step 3: Gaussian Blur to smooth out remaining minor noise
#filtered = cv2.GaussianBlur(bilateral_filtered, (5, 5), 0)

################




# Check if images are loaded
if original is None or filtered is None:
    print("Error: Could not load images.")
    exit()

# Function to calculate PSNR
def psnr(original, filtered):
    mse_value = np.mean((original - filtered) ** 2)
    if mse_value == 0:  # Images are identical
        return float('inf')
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse_value))

# Function to calculate MSE
def mse(original, filtered):
    return np.mean((original - filtered) ** 2)

# Function to calculate SSIM
def compute_ssim(original, filtered):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    return ssim(original_gray, filtered_gray)

# Function to compare histograms
def compare_histograms(original, filtered):
    original_hist = cv2.calcHist([original], [0], None, [256], [0, 256])
    filtered_hist = cv2.calcHist([filtered], [0], None, [256], [0, 256])

    # Plot histograms for both images
    plt.plot(original_hist, label='Original Image', color='blue')
    plt.plot(filtered_hist, label='Filtered Image', color='green')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histogram Comparison')

# Function to calculate the difference image
def difference_image(original, filtered):
    diff = cv2.absdiff(original, filtered)
    return diff

def histogram_analysis(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    mean_intensity = np.mean(hist)
    std_deviation = np.std(hist)
    return mean_intensity, std_deviation

# Analyze histograms for original and filtered images
mean_original, std_original = histogram_analysis(original)
mean_filtered, std_filtered = histogram_analysis(filtered)

print(f"Original Image - Mean Intensity: {mean_original:.2f}, Std Dev: {std_original:.2f}")
print(f"Filtered Image - Mean Intensity: {mean_filtered:.2f}, Std Dev: {std_filtered:.2f}")


# Calculate PSNR, MSE, and SSIM
psnr_value = psnr(original, filtered)
mse_value = mse(original, filtered)
ssim_value = compute_ssim(original, filtered)

# Print the results
print(f"PSNR: {psnr_value:.2f} dB")
print(f"MSE: {mse_value:.2f}")
print(f"SSIM: {ssim_value:.4f}")

# Prepare figure to show all images and histograms
plt.figure(figsize=(12, 8))

# Display original image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Display filtered image
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
plt.title('Filtered Image')
plt.axis('off')



# Display filtered image
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(filtered1, cv2.COLOR_BGR2RGB))
plt.title('reFiltered Image')
plt.axis('off')

# Display histogram comparison
#plt.subplot(2, 2, 3)
#compare_histograms(original, filtered)



# Display PSNR, MSE, and SSIM values
plt.subplot(2, 2, 4)
plt.axis('off')  # Turn off axes for the text display
text_str = f"PSNR: {psnr_value:.2f} dB\nMSE: {mse_value:.2f}\nSSIM: {ssim_value:.4f}"
plt.text(0.5, 0.5, text_str, fontsize=14, va='center', ha='center', bbox=dict(facecolor='white', alpha=0.5))
plt.title("Image Quality Metrics")

# Display difference image
#diff_image = difference_image(original, filtered)
#plt.subplot(2, 2, 4)
#plt.imshow(cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB))
#plt.title('Difference Image')
#plt.axis('off')

# Show the complete figure
plt.tight_layout()
plt.show()
