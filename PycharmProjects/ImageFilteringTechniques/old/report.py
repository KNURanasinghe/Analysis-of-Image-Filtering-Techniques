import cv2
import numpy as np
import time
import psutil
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def gaussian_filter(image):
    """Apply Gaussian filter to remove noise."""
    return cv2.GaussianBlur(image, (5, 5), 0)


def median_filter(image):
    """Apply Median filter to remove noise."""
    return cv2.medianBlur(image, 5)


def bilateral_filter(image):
    """Apply Bilateral filter to remove noise."""
    return cv2.bilateralFilter(image, d=15, sigmaColor=75, sigmaSpace=75)

def non_local_means_filter(image):
    """Apply Non-Local Means filter to remove noise."""
    h = 10  # Strength of luminance denoising
    hForColorComponents = 10  # Strength of color denoising
    templateWindowSize = 7  # Size of template patch
    searchWindowSize = 21  # Size of window for patch search

    # Apply the filter with the above parameters
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, hForColorComponents, templateWindowSize, searchWindowSize
    )



def hybrid_filter(image):
    """Apply hybrid filtering using Bilateral and Non-Local Means filters."""
    bilateral_filtered = bilateral_filter(image)
    nlm_filtered = cv2.fastNlMeansDenoisingColored(bilateral_filtered, None, 10, 10, 7, 21)
    return nlm_filtered


def calculate_metrics(original, filtered):
    """Calculate PSNR, SSIM, and MSE between two images."""
    psnr_value = cv2.PSNR(original, filtered)

    # Set win_size to 3 for SSIM calculation
    win_size = 3
    ssim_value = ssim(original, filtered, multichannel=True, win_size=win_size)

    mse_value = np.mean((original.astype("float") - filtered.astype("float")) ** 2)
    return psnr_value, ssim_value, mse_value


def memory_usage():
    """Get memory usage."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB


def detect_noise(image):
    """Detect noise in the image and classify the type of noise."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    noise_report = {}

    # Gaussian Noise Detection
    std_dev = np.std(gray_image)
    if std_dev > 20:  # Adjust the threshold as needed
        noise_report['Gaussian Noise'] = std_dev

    # Salt-and-Pepper Noise Detection
    num_salt = np.sum(gray_image == 255)
    num_pepper = np.sum(gray_image == 0)
    total_pixels = gray_image.size
    if (num_salt + num_pepper) / total_pixels > 0.01:  # 1% of total pixels as noise
        noise_report['Salt-and-Pepper Noise'] = (num_salt, num_pepper)

    # Speckle Noise Detection (Example: High variance)
    variance = np.var(gray_image)
    if variance > 400:  # Adjust the threshold for your context
        noise_report['Speckle Noise'] = variance

    # Check for luminance and chromatic noise
    # Compare channel variances for color images
    channels = cv2.split(image)
    channel_variances = [np.var(channel) for channel in channels]
    if any(var > 400 for var in channel_variances):  # Adjust thresholds
        noise_report['Chromatic Noise'] = channel_variances

    return noise_report


def process_filtering(image):
    """Apply all filters and display results."""
    filtered_images = []
    titles = []
    metrics = []

    # Detect noise in the original image
    noise_report = detect_noise(image)
    print("Noise Detection Report:")
    for noise_type, details in noise_report.items():
        print(f"{noise_type}: {details}")

    # Process with different filtering methods
    filters = [
        (gaussian_filter, "Gaussian Filter"),
        (median_filter, "Median Filter"),
        (bilateral_filter, "Bilateral Filter"),
        (non_local_means_filter, "Non-Local Means Filter"),  # Added Non-Local Means filter
        (hybrid_filter, "Hybrid Filter (Bilateral + NLM)")
    ]

    for filter_func, title in filters:
        start_time = time.time()
        filtered_image = filter_func(image)
        end_time = time.time()

        psnr_value, ssim_value, mse_value = calculate_metrics(image, filtered_image)

        runtime = end_time - start_time
        memory = memory_usage()

        print(f"{title}:")
        print(f"Filtered Image Size: {filtered_image.shape}")
        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")
        print(f"MSE: {mse_value:.2f}")
        print(f"Runtime: {runtime:.4f} seconds")
        print(f"Memory Usage: {memory:.2f} MB\n")

        # Store the filtered image, title, and metrics for display
        filtered_images.append(filtered_image)
        titles.append(title)
        metrics.append([psnr_value, ssim_value, mse_value])

    # Show everything in one window
    show_all_in_one_window(image, filtered_images, titles, metrics)


def show_all_in_one_window(original_image, filtered_images, titles, metrics):
    """Display the original image, filtered images, and metrics in one window."""
    num_images = len(filtered_images) + 1  # Include the original image

    plt.figure(figsize=(15, 10))

    # Display original and filtered images in one row
    plt.subplot(3, 1, 1)  # First row for images
    plt.title("Images", fontsize=16)

    # Show all images in a single row
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)  # 1 row and num_images columns
        if i == 0:
            plt.imshow(original_image)
            plt.title("Original Image")
        else:
            plt.imshow(filtered_images[i - 1])
            plt.title(titles[i - 1])
        plt.axis("off")

    # Prepare to display metrics
    psnr_values = [metrics[i][0] for i in range(len(filtered_images))]
    mse_values = [metrics[i][2] for i in range(len(filtered_images))]
    ssim_values = [metrics[i][1] for i in range(len(filtered_images))]

    # Create combined PSNR and MSE bar chart
    plt.subplot(3, 1, 2)  # Second row for PSNR and MSE
    bar_width = 0.35
    x = np.arange(len(filtered_images))

    plt.bar(x - bar_width / 2, psnr_values, width=bar_width, label='PSNR', color='blue')
    plt.bar(x + bar_width / 2, mse_values, width=bar_width, label='MSE', color='green')
    plt.xticks(x, titles)
    plt.ylabel("Value")
    plt.title("PSNR and MSE Metrics")
    plt.legend()
    plt.ylim(0, max(max(psnr_values), max(mse_values)) * 1.1)

    # Create SSIM bar chart
    plt.subplot(3, 1, 3)  # Third row for SSIM
    plt.bar(x, ssim_values, width=bar_width, label='SSIM', color='orange')
    plt.xticks(x, titles)
    plt.ylabel("SSIM Value")
    plt.title("SSIM Metrics")
    plt.ylim(0, 1.1)  # SSIM ranges from 0 to 1

    plt.tight_layout()
    plt.show()

def main(image_path):
    """Main function to run the noise detection and filtering process."""
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    process_filtering(image)


if __name__ == "__main__":
    main('/source/noicyImg.jpg')
