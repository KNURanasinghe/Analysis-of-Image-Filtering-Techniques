import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse, structural_similarity as ssim
from tkinter import filedialog, Tk


# Function to calculate PSNR, MSE, SSIM
# Updated function to calculate PSNR, MSE, SSIM
def calculate_metrics(original, filtered):
    psnr_value = psnr(original, filtered)
    mse_value = mse(original, filtered)

    # Set window size (e.g., 7) and channel_axis for color images
    ssim_value, _ = ssim(original, filtered, full=True, multichannel=True, win_size=7, channel_axis=2)

    return psnr_value, mse_value, ssim_value


# Function to apply filters
def apply_filters(image):
    filters = {}

    # Gaussian Blur
    filters['Gaussian Blur'] = cv2.GaussianBlur(image, (5, 5), 0)

    # Median Blur
    filters['Median Blur'] = cv2.medianBlur(image, 5)

    # Bilateral Blur
    filters['Bilateral Blur'] = cv2.bilateralFilter(image, 9, 75, 75)

    # Box Filter
    filters['Box Filter'] = cv2.boxFilter(image, -1, (5, 5))

    # Simple Blur
    filters['Blur Filter'] = cv2.blur(image, (5, 5))

    # Non-local Means Denoising
    filters['Non-Local Means Denoising'] = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return filters


# Function to plot graph
def plot_metrics(metrics):
    filters = list(metrics.keys())
    psnr_values = [metrics[f][0] for f in filters]
    mse_values = [metrics[f][1] for f in filters]
    ssim_values = [metrics[f][2] for f in filters]

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.bar(filters, psnr_values, color='blue')
    plt.title('PSNR Values (high value is better)')
    plt.xticks(rotation=45, ha="right")

    plt.subplot(1, 3, 2)
    plt.bar(filters, mse_values, color='green')
    plt.title('MSE Values (low values is better)')
    plt.xticks(rotation=45, ha="right")

    plt.subplot(1, 3, 3)
    plt.bar(filters, ssim_values, color='red')
    plt.title('SSIM Values (higher is better) ')
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


# Main Function
def main():
    # Open a file dialog to upload an image
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    if not file_path:
        print("No file selected!")
        return

    # Read the image
    image = cv2.imread(file_path)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply filters
    filters = apply_filters(image)

    # Calculate metrics for each filter
    metrics = {}
    for filter_name, filtered_image in filters.items():
        metrics[filter_name] = calculate_metrics(image, filtered_image)

    # Plot the metrics
    plot_metrics(metrics)


if __name__ == "__main__":
    main()
