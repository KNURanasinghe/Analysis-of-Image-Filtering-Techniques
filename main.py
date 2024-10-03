import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse

# Function to upload the image
def upload_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        original_image = cv2.imread(filepath)  
        process_image(original_image)          

# Function to apply filtering techniques and process the image
def process_image(image):
    filter_combination = filter_var.get()  # Get the selected filter combination from the dropdown

    # Applying noise removal filters based on the selected combination
    if filter_combination == "Gaussian + Median":
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)  # Larger Gaussian Blur
        filtered_image = cv2.medianBlur(filtered_image, 5)   # Larger Median Blur
    elif filter_combination == "Bilateral Filter":
        filtered_image = cv2.bilateralFilter(image, 9, 75, 75)  # Adjusted Bilateral Filter
    elif filter_combination == "Gaussian + Bilateral":
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)  # Larger Gaussian Blur
        filtered_image = cv2.bilateralFilter(filtered_image, 9, 75, 75)  # Bilateral Filter
    else:  # Default to No filtering
        filtered_image = image

    # Now, apply sharpening to remove blur
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])  # Sharpening kernel
    sharpened_image = cv2.filter2D(filtered_image, -1, kernel)  # Apply sharpening

    # Enhance contrast (optional) to improve clarity
    sharpened_image = cv2.convertScaleAbs(sharpened_image, alpha=1.2, beta=30)  # Adjust contrast and brightness

    # Now, compare the original image with the filtered and sharpened images using different metrics
    display_details(image, filtered_image, sharpened_image)  # Compare using PSNR, SSIM, and MSE
    display_comparisons(image, filtered_image, sharpened_image)  # Visual comparison including both images and histograms

# Function to calculate and display the metrics for each image
def display_details(original, filtered, sharpened):
    # Compute PSNR, SSIM, and MSE for the original vs filtered image
    psnr_filtered = psnr(original, filtered)
    min_dimension_filtered = min(original.shape[0], original.shape[1])
    win_size_filtered = min(7, min_dimension_filtered)  # Ensure window size is appropriate for image dimensions
    ssim_filtered, _ = ssim(original, filtered, win_size=win_size_filtered, channel_axis=2, full=True)
    mse_filtered = mse(original, filtered)

    # Compute PSNR, SSIM, and MSE for the original vs sharpened image
    psnr_sharpened = psnr(original, sharpened)
    min_dimension_sharpened = min(original.shape[0], original.shape[1])
    win_size_sharpened = min(7, min_dimension_sharpened)  # Ensure window size is appropriate for image dimensions
    ssim_sharpened, _ = ssim(original, sharpened, win_size=win_size_sharpened, channel_axis=2, full=True)
    mse_sharpened = mse(original, sharpened)

    # Display the results in the GUI
    details_text = (f"Filtered Image:\n"
                    f"PSNR: {psnr_filtered:.2f} dB\n"
                    f"SSIM: {ssim_filtered:.4f}\n"
                    f"MSE: {mse_filtered:.2f}\n\n"
                    f"Sharpened Image:\n"
                    f"PSNR: {psnr_sharpened:.2f} dB\n"
                    f"SSIM: {ssim_sharpened:.4f}\n"
                    f"MSE: {mse_sharpened:.2f}")
    result_label.config(text=details_text)  # Update the label with the computed values

# Function to plot visual comparisons (original image, filtered image, and histograms)
def display_comparisons(original, filtered, sharpened):
    # Convert to grayscale for histogram comparison
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    sharpened_gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

    # Create a new figure for the comparison
    plt.figure(figsize=(16, 8))
    
    # Display the original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')  # Hide the axes

    # Display the filtered image
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Image')
    plt.axis('off')  # Hide the axes

    # Display the sharpened image
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.title('Sharpened Image')
    plt.axis('off')  # Hide the axes

    # Plot histogram of the original image
    plt.subplot(2, 3, 4)
    plt.hist(original_gray.ravel(), 256, [0, 256], color='blue', label='Original')
    plt.title('Original Image Histogram')

    # Plot histogram of the filtered image
    plt.subplot(2, 3, 5)
    plt.hist(filtered_gray.ravel(), 256, [0, 256], color='green', label='Filtered')
    plt.title('Filtered Image Histogram')

    # Plot histogram of the sharpened image
    plt.subplot(2, 3, 6)
    plt.hist(sharpened_gray.ravel(), 256, [0, 256], color='red', label='Sharpened')
    plt.title('Sharpened Image Histogram')

    # Display the plots
    plt.tight_layout()
    plt.show()

# Create the GUI window
root = tk.Tk()
root.title("Image Noise Removal and Comparison")

# Dropdown menu for filter selection
filter_var = tk.StringVar(value="No Filter")  # Default value
filter_options = ["No Filter", "Gaussian + Median", "Bilateral Filter", "Gaussian + Bilateral"]
filter_dropdown = ttk.Combobox(root, textvariable=filter_var, values=filter_options, state="readonly")
filter_dropdown.pack(pady=10)  # Add padding for better appearance

# Upload button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=10)  # Add padding for better appearance

# Result label to display metrics
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)  # Add padding for better appearance

# Run the Tkinter event loop
root.mainloop()
