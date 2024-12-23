import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import Label
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from scipy.signal import wiener

# Function to calculate MSE
def calculate_mse(original, filtered):
    return np.mean((original - filtered) ** 2)

# Function to calculate PSNR
def calculate_psnr(original, filtered):
    mse = calculate_mse(original, filtered)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Function to calculate SSIM
def calculate_ssim(original, filtered):
    ssim_value, _ = ssim(original, filtered, full=True)
    return ssim_value

# Function to calculate SNR
def calculate_snr(original, filtered):
    signal = np.mean(original)
    noise = np.std(original - filtered)
    return signal / noise

# Function to calculate Entropy
def calculate_entropy(image):
    return shannon_entropy(image)

# Function to calculate CAD
def calculate_cad(original, filtered):
    return np.sum(np.abs(original - filtered))

# Function to upload image
def upload_image():
    global img, img_display
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        img_display = ImageTk.PhotoImage(image=Image.fromarray(img))
        img_label.config(image=img_display)
        img_label.image = img_display

# Function to apply the selected combination of filters
def apply_filter():
    if img is None:
        return

    selected_filter = filter_combobox.get()

    # Apply different filter combinations based on the selection
    if selected_filter == "Gaussian with Median":
        gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 0)
        result = cv2.medianBlur(gaussian_filtered, 5)

    elif selected_filter == "Median with Gaussian":
        median_filtered = cv2.medianBlur(img, 5)
        result = cv2.GaussianBlur(median_filtered, (5, 5), 0)

    elif selected_filter == "Bilateral with Gaussian":
        bilateral_filtered = cv2.bilateralFilter(img, 9, 75, 75)
        result = cv2.GaussianBlur(bilateral_filtered, (5, 5), 0)

    elif selected_filter == "Gaussian with Bilateral":
        gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 0)
        result = cv2.bilateralFilter(gaussian_filtered, 9, 75, 75)

    elif selected_filter == "Non-Local Means Denoising":
        result = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)

    elif selected_filter == "Wiener Filter":
        result = wiener(img, (5, 5))  # Apply Wiener filter with a 5x5 kernel
        result = np.uint8(result)  # Convert the result back to 8-bit

    # Display filtered image
    filtered_image_display = ImageTk.PhotoImage(image=Image.fromarray(result))
    filtered_label.config(image=filtered_image_display)
    filtered_label.image = filtered_image_display

    # Display comparison metrics
    show_comparison_metrics(img, result)

    # Plot graph to compare filters
    compare_filters(img, result)

# Function to show comparison metrics
def show_comparison_metrics(original, filtered):
    mse_value = calculate_mse(original, filtered)
    psnr_value = calculate_psnr(original, filtered)
    ssim_value = calculate_ssim(original, filtered)
    snr_value = calculate_snr(original, filtered)
    entropy_original = calculate_entropy(original)
    entropy_filtered = calculate_entropy(filtered)
    cad_value = calculate_cad(original, filtered)

    metrics_text = (f"MSE: {mse_value:.2f}\nPSNR: {psnr_value:.2f} dB\n"
                    f"SSIM: {ssim_value:.4f}\nSNR: {snr_value:.2f}\n"
                    f"Entropy (Original): {entropy_original:.2f}\n"
                    f"Entropy (Filtered): {entropy_filtered:.2f}\n"
                    f"CAD: {cad_value:.2f}")

    metrics_label.config(text=metrics_text)

# Function to compare filters using graph
def compare_filters(original, filtered):
    plt.figure(figsize=(10, 5))

    # Original image histogram
    plt.subplot(121)
    plt.hist(original.ravel(), bins=256, color='black', alpha=0.7, label='Original')
    plt.legend()

    # Filtered image histogram
    plt.subplot(122)
    plt.hist(filtered.ravel(), bins=256, color='blue', alpha=0.7, label='Filtered')
    plt.legend()

    plt.show()

# GUI setup
root = tk.Tk()
root.title("Image Noise Reduction GUI")

# Upload button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack()

# Display area for original image
img_label = Label(root)
img_label.pack()

# Dropdown menu for selecting filter combination
filter_combobox = ttk.Combobox(root, values=[
    "Gaussian with Median",
    "Median with Gaussian",
    "Bilateral with Gaussian",
    "Gaussian with Bilateral",
    "Non-Local Means Denoising",
    "Wiener Filter"
])
filter_combobox.set("Select Filter Combination")
filter_combobox.pack()

# Apply filter button
filter_btn = tk.Button(root, text="Apply Filters", command=apply_filter)
filter_btn.pack()

# Display area for filtered image
filtered_label = Label(root)
filtered_label.pack()

# Label to display comparison metrics
metrics_label = tk.Label(root, text="", justify="left")
metrics_label.pack()

# Start the GUI loop
root.mainloop()
