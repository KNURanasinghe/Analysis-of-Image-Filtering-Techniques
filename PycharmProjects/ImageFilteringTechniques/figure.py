import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(original, filtered):
    # Convert images to float32 for more accurate calculations
    original = original.astype(np.float32)
    filtered = filtered.astype(np.float32)

    # Calculate MSE
    mse_value = np.mean((original - filtered) ** 2)

    # Calculate PSNR
    if mse_value == 0:  # Prevent division by zero
        psnr_value = float('inf')
    else:
        psnr_value = 20 * np.log10(255.0 / np.sqrt(mse_value))

    # Calculate SSIM
    ssim_value = ssim(original, filtered, multichannel=True)

    return psnr_value, ssim_value, mse_value

# Load the original image
original_image_path = 'source/noicyImg.jpg'  # Replace with your image path
original = cv2.imread(original_image_path)

# Apply different filters
filters = {
    'Gaussian Filter': cv2.GaussianBlur(original, (5, 5), 0),
    'Median Filter': cv2.medianBlur(original, 5),
    'Bilateral Filter': cv2.bilateralFilter(original, d=9, sigmaColor=75, sigmaSpace=75),
    'Non-Local Means Filter': cv2.fastNlMeansDenoisingColored(original, None, 10, 10, 21)
}

# Initialize a results list
results = []

# Calculate metrics for each filtered image
for filter_name, filtered_image in filters.items():
    psnr_value, ssim_value, mse_value = calculate_metrics(original, filtered_image)
    results.append({
        'Filter Type': filter_name,
        'PSNR (dB)': psnr_value,
        'SSIM': ssim_value,
        'MSE': mse_value
    })

# Create a DataFrame from the results
import pandas as pd
results_df = pd.DataFrame(results)

# Display the results
print("Metric Evaluation Matrix for Individual Filters:")
print(results_df)

# Optionally, plot the results
import matplotlib.pyplot as plt

# Set the index to 'Filter Type' and transpose the DataFrame
results_df.set_index('Filter Type', inplace=True)
results_df_transposed = results_df.transpose()

# Plotting the metrics
ax = results_df_transposed.plot(kind='bar', figsize=(10, 6))
plt.title('Metric Evaluation for Individual Filters')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.legend(title='Filter Type')
plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()
