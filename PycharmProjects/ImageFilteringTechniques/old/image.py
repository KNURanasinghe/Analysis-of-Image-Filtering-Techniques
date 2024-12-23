import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image with noise
image_path = '../source/OIP.jpeg'  # Replace with your image path
img = cv2.imread(image_path)

# Convert to grayscale (if needed)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the original noisy image
cv2.imshow("Original Image", gray_img)

# 1. Gaussian Blur (for Gaussian noise)
gaussian_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
cv2.imshow("Gaussian Blur", gaussian_blur)

# 2. Median Filtering (for salt-and-pepper noise)
median_filtered = cv2.medianBlur(gray_img, 5)
cv2.imshow("Median Filter", median_filtered)

# 3. Bilateral Filtering (preserves edges)
bilateral_filtered = cv2.bilateralFilter(gray_img, 9, 75, 75)
cv2.imshow("Bilateral Filter", bilateral_filtered)

# 4. Fast Non-Local Means Denoising (for strong noise reduction)
denoised_image = cv2.fastNlMeansDenoising(gray_img, None, 30, 7, 21)
cv2.imshow("Denoised Image", denoised_image)

# Display all images in a single window using matplotlib (Optional)
plt.subplot(2, 2, 1), plt.imshow(gray_img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(gaussian_blur, cmap='gray')
plt.title('Gaussian Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(median_filtered, cmap='gray')
plt.title('Median Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(bilateral_filtered, cmap='gray')
plt.title('Bilateral Filter'), plt.xticks([]), plt.yticks([])

plt.show()

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
